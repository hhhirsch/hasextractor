#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAS Extractor - Clean v1
- Liest Excel
- Extrahiert Text aus PDFs (pdfminer.six)
- Nutzt YAML-Pattern (targets / patterns / stop_after / forbid)
- Schreibt Treffer in neue Spalten der Excel

CLI:
  python has_extractor_clean_v1.py --pdf-root "C:\\pfad\\zu\\pdfs" --excel-in "..\\data\\input.xlsx" --excel-out "..\\data\\output.xlsx" --config "..\\config\\has_patterns.yaml"
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

logger = logging.getLogger("has_extractor_clean")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------- Regex Utilities ----------

def _compile_many(patterns: List[str], flags: int) -> List[re.Pattern]:
    compiled: List[re.Pattern] = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, flags))
        except re.error as e:
            logger.warning("Regex-Fehler in Pattern %r: %s", p, e)
    return compiled

class TargetSpec:
    def __init__(self, name: str, spec: Dict):
        self.name = name
        self.patterns_raw: List[str] = spec.get("patterns", []) or []
        self.stop_after_raw: List[str] = spec.get("stop_after", []) or []
        self.forbid_raw: List[str] = spec.get("forbid", []) or []

        # Header/Marker: multiline, case-insensitive
        flags = re.IGNORECASE | re.MULTILINE
        self.patterns = _compile_many(self.patterns_raw, flags)
        self.stop_after = _compile_many(self.stop_after_raw, flags)
        # forbid kann auch auf Inhalt angewandt werden, also ignorecase (nicht multiline nötig)
        self.forbid = _compile_many(self.forbid_raw, re.IGNORECASE)

    def __repr__(self) -> str:
        return f"TargetSpec({self.name}, {len(self.patterns)} pat, {len(self.stop_after)} stop, {len(self.forbid)} forbid)"


def load_config(yaml_path: Path) -> Dict[str, TargetSpec]:
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "targets" not in config:
        raise ValueError("YAML-Konfiguration muss den Schlüssel 'targets' enthalten.")
    targets = {}
    for k, v in config["targets"].items():
        targets[k] = TargetSpec(k, v or {})
    return targets

# ---------- PDF Text ----------

def read_pdf_text(pdf_path: Path) -> str:
    """
    Holt den zusammenhängenden Text aus einem PDF.
    LAParams: leichte Wort-/Zeilenfusion, ohne aggressive Spaltenlogik.
    """
    laparams = LAParams(
        char_margin=2.0,
        line_margin=0.3,
        word_margin=0.1,
        boxes_flow=None  # None => neutraler Fluss
    )
    try:
        text = extract_text(str(pdf_path), laparams=laparams)
    except Exception as e:
        logger.error("PDF-Fehler bei %s: %s", pdf_path, e)
        text = ""
    # Normalisieren: BOM/CR etc. entschärfen
    text = text.replace("\r", "")
    return text

# ---------- Section Extraction ----------

def find_first(text: str, pattern: re.Pattern) -> Optional[re.Match]:
    m = pattern.search(text)
    return m

def find_earliest_stop(text: str, start_idx: int, stops: List[re.Pattern]) -> int:
    """
    Finde das früheste Eintreten eines Stop-Musters nach start_idx.
    Wenn keines gefunden: Ende des Textes.
    """
    earliest = len(text)
    for sp in stops:
        m = sp.search(text, pos=start_idx)
        if m and m.start() < earliest:
            earliest = m.start()
    return earliest

def apply_forbid(block: str, forbids: List[re.Pattern]) -> str:
    """
    Entfernt Zeilen, die Forbid-Patterns enthalten.
    Wenn alles entfernt wäre, geben wir den Originalblock zurück (fail open).
    """
    lines = [ln for ln in block.splitlines()]
    keep: List[str] = []
    for ln in lines:
        if any(f.search(ln) for f in forbids):
            continue
        keep.append(ln)
    cleaned = "\n".join(keep).strip()
    return cleaned if cleaned else block.strip()

def extract_target_block(text: str, target: TargetSpec, max_chars: int = 3000) -> str:
    """
    Logik:
      1) Suche den ersten Treffer eines der header/marker patterns.
      2) Nimm Text ab Marker-Ende bis zum frühesten stop_after.
      3) Fallback: wenn kein Stop gefunden, nimm bis max_chars.
      4) forbid-Filter anwenden.
    """
    for pat in target.patterns:
        m = find_first(text, pat)
        if not m:
            continue
        start = m.end()
        stop = find_earliest_stop(text, start, target.stop_after) if target.stop_after else len(text)
        if stop <= start:
            # Wenn Stop direkt/zu früh kommt, nimm kurzer Bereich als Fallback
            stop = min(start + max_chars, len(text))
        block = text[start:stop].strip()
        block = block[:max_chars].strip()
        block = apply_forbid(block, target.forbid)
        if block:
            return block
    return ""  # nichts gefunden

# ---------- Excel Mapping ----------

def resolve_pdf_path(row: pd.Series, pdf_root: Path) -> Optional[Path]:
    """
    Regeln:
      - Wenn 'pdf_path' existiert und Datei existiert -> diesen Pfad.
      - Sonst wenn 'pdf_file' existiert -> pdf_root / pdf_file.
      - Sonst None.
    """
    # 1) voller Pfad
    for col in ("pdf_path", "PDF_Path", "pdf_fullpath"):
        if col in row and isinstance(row[col], str) and row[col].strip():
            p = Path(row[col].strip())
            if p.suffix.lower() != ".pdf":
                p = p.with_suffix(".pdf")
            if p.exists():
                return p

    # 2) Dateiname unter pdf_root
    for col in ("pdf_file", "PDF", "file_name"):
        if col in row and isinstance(row[col], str) and row[col].strip():
            p = pdf_root / row[col].strip()
            if p.suffix.lower() != ".pdf":
                p = p.with_suffix(".pdf")
            if p.exists():
                return p

    return None

def ensure_columns(df: pd.DataFrame, targets: Dict[str, TargetSpec]) -> None:
    for col in targets.keys():
        if col not in df.columns:
            df[col] = pd.NA

# ---------- Main Processing ----------

def process(pdf_root: Path, excel_in: Path, excel_out: Path, yaml_cfg: Path) -> None:
    targets = load_config(yaml_cfg)
    logger.info("Targets geladen: %s", ", ".join(targets.keys()))

    df = pd.read_excel(excel_in)
    ensure_columns(df, targets)

    # Spaltennamen für spätere Referenz (Logging)
    logger.info("Excel-Spalten: %s", list(df.columns))

    # Cache für PDF-Text, falls Datei mehrfach vorkommt
    text_cache: Dict[Path, str] = {}

    updated = 0
    total = len(df)

    for idx, row in df.iterrows():
        pdf_path = resolve_pdf_path(row, pdf_root)
        if not pdf_path:
            logger.debug("Zeile %s: Kein PDF-Pfad ermittelbar.", idx)
            continue
        if pdf_path not in text_cache:
            text_cache[pdf_path] = read_pdf_text(pdf_path)

        txt = text_cache[pdf_path]
        if not txt.strip():
            logger.debug("Zeile %s (%s): Leerer/fehlender Text.", idx, pdf_path.name)
            continue

        row_changed = False
        for colname, spec in targets.items():
            if pd.notna(row.get(colname, pd.NA)) and str(row.get(colname)).strip():
                # Schon befüllt -> überspringen (nur Ergänzung fehlender Felder)
                continue
            block = extract_target_block(txt, spec)
            if block:
                df.at[idx, colname] = block
                row_changed = True

        if row_changed:
            updated += 1

        if (idx + 1) % 25 == 0:
            logger.info("Fortschritt: %d/%d Zeilen...", idx + 1, total)

    # Speichern
    excel_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_out, index=False)
    logger.info("Fertig. %d Zeilen aktualisiert. Ausgabe: %s", updated, excel_out)


def main():
    ap = argparse.ArgumentParser(description="HAS PDF -> Excel Extractor (clean v1)")
    ap.add_argument("--pdf-root", required=True, help="Wurzelordner der PDFs")
    ap.add_argument("--excel-in", required=True, help="Eingabe-Excel (xlsx)")
    ap.add_argument("--excel-out", required=True, help="Ausgabe-Excel (xlsx)")
    ap.add_argument("--config", required=True, help="YAML-Config mit targets/patterns")
    args = ap.parse_args()

    pdf_root = Path(args.pdf_root).resolve()
    excel_in = Path(args.excel_in).resolve()
    excel_out = Path(args.excel_out).resolve()
    yaml_cfg = Path(args.config).resolve()

    if not pdf_root.exists():
        raise FileNotFoundError(f"pdf-root existiert nicht: {pdf_root}")
    if not excel_in.exists():
        raise FileNotFoundError(f"excel-in existiert nicht: {excel_in}")
    if not yaml_cfg.exists():
        raise FileNotFoundError(f"config existiert nicht: {yaml_cfg}")

    process(pdf_root, excel_in, excel_out, yaml_cfg)


if __name__ == "__main__":
    main()
