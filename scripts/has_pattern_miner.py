#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAS Pattern Miner
-----------------
Zweck:
- Scannt rekursiv alle PDFs in einem Ordner.
- Extrahiert Text (pdfminer; Fallback PyPDF2), normalisiert Weichtrennungen/Umlaute/Whitespace.
- Findet Überschifts-Kandidaten und mappt sie auf Ziel-Sektionen via Seed-Stichwörter.
- Lernt daraus häufige Start-Überschriften (patterns) und häufige Folge-Überschriften (stop_after).
- Schreibt:
    1) YAML-Vorschlag (targets: patterns/stop_after/forbid),
    2) CSV mit Häufigkeiten & Beispielen.

Nutzung (Beispiele):
    python has_pattern_miner.py --pdf-root "Pfad\\zu\\pdfs_since_2020" --out-yaml auto_has_config.yml --out-csv auto_has_patterns.csv
    python has_pattern_miner.py --pdf-root "/data/has" --min-count 3

Voraussetzungen:
    pip install pdfminer.six PyPDF2 pyyaml
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# -------------------- Text-Extraktion --------------------

def extract_text_pdfminer(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        return extract_text(str(path)) or ""
    except Exception:
        return ""

def extract_text_pypdf2(path: Path) -> str:
    try:
        import PyPDF2  # type: ignore
        text_parts: List[str] = []
        with path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for p in reader.pages:
                try:
                    text_parts.append(p.extract_text() or "")
                except Exception:
                    text_parts.append("")
        return "\n".join(text_parts).strip()
    except Exception:
        return ""

def extract_text_any(path: Path) -> str:
    txt = extract_text_pdfminer(path)
    if not txt.strip():
        txt = extract_text_pypdf2(path)
    return txt

# -------------------- Normalisierung --------------------

SOFT_HYPHEN = "\u00AD"

def normalize_text(raw: str) -> str:
    t = raw.replace(SOFT_HYPHEN, "")
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)  # Silbentrennung am Zeilenende zusammenziehen
    t = t.replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t

def fold(s: str) -> str:
    """
    Accent-/case-insensitive Normalisierung für Vergleiche.
    - Kleinbuchstaben
    - NFD -> entferne diakritische Zeichen
    - trim mehrfachspaces
    """
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s:;,\-\u2022]", "", s)  # behalte Wort + wenige Satzzeichen (inkl. Bullet)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------- Heuristiken für Überschriften --------------------

HEADING_MAXLEN = 120

def looks_like_heading(line: str) -> bool:
    raw = line.strip()
    if not raw or len(raw) > HEADING_MAXLEN:
        return False
    # häufige Marker: Doppelpunkt, nummerierte Abschnitte, Bullet, komplett/weitgehend kapitalisiert
    if raw.endswith(":"):
        return True
    if re.match(r"^\d{1,2}(\.|[)\-])\s+\S+", raw):
        return True
    if re.match(r"^[•\-\u2022\u2794]\s+\S+", raw):
        return True
    # Schlüsselwörter (fr)
    f = fold(raw)
    if any(k in f for k in ("population", "comparateur", "indication", "conclusion", "avis", "service medical rendu", "asmr", "smr")):
        return True
    return False

# -------------------- Seeds pro Zielsektion --------------------

SEEDS: Dict[str, Sequence[str]] = {
    "Population": [
        "population cible",
        "population concernee",
        "population concern e",
        "population concernée",
        "population",
        "sous population",
        "sous-population",
        "population specifique",
    ],
    "Indication (normalized)": [
        "indication concernee",
        "indication concernee par",
        "indication de l amm",
        "indications de l amm",
        "libelle de l indication",
        "indication:",
        "indications:",
    ],
    "Comparator(s)": [
        "comparateurs cliniquement pertinents",
        "comparateurs pertinents",
        "strategies comparatives",
        "comparateurs:",
        "liste des traitements evalues par ct",
    ],
    "Conclusion (CT)": [
        "avis favorable",
        "avis defavorable",
        "maintien d inscription",
        "renouvellement d inscription",
        "inscription",
        "taux de remboursement",
        "service medical rendu",
        "smr",
        "asmr",
    ],
}

DEFAULT_FORBIDS: Dict[str, Sequence[str]] = {
    "Population": [r"\b(\d+([\.,]\d+)?)\s*(%|mg|µg|mcg|ml)\b"],
    "Subpopulation": [],
    "Indication (normalized)": [],
    "Comparator(s)": [r"\b(\d+([\.,]\d+)?)\s*(%|mg|µg|mcg|ml|ans|mois)\b"],
    "Conclusion (CT)": [],
}

# -------------------- Regex-Synthese --------------------

ACCENT_MAP = {
    "a": "[aàâ]",
    "c": "[cç]",
    "e": "[eéèêë]",
    "i": "[iîï]",
    "o": "[oôö]",
    "u": "[uùûü]",
}

def regexify_phrase(phrase: str) -> str:
    """
    Aus einer gefundenen Überschrift wie 'Population cible' wird:
        ^Population\s+cibl[eé]
    (grobe Akzenttoleranz, \s+ zwischen Wörtern; Case-Insensitive im Aufrufer setzen)
    """
    raw = phrase.strip().strip(":").strip()
    # Tokens
    tokens = re.split(r"\s+", raw)
    parts = []
    for tok in tokens:
        # Bindestriche als \s*-\s* erlauben
        tok = tok.replace("-", r"\s*-\s*")
        # Akzenttoleranz pro Buchstabe grob
        buf = []
        for ch in tok:
            base = fold(ch)
            if base in ACCENT_MAP:
                buf.append(ACCENT_MAP[base])
            else:
                buf.append(re.escape(ch))
        parts.append("".join(buf))
    if not parts:
        return ""
    return r"^" + r"\s+".join(parts)

# -------------------- Mining --------------------

def mine_patterns(pdf_root: Path, min_count: int = 2) -> Tuple[Dict[str, Counter], Dict[str, Counter], Dict[str, List[str]]]:
    """
    Liefert:
      start_counts[target] -> Counter(heading -> count)
      stop_counts[target]  -> Counter(heading -> count)  (häufige Folgeüberschriften)
      examples[target]     -> Beispiel-Strings (bis zu 5)
    """
    start_counts: Dict[str, Counter] = {t: Counter() for t in SEEDS}
    stop_counts: Dict[str, Counter] = {t: Counter() for t in SEEDS}
    examples: Dict[str, List[str]] = {t: [] for t in SEEDS}

    for pdf in pdf_root.rglob("*.pdf"):
        try:
            raw = extract_text_any(pdf)
            if not raw.strip():
                continue
            text = normalize_text(raw)
            lines = [ln.strip() for ln in text.splitlines()]
            # Index -> line (nur pot. headings vormerken)
            cand_idx = [i for i, ln in enumerate(lines) if looks_like_heading(ln)]
            for i in cand_idx:
                ln = lines[i].strip(": ").strip()
                folded = fold(ln)
                for target, seed_list in SEEDS.items():
                    if any(seed in folded for seed in seed_list):
                        start_counts[target][ln] += 1
                        if len(examples[target]) < 5:
                            examples[target].append(ln)
                        # Stop-Kandidaten: nächste 1..8 Überschriften
                        for j in cand_idx:
                            if j <= i or j > i + 8:
                                continue
                            nxt = lines[j].strip(": ").strip()
                            stop_counts[target][nxt] += 1
                        break
        except Exception:
            # still und leise weiter
            continue

    # Filter nach min_count
    for target in list(start_counts.keys()):
        start_counts[target] = Counter({k: v for k, v in start_counts[target].items() if v >= min_count})
        stop_counts[target]  = Counter({k: v for k, v in stop_counts[target].items() if v >= min_count})

    return start_counts, stop_counts, examples

def build_yaml(start_counts: Dict[str, Counter], stop_counts: Dict[str, Counter]) -> Dict:
    # Wähle Top-N Kandidaten und regexifiziere
    TOP = 20
    out = {"targets": {}}
    for target in SEEDS.keys():
        starts = [h for h, _ in start_counts.get(target, Counter()).most_common(TOP)]
        stops  = [h for h, _ in stop_counts.get(target, Counter()).most_common(TOP)]
        pattern_regexes = [regexify_phrase(s) for s in starts if regexify_phrase(s)]
        stop_regexes = [regexify_phrase(s) for s in stops if regexify_phrase(s)]
        out["targets"][target] = {
            "patterns": pattern_regexes or [],
            "stop_after": stop_regexes or [],
            "forbid": list(DEFAULT_FORBIDS.get(target, [])),
        }
    # Subpopulation: falls nicht gesondert gemined, minimal ableiten
    if "Subpopulation" not in out["targets"]:
        out["targets"]["Subpopulation"] = {
            "patterns": [r"^Sous[\s\-]?population", r"^Population\s+sp[eé]cifique"],
            "stop_after": [r"^Autres\s+recommandations", r"^HAS\s*•"],
            "forbid": [],
        }
    return out

# -------------------- IO --------------------

def write_yaml(path: Path, data: Dict) -> None:
    try:
        import yaml  # type: ignore
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=120)
    except Exception:
        # Fallback: JSON-ähnlich schreiben
        with path.open("w", encoding="utf-8") as f:
            f.write("# (Hinweis) pyyaml nicht verfügbar – Rohstruktur als Python-Dict:\n")
            f.write(repr(data))

def write_csv(path: Path, start_counts: Dict[str, Counter], stop_counts: Dict[str, Counter], examples: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "type", "raw_heading", "count"])
        for target, ctr in start_counts.items():
            for raw, cnt in ctr.most_common():
                w.writerow([target, "pattern_candidate", raw, cnt])
        for target, ctr in stop_counts.items():
            for raw, cnt in ctr.most_common():
                w.writerow([target, "stop_candidate", raw, cnt])

    # Zusätzlich: kleine Examples-Datei
    ex_path = path.with_suffix(".examples.csv")
    with ex_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "example"])
        for target, exs in examples.items():
            for ex in exs:
                w.writerow([target, ex])

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Mine section heading patterns from HAS PDFs")
    ap.add_argument("--pdf-root", type=Path, required=True, help="Ordner mit PDFs (rekursiv)")
    ap.add_argument("--out-yaml", type=Path, default=Path("auto_has_config.yml"), help="Ziel-YAML (Patterns/Stops)")
    ap.add_argument("--out-csv", type=Path, default=Path("auto_has_patterns.csv"), help="Ziel-CSV (Rohkandidaten)")
    ap.add_argument("--min-count", type=int, default=2, help="Mindesthäufigkeit, um Kandidat aufzunehmen")
    args = ap.parse_args()

    if not args.pdf_root.exists():
        raise SystemExit(f"PDF-Root nicht gefunden: {args.pdf_root}")

    print(f"[INFO] Scanne PDFs unter: {args.pdf_root}")
    start_counts, stop_counts, examples = mine_patterns(args.pdf_root, min_count=args.min_count)
    print("[INFO] Mining abgeschlossen. Schreibe Ausgaben …")

    write_csv(args.out_csv, start_counts, stop_counts, examples)
    cfg = build_yaml(start_counts, stop_counts)
    write_yaml(args.out_yaml, cfg)

    print(f"[OK] CSV:  {args.out_csv.resolve()}")
    print(f"[OK] YAML: {args.out_yaml.resolve()}")
    print("[HINWEIS] Bitte YAML kurz prüfen; danach im Extraktor als --config verwenden.")

if __name__ == "__main__":
    main()
