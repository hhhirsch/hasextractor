"""Improved HAS fill-only extractor.

This rewrite keeps the behaviour of v2.3d but introduces caching, structured
logging and safer section slicing while retaining compatibility with the old
CLI.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "has_fill_only_config_v2_4.yml"

SOFT_HYPHEN = "\u00AD"
TARGETS = [
    "Population",
    "Subpopulation",
    "Indication (normalized)",
    "Comparator(s)",
    "Conclusion (CT)",
]
RE_CT = re.compile(r"(?i)CT[-\s]?\d{4,6}(?!\d)")
UNIT_PATTERN = re.compile(r"\b(\d+([\.,]\d+)?)\s*(%|mg|µg|mcg|ml|ans|mois)\b", re.I)


def extract_text_pdfminer(path: Path) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        logger.debug("extracting %s with pdfminer", path)
        return extract_text(str(path))
    except Exception as exc:  # pragma: no cover - pdfminer failures depend on env
        logger.warning("pdfminer failed for %s: %s", path, exc)
        return None


def extract_text_pypdf2(path: Path) -> Optional[str]:
    try:
        import PyPDF2  # type: ignore

        text = []
        with path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception as exc:  # pragma: no cover
                    logger.warning("PyPDF2 failed on %s page: %s", path, exc)
                    text.append("")
        combined = "\n".join(text).strip()
        return combined or None
    except Exception as exc:  # pragma: no cover - depends on optional dep
        logger.warning("PyPDF2 failed for %s: %s", path, exc)
        return None


def normalize_text(raw: str) -> str:
    text = raw.replace(SOFT_HYPHEN, "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text


@dataclass
class PDFTextRepository:
    """Caches raw and normalised PDF text."""

    cache: Dict[Path, str] = field(default_factory=dict)

    def get_normalized_text(self, path: Path) -> Optional[str]:
        if path in self.cache:
            return self.cache[path]
        raw = extract_text_pdfminer(path) or extract_text_pypdf2(path)
        if not raw:
            self.cache[path] = ""
            return None
        normalised = normalize_text(raw)
        self.cache[path] = normalised
        return normalised


@dataclass
class SectionSpec:
    name: str
    patterns: Sequence[Pattern]
    stops: Sequence[Pattern]
    forbids: Sequence[Pattern]


class SectionExtractor:
    def __init__(self, config: dict):
        self.specs: Dict[str, SectionSpec] = {}
        for name, cfg in config.get("targets", {}).items():
            self.specs[name] = SectionSpec(
                name=name,
                patterns=[re.compile(p, re.I | re.M) for p in cfg.get("patterns", [])],
                stops=[re.compile(p, re.I | re.M) for p in cfg.get("stop_after", [])],
                forbids=[re.compile(p, re.I) for p in cfg.get("forbid", [])],
            )

    def extract(self, text: str, target: str) -> str:
        spec = self.specs.get(target)
        if not spec:
            return ""
        match = None
        for pat in spec.patterns:
            match = pat.search(text)
            if match:
                break
        if not match:
            return ""
        after = text[match.end() :]
        snippet = self._trim_with_stops(after, spec.stops)
        snippet = self._sanitize_footer(snippet)
        if target == "Comparator(s)":
            snippet = normalize_comparators(snippet)
        if any(r.search(snippet) for r in spec.forbids):
            return ""
        return snippet.strip()

    @staticmethod
    def _trim_with_stops(text: str, stops: Sequence[Pattern]) -> str:
        for stop in stops:
            stop_match = stop.search(text)
            if stop_match:
                text = text[: stop_match.start()]
                break
        return text.strip()

    @staticmethod
    def _sanitize_footer(text: str) -> str:
        replacements = [
            r"(?i)ce document ainsi que sa r[ée]f[ée]rence.*",
            r"(?i)haute autorit[ée] de sant[ée].*",
            r"(?i)www\.has-sante\.fr.*",
            r"(?i)\bt[ée]l\.\s*:+.*",
        ]
        for pattern in replacements:
            text = re.sub(pattern, "", text)
        return text


def normalize_comparators(block: str) -> str:
    stripped = block.strip()
    if not stripped:
        return stripped
    if re.search(r"\bil n[’']existe pas de\b", stripped, re.I) or re.search(
        r"\baucun comparateur\b", stripped, re.I
    ):
        return "Aucun CCP"
    lines = [line.strip("•-– \t") for line in stripped.splitlines() if line.strip()]
    keep: List[str] = []
    for line in lines:
        if UNIT_PATTERN.search(line):
            continue
        if (";" in line) or ("," in line) or re.search(r"\s{2,}", line):
            keep.append(line)
            continue
        if len(line.split()) <= 4:
            keep.append(line)
    if keep:
        items: List[str] = []
        for entry in keep:
            for part in re.split(r"[;,]", entry):
                cleaned = part.strip()
                if cleaned and cleaned not in items:
                    items.append(cleaned)
        return "; ".join(items)
    if re.search(r"m[êe]mes? que .*autres pr[ée]sentations", stripped, re.I) or re.search(
        r"sont les m[êe]mes que",
        stripped,
        re.I,
    ):
        match = re.search(r"pr[ée]sentations?\s+d[’']([A-Z0-9][A-Z0-9\-_]+)", stripped)
        product = match.group(1) if match else ""
        suffix = f" de {product}" if product else ""
        return f"Même CCP que autres présentations{suffix}"
    return ""


@dataclass
class DiagnosticRow:
    ct: str
    file: str
    reason: str
    used_ocr: bool = False


@dataclass
class ChangeRow:
    index: int
    ct: str
    column: str
    new_value: str
    source_pdf: str


@dataclass
class UnmatchedRow:
    index: int
    reason: str
    ct: Optional[str] = None


def normalize_ct(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    match = RE_CT.search(str(value))
    return match.group(0).replace(" ", "").upper() if match else None


def find_ct_columns(columns: Iterable[str]) -> List[str]:
    selected = []
    for column in columns:
        low = column.lower()
        if ("ct" in low or "evamed" in low) and "conclusion" not in low:
            selected.append(column)
    return selected


def best_ct_from_row(row: pd.Series, ct_columns: Sequence[str]) -> Optional[str]:
    for column in ct_columns:
        val = row.get(column)
        if pd.isna(val):
            continue
        candidate = normalize_ct(str(val))
        if candidate:
            return candidate
    return None


DEFAULT_YAML = r"""
targets:
  Population:
    patterns:
      - '^Population\s*cible'
      - '^Population\s+concern[ée]e'
      - '^09\.?\s*Population\s*cible'
      - "^[•\-–—\u2022\u2794\uF0A7]?\s*(?:Enfants?|Adolescents?|Adultes?|Femmes?|Hommes?|Personnes?).{0,80}?(?:âg[ée]s?\s+de\s+\d+(?:\s*(?:à|-|–|—)\s*\d+)?\s*ans|≥\s*\d+\s*ans|>\s*\d+\s*ans|<\s*\d+\s*ans)\b.*"
      - "(?i)\bchez\s+les?\s+(?:patients?|personnes?)\s+(?:ambulatoires\s+)?(?:adultes?|enfants?|adolescents?|femmes?|hommes?).{0,120}"
      - "(?i)\bpatients?\s+(?:[^\n]{0,60})?\bâg[ée]s?\s+de\s+\d+(?:\s*(?:à|-|–|—)\s*\d+)?\s*ans\b.*"
      - "(?is)^Indication\s+concern[ée]e.*?\b(?:chez\s+les\s+)?patients?\b[^\n]{0,160}?\bâg[ée]s?\s+de\s+\d+(?:\s*à\s*\d+)?\s*ans\b.*"
    stop_after:
      - '^Autres\s+recommandations'
      - '^Conditionnements?'
      - '^HAS\s*•'
      - '^Place\s+du\s+m[ée]dicament'
      - '^Service\s+M[ée]dical\s+Rendu'
      - '^SMR\b'
      - '^Crit[èe]res?\s+d[’']?[ée]ligibilit[ée]'
      - '^Contexte\b'
      - '^Synth[èe]se\b'
      - '^Prise\s+en\s+charge'
    forbid:
      - '\\b(\\d+([\\.,]\\d+)?)\\s*(%|mg|µg|mcg|ml|ans|mois)\\b'
  Subpopulation:
    patterns:
      - '^Sous-population'
      - '^Sous\s+populations?'
      - '^Population\s+sp[ée]cifique'
    stop_after:
      - '^Autres\s+recommandations'
      - '^Conditionnements?'
      - '^HAS\s*•'
  Indication (normalized):
    patterns:
      - "^Indications?\s+de\s+l[’']AMM"
      - "^Indications?\s+concern[ée]es?\s+par\s+l[’']évaluation"
      - "^Libell[ée]\s+de\s+l[’']indication"
      - "^Indication\s+concer[\- ]n[ée]e\s+par"
      - "^Indication\s+concern[ée]e"
      - "^Indication\s*:\s*"
      - "^Indications?\s*:\s*"
    stop_after:
      - '^DCI\b'
      - '^ATC\b'
      - '^Pr[ée]sentations?\s+concern[ée]es?'
      - '^Listes?\s+concern[ée]es?'
      - '^Conditions?\s+de\s+prescription'
      - '^AMM\s*\('
      - '^Laboratoire'
      - '^Taux\s+de\s+remboursement'
      - '^Service\s+M[ée]dical\s+Rendu'
      - '^Place\s+du\s+m[ée]dicament'
      - '^Autres\s+recommandations'
      - '^HAS\s*•'
  Comparator(s):
    patterns:
      - '^Comparateurs?\s+cliniquement\s+pertinents?'
      - '^Strat[ée]gies?\s+comparatives?'
      - '^Comparateurs?\s*:'
      - '^Liste\s+des\s+traitements.*\\b[ée]valu[ée]s?\s+par\s+CT'
    stop_after:
      - '^Service\s+M[ée]dical\s+Rendu'
      - '^Place\s+du\s+m[ée]dicament'
      - '^Autres\s+recommandations'
      - '^HAS\s*•'
    forbid:
      - '\\b(\\d+([\\.,]\\d+)?)\\s*(%|mg|µg|mcg|ml|ans|mois)\\b'
  Conclusion (CT):
    patterns:
      - 'avis\s+(favorable|d[ée]favorable)'
      - '\\b(maintien|renouvellement)\s+d[’'']inscription\\b'
      - '\\b(non-)?inscription\\b'
      - 'taux\s+de\s+remboursement\s*:\s*\\d+\s*%'
    stop_after:
      - '^Autres\s+recommandations'
      - '^HAS\s*•'
      - '^Service\s+M[ée]dical\s+Rendu'
"""


def load_config(path: Optional[Path]) -> dict:
    if path is None:
        return yaml.safe_load(DEFAULT_YAML)
    if not path.exists():
        logger.warning("config %s not found, using defaults", path)
        return yaml.safe_load(DEFAULT_YAML)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def map_ct_to_pdfs(pdf_root: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for pdf in pdf_root.rglob("*.pdf"):
        match = RE_CT.search(pdf.name)
        if match:
            key = match.group(0).replace(" ", "").upper()
            mapping.setdefault(key, []).append(pdf)
    return mapping


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process(
    pdf_root: Path,
    excel_in: Path,
    excel_out: Path,
    config_path: Optional[Path],
) -> None:
    logger.info("loading config %s", config_path or "<embedded>")
    config = load_config(config_path)
    extractor = SectionExtractor(config)
    pdf_repo = PDFTextRepository()

    logger.info("loading spreadsheet %s", excel_in)
    df = pd.read_excel(excel_in)
    for column in TARGETS:
        if column in df.columns:
            df[column] = df[column].astype("object")

    ct_columns = find_ct_columns(df.columns)
    df["_CT"] = df.apply(lambda r: best_ct_from_row(r, ct_columns), axis=1)

    ct_to_pdfs = map_ct_to_pdfs(pdf_root)

    changes: List[ChangeRow] = []
    diagnostics: List[DiagnosticRow] = []
    unmatched: List[UnmatchedRow] = []

    for idx, row in df.iterrows():
        ct_value = row["_CT"]
        if not ct_value:
            unmatched.append(UnmatchedRow(index=idx, reason="no_ct_in_row"))
            continue
        pdf_candidates = ct_to_pdfs.get(ct_value, [])
        if not pdf_candidates:
            unmatched.append(UnmatchedRow(index=idx, ct=ct_value, reason="no_pdf"))
            continue

        normalized_text: Optional[str] = None
        used_file: Optional[Path] = None
        for candidate in pdf_candidates:
            normalized_text = pdf_repo.get_normalized_text(candidate)
            if normalized_text:
                used_file = candidate
                break
        if not normalized_text:
            diagnostics.append(
                DiagnosticRow(ct=ct_value, file=str(pdf_candidates[0]), reason="no_text")
            )
            continue

        filled_any = False
        snippet_cache: Dict[str, str] = {}

        def get_snippet(name: str) -> str:
            if name not in snippet_cache:
                snippet_cache[name] = extractor.extract(normalized_text, name)
            return snippet_cache[name]

        for target in TARGETS:
            if target not in df.columns:
                continue
            current_value = row.get(target)
            if current_value is not None and str(current_value).strip():
                continue
            snippet = get_snippet(target)
            if target == "Subpopulation" and not snippet:
                indication_hint = snippet_cache.get("Indication (normalized)") or get_snippet(
                    "Indication (normalized)"
                )
                snippet = infer_subpopulation(indication_hint)
            if target == "Population" and not snippet:
                indication_hint = snippet_cache.get("Indication (normalized)") or get_snippet(
                    "Indication (normalized)"
                )
                snippet = infer_population(normalized_text, indication_hint)
            if snippet:
                snippet_cache[target] = snippet
                df.at[idx, target] = snippet
                changes.append(
                    ChangeRow(
                        index=idx,
                        ct=ct_value,
                        column=target,
                        new_value=snippet,
                        source_pdf=str(used_file),
                    )
                )
                filled_any = True
        diagnostics.append(
            DiagnosticRow(
                ct=ct_value,
                file=str(used_file) if used_file else str(pdf_candidates[0]),
                reason="filled" if filled_any else "no_blank_targets_or_no_sections",
            )
        )

    excel_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_out, index=False)
    logger.info("saved Excel %s", excel_out)

    change_rows = [change.__dict__ for change in changes]
    diag_rows = [diag.__dict__ for diag in diagnostics]
    unmatched_rows = [row.__dict__ for row in unmatched]

    write_csv(
        excel_out.with_suffix(".changes.csv"), ["index", "ct", "column", "new_value", "source_pdf"], change_rows
    )
    write_csv(
        excel_out.with_suffix(".pdf_diagnostics.csv"), ["ct", "file", "reason", "used_ocr"], diag_rows
    )
    if unmatched_rows:
        write_csv(
            excel_out.with_suffix(".unmatched_cts.csv"), ["index", "ct", "reason"], unmatched_rows
        )


def infer_subpopulation(indication_hint: str) -> str:
    if not indication_hint:
        return ""
    match = re.search(
        r"(?i)(\badultes?\b|\benfants?\b|\badolescents?\b|\b1(?:re|ère)\s*ligne\b|\b2(?:e|ème|eme)\s*ligne\b|HER2\s*(positif|n[ée]gatif)|EGFR|ALK|MSI[- ]?H|dMMR)",
        indication_hint,
    )
    return match.group(0) if match else ""


def infer_population(text: str, indication_hint: str = "") -> str:
    blob = f"{indication_hint}\n{text}" if indication_hint else text
    patterns = [
        r"(?im)^[•\-\u2022\u2794]?\s*(?:Enfants?|Adolescents?|Adultes?|Femmes?|Hommes?|Personnes?).{0,80}?(?:âg[ée]s?\s+de\s+\d+(?:\s*(?:à|–|-|—)\s*\d+)?\s*ans|≥\s*\d+\s*ans|>\s*\d+\s*ans|<\s*\d+\s*ans)\b.*",
        r"(?i)\bchez\s+les?\s+(?:patients?|personnes?)\s+(?:ambulatoires\s+)?(?:adultes?|enfants?|adolescents?|femmes?|hommes?).{0,120}",
        r"(?i)\bpatients?\b.{0,60}?âg[ée]s?\s+de\s+\d+(?:\s*(?:à|–|-|—)\s*\d+)?\s*ans\b.*",
    ]
    for pattern in patterns:
        match = re.search(pattern, blob, re.I | re.M | re.S)
        if match:
            return match.group(0).strip()
    return ""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HAS fill-only extractor v2.4")
    parser.add_argument("pdf_root", type=Path)
    parser.add_argument("excel_in", type=Path)
    parser.add_argument("excel_out", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    process(args.pdf_root, args.excel_in, args.excel_out, args.config)


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()
