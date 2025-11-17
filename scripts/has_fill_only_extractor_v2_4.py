"""Improved HAS fill-only extractor.

This rewrite keeps the behaviour of v2.3d but introduces caching, structured
logging and safer section slicing while retaining compatibility with the old
CLI.
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Pattern, Sequence

try:  # pragma: no cover - optional dependency required for spreadsheet IO
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised via tests
    pd = None  # type: ignore[assignment]
try:  # pragma: no cover - optional dependency for custom configs
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised via tests
    yaml = None  # type: ignore[assignment]

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


class PDFExtractionError(RuntimeError):
    """Raised when both PDF extraction backends fail for a file."""


class ConfigValidationError(RuntimeError):
    """Raised when the YAML configuration cannot be parsed."""


class SpreadsheetProcessingError(RuntimeError):
    """Raised when the spreadsheet could not be loaded."""


def ensure_pandas() -> None:
    """Ensure pandas is available before touching spreadsheets."""

    if pd is None:
        raise SpreadsheetProcessingError(
            "pandas is required for spreadsheet processing; install pandas to continue"
        )


def extract_text_pdfminer(path: Path) -> str:
    """Return the raw text of ``path`` using pdfminer."""

    try:
        from pdfminer.high_level import extract_text  # type: ignore

        logger.debug("extracting %s with pdfminer", path)
        return extract_text(str(path))
    except Exception as exc:  # pragma: no cover - pdfminer failures depend on env
        raise PDFExtractionError(f"pdfminer failed for {path}: {exc}") from exc


def extract_text_pypdf2(path: Path) -> str:
    """Return the raw text of ``path`` using PyPDF2."""

    try:
        import PyPDF2  # type: ignore

        text: List[str] = []
        with path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception as exc:  # pragma: no cover
                    logger.warning("PyPDF2 failed on %s page: %s", path, exc)
                    text.append("")
        combined = "\n".join(text).strip()
        if not combined:
            raise PDFExtractionError(f"PyPDF2 returned no text for {path}")
        return combined
    except PDFExtractionError:
        raise
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise PDFExtractionError(f"PyPDF2 failed for {path}: {exc}") from exc


def extract_text_any(path: Path) -> str:
    """Try pdfminer first, then PyPDF2, raising :class:`PDFExtractionError` on failure."""

    try:
        return extract_text_pdfminer(path)
    except PDFExtractionError:
        logger.debug("pdfminer fallback for %s", path)
    return extract_text_pypdf2(path)


def normalize_text(raw: str) -> str:
    """Clean up hyphenation and whitespace artefacts from ``raw`` text."""

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
        """Return cached, normalised text if possible, logging failures per file."""

        if path in self.cache:
            return self.cache[path]
        try:
            raw = extract_text_any(path)
        except PDFExtractionError as exc:
            logger.warning("no extractor succeeded for %s: %s", path, exc)
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
    """Extracts configured sections from normalised PDF text blobs."""

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
        """Return the snippet for ``target`` or an empty string when missing."""

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
        if not snippet.strip() and target == "Population":
            snippet = text[match.start() : match.end()]
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
    """Normalise comparator blocks into a consistent semicolon-separated value."""

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


DEFAULT_CONFIG: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "targets": {
        "Population": {
            "patterns": [
                r"^Population\s*cible",
                r"^Population\s+concern[ée]e",
                r"^09\.?\s*Population\s*cible",
                r"^[•\-–—\u2022\u2794\uF0A7]?\s*(?:Enfants?|Adolescents?|Adultes?|Femmes?|Hommes?|Personnes?).{0,80}?(?:âg[ée]s?\s+de\s+\d+(?:\s*(?:à|-|–|—)\s*\d+)?\s*ans|≥\s*\d+\s*ans|>\s*\d+\s*ans|<\s*\d+\s*ans)\b.*",
                r"(?i)\bchez\s+les?\s+(?:patients?|personnes?)\s+(?:ambulatoires\s+)?(?:adultes?|enfants?|adolescents?|femmes?|hommes?).{0,120}",
                r"(?i)\bpatients?\s+(?:[^\n]{0,60})?\bâg[ée]s?\s+de\s+\d+(?:\s*(?:à|-|–|—)\s*\d+)?\s*ans\b.*",
                r"(?is)^Indication\s+concern[ée]e.*?\b(?:chez\s+les\s+)?patients?\b[^\n]{0,160}?\bâg[ée]s?\s+de\s+\d+(?:\s*à\s*\d+)?\s*ans\b.*",
            ],
            "stop_after": [
                r"^Autres\s+recommandations",
                r"^Conditionnements?",
                r"^HAS\s*•",
                r"^Place\s+du\s+m[ée]dicament",
                r"^Service\s+M[ée]dical\s+Rendu",
                r"^SMR\b",
                r"^Crit[èe]res?\s+d[’']?[ée]ligibilit[ée]",
                r"^Contexte\b",
                r"^Synth[èe]se\b",
                r"^Prise\s+en\s+charge",
            ],
            "forbid": [r"\b(\d+([\.,]\d+)?)\s*(%|mg|µg|mcg|ml)\b"],
        },
        "Subpopulation": {
            "patterns": [
                r"^Sous-population",
                r"^Sous\s+populations?",
                r"^Population\s+sp[ée]cifique",
            ],
            "stop_after": [
                r"^Autres\s+recommandations",
                r"^Conditionnements?",
                r"^HAS\s*•",
            ],
            "forbid": [],
        },
        "Indication (normalized)": {
            "patterns": [
                r"^Indications?\s+de\s+l[’']AMM",
                r"^Indications?\s+concern[ée]es?\s+par\s+l[’']évaluation",
                r"^Libell[ée]\s+de\s+l[’']indication",
                r"^Indication\s+concer[\- ]n[ée]e\s+par",
                r"^Indication\s+concern[ée]e",
                r"^Indication\s*:\s*",
                r"^Indications?\s*:\s*",
            ],
            "stop_after": [
                r"^DCI\b",
                r"^ATC\b",
                r"^Pr[ée]sentations?\s+concern[ée]es?",
                r"^Listes?\s+concern[ée]es?",
                r"^Conditions?\s+de\s+prescription",
                r"^AMM\s*\(",
                r"^Laboratoire",
                r"^Taux\s+de\s+remboursement",
                r"^Service\s+M[ée]dical\s+Rendu",
                r"^Place\s+du\s+m[ée]dicament",
                r"^Autres\s+recommandations",
                r"^HAS\s*•",
            ],
            "forbid": [],
        },
        "Comparator(s)": {
            "patterns": [
                r"^Comparateurs?\s+cliniquement\s+pertinents?",
                r"^Strat[ée]gies?\s+comparatives?",
                r"^Comparateurs?\s*:",
                r"^Liste\s+des\s+traitements.*\b[ée]valu[ée]s?\s+par\s+CT",
            ],
            "stop_after": [
                r"^Service\s+M[ée]dical\s+Rendu",
                r"^Place\s+du\s+m[ée]dicament",
                r"^Autres\s+recommandations",
                r"^HAS\s*•",
            ],
            "forbid": [r"\b(\d+([\.,]\d+)?)\s*(%|mg|µg|mcg|ml|ans|mois)\b"],
        },
        "Conclusion (CT)": {
            "patterns": [
                r"avis\s+(favorable|d[ée]favorable)",
                r"\b(maintien|renouvellement)\s+d[’']inscription\b",
                r"\b(non-)?inscription\b",
                r"taux\s+de\s+remboursement\s*:\s*\d+\s*%",
            ],
            "stop_after": [
                r"^Autres\s+recommandations",
                r"^HAS\s*•",
                r"^Service\s+M[ée]dical\s+Rendu",
            ],
            "forbid": [],
        },
    }
}


def load_config(path: Optional[Path]) -> dict:
    """Load configuration from ``path`` or fall back to the embedded YAML."""

    if path is None or not path.exists():
        if path and not path.exists():
            logger.warning("config %s not found, using defaults", path)
        return copy.deepcopy(DEFAULT_CONFIG)
    if yaml is None:
        raise ConfigValidationError(
            "PyYAML is required to parse external config files; install pyyaml to continue"
        )
    raw = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:  # pragma: no cover - malformed config is rare
        raise ConfigValidationError(f"invalid YAML config {path}: {exc}") from exc


def map_ct_to_pdfs(pdf_root: Path) -> Dict[str, List[Path]]:
    """Return a mapping of CT identifiers to candidate PDF paths."""

    mapping: Dict[str, List[Path]] = {}
    for pdf in pdf_root.rglob("*.pdf"):
        match = RE_CT.search(pdf.name)
        if match:
            key = match.group(0).replace(" ", "").upper()
            mapping.setdefault(key, []).append(pdf)
    return mapping


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    """Persist ``rows`` as CSV at ``path`` with the provided ``fieldnames`` order."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class RowProcessor:
    """Derives column updates for a single spreadsheet row."""

    def __init__(self, extractor: SectionExtractor, targets: Sequence[str]):
        self.extractor = extractor
        self.targets = tuple(targets)

    def compute_updates(self, row: pd.Series, normalized_text: str) -> Dict[str, str]:
        """Return a column -> value mapping for blank targets within ``row``."""

        snippet_cache: MutableMapping[str, str] = {}

        def get_snippet(name: str) -> str:
            if name not in snippet_cache:
                snippet_cache[name] = self.extractor.extract(normalized_text, name)
            return snippet_cache[name]

        updates: Dict[str, str] = {}
        for target in self.targets:
            if target not in row.index:
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
                updates[target] = snippet
        return updates


def process(
    pdf_root: Path,
    excel_in: Path,
    excel_out: Path,
    config_path: Optional[Path],
) -> None:
    """Drive the PDF-to-spreadsheet enrichment workflow."""

    logger.info("loading config %s", config_path or "<embedded>")
    config = load_config(config_path)
    extractor = SectionExtractor(config)
    pdf_repo = PDFTextRepository()

    logger.info("loading spreadsheet %s", excel_in)
    ensure_pandas()
    try:
        df = pd.read_excel(excel_in)
    except Exception as exc:  # pragma: no cover - depends on engine/runtime
        raise SpreadsheetProcessingError(f"failed to read {excel_in}: {exc}") from exc
    for column in TARGETS:
        if column in df.columns:
            df[column] = df[column].astype("object")

    ct_columns = find_ct_columns(df.columns)
    df["_CT"] = df.apply(lambda r: best_ct_from_row(r, ct_columns), axis=1)
    available_targets = [target for target in TARGETS if target in df.columns]
    row_processor = RowProcessor(extractor, available_targets)

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

        updates = row_processor.compute_updates(row, normalized_text)
        for column, snippet in updates.items():
            df.at[idx, column] = snippet
            changes.append(
                ChangeRow(
                    index=idx,
                    ct=ct_value,
                    column=column,
                    new_value=snippet,
                    source_pdf=str(used_file),
                )
            )
        diagnostics.append(
            DiagnosticRow(
                ct=ct_value,
                file=str(used_file) if used_file else str(pdf_candidates[0]),
                reason="filled" if updates else "no_blank_targets_or_no_sections",
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
    """Extract a short biomarker/line-of-therapy token from indication text."""

    if not indication_hint:
        return ""
    match = re.search(
        r"(?i)(\badultes?\b|\benfants?\b|\badolescents?\b|\b1(?:re|ère)\s*ligne\b|\b2(?:e|ème|eme)\s*ligne\b|HER2\s*(positif|n[ée]gatif)|EGFR|ALK|MSI[- ]?H|dMMR)",
        indication_hint,
    )
    return match.group(0) if match else ""


def infer_population(text: str, indication_hint: str = "") -> str:
    """Infer a population snippet from free text or indication context."""

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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML file overriding the embedded defaults; requires PyYAML when provided"
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    process(args.pdf_root, args.excel_in, args.excel_out, args.config)


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()
