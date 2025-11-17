import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.has_fill_only_extractor_v2_4 import (
    DEFAULT_CONFIG,
    RowProcessor,
    SectionExtractor,
    infer_population,
    normalize_comparators,
)

class DummySeries(dict):
    """Minimal dict-like stand-in for :class:`pandas.Series` used in tests."""

    @property
    def index(self):  # type: ignore[override]
        return tuple(self.keys())


def _build_extractor() -> SectionExtractor:
    return SectionExtractor(copy.deepcopy(DEFAULT_CONFIG))


def test_population_pattern_matches_bullet() -> None:
    extractor = _build_extractor()
    text = (
        "Synthèse\n"
        "• Enfant âgé de 4 à 7 ans recevant un traitement ambulatoire\n"
        "Autres recommandations\n"
    )
    snippet = extractor.extract(text, "Population")
    assert "4 à 7 ans" in snippet


def test_row_processor_infers_population_from_indication() -> None:
    extractor = _build_extractor()
    processor = RowProcessor(extractor, ("Population", "Indication (normalized)"))
    row = DummySeries({"Population": "", "Indication (normalized)": ""})
    text = (
        "Indication concernée par l'évaluation\n"
        "Chez les patients adultes âgés de 18 ans atteints de la maladie X\n"
    )
    updates = processor.compute_updates(row, text)
    assert "Population" in updates
    assert "adultes" in updates["Population"].lower()


def test_normalize_comparators_merges_unique_items() -> None:
    raw = "\n".join(
        [
            "• STRAT 1",
            "• STRAT 1",  # duplicate should collapse
            "Traitement B, Traitement C",
            "Aucun comparateur",
        ]
    )
    normalized = normalize_comparators(raw)
    # Once "Aucun comparateur" is mentioned, it should override the rest
    assert normalized == "Aucun CCP"


def test_infer_population_picks_sentence_without_header() -> None:
    text = "Chez les patients adolescents âgés de 12 à 18 ans suivis en ambulatoire"
    inferred = infer_population(text)
    assert "12 à 18" in inferred
