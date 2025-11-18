from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from scripts.has_fill_only_extractor_v2_4 import SpreadsheetProcessingError, load_spreadsheet


def _make_workbook(path: Path) -> None:
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Empty", index=False)
        pd.DataFrame({"CT": ["CT1234"], "Population": [""]}).to_excel(
            writer, sheet_name="Data", index=False
        )


def test_load_spreadsheet_picks_first_non_empty_sheet(tmp_path):
    workbook = tmp_path / "sample.xlsx"
    _make_workbook(workbook)

    df = load_spreadsheet(workbook)

    assert list(df.columns) == ["CT", "Population"]
    assert len(df) == 1


def test_load_spreadsheet_validates_sheet_name(tmp_path):
    workbook = tmp_path / "sample.xlsx"
    _make_workbook(workbook)

    df = load_spreadsheet(workbook, sheet_name="Data")
    assert "CT" in df.columns

    try:
        load_spreadsheet(workbook, sheet_name="Missing")
    except SpreadsheetProcessingError:
        pass
    else:  # pragma: no cover - the helper must raise
        raise AssertionError("expected SpreadsheetProcessingError for missing sheet")
