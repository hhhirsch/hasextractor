import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.has_fill_only_extractor_v2_4 import (  # noqa: E402
    DEFAULT_CONFIG,
    ConfigValidationError,
    load_config,
)


def test_load_config_returns_copy_of_defaults():
    config = load_config(None)
    config["targets"]["Population"]["patterns"].append("^dummy$")
    assert "^dummy$" not in DEFAULT_CONFIG["targets"]["Population"]["patterns"]


def test_load_config_raises_for_missing_custom_file(tmp_path: Path):
    missing = tmp_path / "neu.yml"
    with pytest.raises(ConfigValidationError) as excinfo:
        load_config(missing)
    assert str(missing) in str(excinfo.value)


def test_load_config_uses_external_yaml(monkeypatch, tmp_path: Path):
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda text: {"targets": {"Population": {"patterns": ["^foo$"]}}},
        YAMLError=Exception,
    )
    monkeypatch.setattr(
        "scripts.has_fill_only_extractor_v2_4.yaml", fake_yaml, raising=False
    )
    config_file = tmp_path / "neu.yml"
    config_file.write_text("targets: {}", encoding="utf-8")

    config = load_config(config_file)

    assert config["targets"]["Population"]["patterns"] == ["^foo$"]
