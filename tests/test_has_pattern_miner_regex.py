import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.has_pattern_miner import regexify_phrase


def test_regexify_handles_hyphenated_tokens():
    pattern = regexify_phrase("Sous-population")
    assert r"\s*-\s*" in pattern
    assert re.match(pattern, "Sous-population", re.IGNORECASE)
