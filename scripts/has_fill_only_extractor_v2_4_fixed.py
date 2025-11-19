#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the HAS fill-only extractor v2.4.

This module historically carried a fork of the extractor logic. Maintaining
that copy drifted out of sync with :mod:`scripts.has_fill_only_extractor_v2_4`
which now contains the canonical, well-tested implementation. To keep legacy
invocations of ``has_fill_only_extractor_v2_4_fixed.py`` working we simply
delegate to the shared module and re-export its public API.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence

# ``python scripts/has_fill_only_extractor_v2_4_fixed.py`` executes the file as a
# standalone script, so ``scripts`` is not necessarily importable. Inject the
# repository root to ensure absolute imports keep working.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.has_fill_only_extractor_v2_4 import *  # type: ignore[F401,F403]
from scripts.has_fill_only_extractor_v2_4 import main as _delegate_main


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point that forwards to :func:`scripts.has_fill_only_extractor_v2_4.main`."""

    _delegate_main(argv)


if __name__ == "__main__":
    main()
