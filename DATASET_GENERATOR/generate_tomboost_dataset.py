#!/usr/bin/env python
from __future__ import annotations

"""Compatibility wrapper for the unified phrase generator.

The tom-boost dataset logic now lives in PHRASE_GENERATOR/SRC/phrase_generator.py
so there is only one generation pipeline to maintain.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "SRC"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phrase_generator import (  # noqa: E402,F401
    CLASS_NAMES,
    CLASS_TO_MIDI,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PEAK_LIMIT,
    DEFAULT_TARGET_TOM_SHARE,
    DEFAULT_TOM_BOOST,
    DEFAULT_WAV_SUBTYPE,
    SAMPLES_DIR,
    TARGET_SR,
    TOM_CLASSES,
    allocate_class_quotas,
    build_balanced_phrase,
    build_target_shares,
    boosted_profile,
    generate_split_dataset,
    main,
)


if __name__ == "__main__":
    main()
