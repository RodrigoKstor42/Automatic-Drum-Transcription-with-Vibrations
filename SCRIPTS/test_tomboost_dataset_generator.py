#!/usr/bin/env python

import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "PHRASE_GENERATOR"
if str(GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATOR_ROOT))

from generate_tomboost_dataset import (
    CLASS_NAMES,
    CLASS_TO_MIDI,
    allocate_class_quotas,
    build_target_shares,
)


def main() -> None:
    target_shares = build_target_shares(
        target_tom_share=0.36,
        balance_individual_toms=True,
    )
    cumulative_counts: Counter[str] = Counter()
    for _ in range(1000):
        quotas = allocate_class_quotas(12, cumulative_counts, target_shares)
        assert all(quotas[class_name] > 0 for class_name in CLASS_NAMES)
        cumulative_counts.update(quotas)

    total = sum(cumulative_counts.values())
    percentages = {
        class_name: 100 * cumulative_counts[class_name] / total
        for class_name in CLASS_NAMES
    }

    assert CLASS_TO_MIDI == {
        "KD": 35,
        "SD": 38,
        "T12": 47,
        "T14": 45,
        "T16": 43,
    }
    assert 30 <= percentages["KD"] <= 35
    assert 30 <= percentages["SD"] <= 35
    for class_name in ("T12", "T14", "T16"):
        assert 10 <= percentages[class_name] <= 15

    print(f"class_names={list(CLASS_NAMES)}")
    print(f"labels={[CLASS_TO_MIDI[name] for name in CLASS_NAMES]}")
    print(f"percentages={percentages}")
    print("Tomboost quota smoke test passed.")


if __name__ == "__main__":
    main()
