from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = PROJECT_ROOT / "GENERATED_RAW" / "AUDIO"
RAW_LABELS_DIR = PROJECT_ROOT / "GENERATED_RAW" / "LABELS"
SUMMARY_PATH = PROJECT_ROOT / "dataset_summary.json"
DEFAULT_SEED = 7
SPLIT_RATIOS = {
    "TRAIN": 0.8,
    "VAL": 0.1,
    "TEST": 0.1,
}


@dataclass(frozen=True)
class DatasetPair:
    stem: str
    audio_path: Path
    json_path: Path


@dataclass(frozen=True)
class SplitResult:
    train: list[DatasetPair]
    val: list[DatasetPair]
    test: list[DatasetPair]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def collect_files(directory: Path, pattern: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(directory.glob(pattern)):
        if path.stem in files:
            duplicates.append(path.name)
            continue
        files[path.stem] = path

    for filename in duplicates:
        logging.warning("Duplicate stem ignored: %s", filename)
    return files


def detect_pairs(audio_dir: Path = RAW_AUDIO_DIR, labels_dir: Path = RAW_LABELS_DIR) -> list[DatasetPair]:
    audio_files = collect_files(audio_dir, "*.wav")
    json_files = collect_files(labels_dir, "*.json")

    audio_stems = set(audio_files)
    json_stems = set(json_files)
    missing_json = sorted(audio_stems - json_stems)
    missing_audio = sorted(json_stems - audio_stems)

    for stem in missing_json:
        logging.warning("Missing JSON label for audio: %s.wav", stem)
    for stem in missing_audio:
        logging.warning("Missing WAV audio for label: %s.json", stem)

    valid_stems = sorted(audio_stems & json_stems)
    pairs = [
        DatasetPair(stem=stem, audio_path=audio_files[stem], json_path=json_files[stem])
        for stem in valid_stems
    ]
    logging.info("Detected %d valid WAV/JSON pairs.", len(pairs))
    logging.info("Ignored %d orphan audio files and %d orphan label files.", len(missing_json), len(missing_audio))
    return pairs


def shuffle_pairs(pairs: Sequence[DatasetPair], seed: int = DEFAULT_SEED) -> list[DatasetPair]:
    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def split_pairs(pairs: Sequence[DatasetPair]) -> SplitResult:
    total = len(pairs)
    train_count = int(total * SPLIT_RATIOS["TRAIN"])
    val_count = int(total * SPLIT_RATIOS["VAL"])

    train = list(pairs[:train_count])
    val = list(pairs[train_count : train_count + val_count])
    test = list(pairs[train_count + val_count :])
    return SplitResult(train=train, val=val, test=test)


def ensure_split_dirs(split_name: str, output_root: Path) -> tuple[Path, Path]:
    audio_dir = output_root / split_name / "AUDIO"
    labels_dir = output_root / split_name / "LABELS"
    audio_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir, labels_dir


def clean_split_dirs(output_root: Path) -> None:
    for split_name in SPLIT_RATIOS:
        audio_dir, labels_dir = ensure_split_dirs(split_name, output_root)
        for directory, pattern in ((audio_dir, "*.wav"), (labels_dir, "*.json")):
            for path in directory.glob(pattern):
                path.unlink()


def copy_pairs(pairs: Iterable[DatasetPair], split_name: str, output_root: Path) -> None:
    audio_dir, labels_dir = ensure_split_dirs(split_name, output_root)
    for pair in pairs:
        shutil.copy2(pair.audio_path, audio_dir / pair.audio_path.name)
        shutil.copy2(pair.json_path, labels_dir / pair.json_path.name)


def read_metadata(pair: DatasetPair) -> dict[str, Any] | None:
    try:
        return json.loads(pair.json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning("Invalid JSON ignored in summary: %s (%s)", pair.json_path.name, exc)
    except OSError as exc:
        logging.warning("Could not read label ignored in summary: %s (%s)", pair.json_path.name, exc)
    return None


def numeric_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
        }

    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "stdev": round(statistics.pstdev(values), 6),
    }


def build_summary(split_result: SplitResult, seed: int) -> dict[str, Any]:
    all_pairs = split_result.train + split_result.val + split_result.test
    metadata = [data for pair in all_pairs if (data := read_metadata(pair)) is not None]

    profiles = Counter(str(data.get("profile_name", "unknown")) for data in metadata)
    densities = Counter(str(data.get("density_level", "unknown")) for data in metadata)
    overlap_counter = Counter(bool(data.get("contains_overlap", False)) for data in metadata)
    bpm_values = [float(data["bpm"]) for data in metadata if isinstance(data.get("bpm"), (int, float))]
    event_counts = [
        float(data["total_events"])
        for data in metadata
        if isinstance(data.get("total_events"), (int, float))
    ]

    overlap_true = overlap_counter[True]
    overlap_false = overlap_counter[False]
    overlap_total = overlap_true + overlap_false
    overlap_ratio = overlap_true / overlap_total if overlap_total else 0.0

    return {
        "seed": seed,
        "split_ratios": SPLIT_RATIOS,
        "total_samples": len(all_pairs),
        "train_count": len(split_result.train),
        "val_count": len(split_result.val),
        "test_count": len(split_result.test),
        "profile_distribution": dict(sorted(profiles.items())),
        "bpm_statistics": numeric_stats(bpm_values),
        "overlap_statistics": {
            "with_overlap": overlap_true,
            "without_overlap": overlap_false,
            "overlap_ratio": round(overlap_ratio, 6),
        },
        "density_statistics": {
            "density_level_distribution": dict(sorted(densities.items())),
            "total_events_statistics": numeric_stats(event_counts),
        },
    }


def write_summary(summary: dict[str, Any], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def prepare_dataset(
    raw_audio_dir: Path = RAW_AUDIO_DIR,
    raw_labels_dir: Path = RAW_LABELS_DIR,
    output_root: Path = PROJECT_ROOT,
    summary_path: Path = SUMMARY_PATH,
    seed: int = DEFAULT_SEED,
    clean: bool = False,
) -> SplitResult:
    pairs = detect_pairs(raw_audio_dir, raw_labels_dir)
    shuffled = shuffle_pairs(pairs, seed=seed)
    split_result = split_pairs(shuffled)

    if clean:
        logging.info("Cleaning previous split WAV/JSON files.")
        clean_split_dirs(output_root)

    copy_pairs(split_result.train, "TRAIN", output_root)
    copy_pairs(split_result.val, "VAL", output_root)
    copy_pairs(split_result.test, "TEST", output_root)

    summary = build_summary(split_result, seed=seed)
    write_summary(summary, summary_path)

    logging.info(
        "Final split: TRAIN=%d, VAL=%d, TEST=%d.",
        len(split_result.train),
        len(split_result.val),
        len(split_result.test),
    )
    logging.info("Summary written to: %s", summary_path)
    return split_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reproducible ADTOF train/val/test splits.")
    parser.add_argument("--raw-audio-dir", type=Path, default=RAW_AUDIO_DIR)
    parser.add_argument("--raw-labels-dir", type=Path, default=RAW_LABELS_DIR)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing WAV/JSON files from TRAIN, VAL and TEST before copying.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    prepare_dataset(
        raw_audio_dir=args.raw_audio_dir,
        raw_labels_dir=args.raw_labels_dir,
        output_root=args.output_root,
        summary_path=args.summary_path,
        seed=args.seed,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
