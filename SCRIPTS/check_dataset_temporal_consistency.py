from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf


SPLITS = ("TRAIN", "VAL", "TEST")
DEFAULT_STEPS = 16
DEFAULT_BEATS = 4.0
DETAIL_LIMIT = 20


@dataclass(frozen=True)
class PhraseFiles:
    split: str
    stem: str
    wav: Path | None
    json: Path | None
    txt: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check temporal consistency between dataset WAV, JSON and TXT files."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--duration-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--min-tail-margin-ms", type=float, default=100.0)
    return parser.parse_args()


def numeric_value(data: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = data.get(name)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def collect_phrase_files(dataset_root: Path) -> list[PhraseFiles]:
    phrases: list[PhraseFiles] = []
    for split in SPLITS:
        split_dir = dataset_root / split
        directories = {
            "wav": split_dir / "AUDIO",
            "json": split_dir / "LABELS",
            "txt": split_dir / "ANNOTATIONS",
        }
        indexed = {
            kind: {path.stem: path for path in directory.glob(pattern)}
            for kind, directory, pattern in (
                ("wav", directories["wav"], "*.wav"),
                ("json", directories["json"], "*.json"),
                ("txt", directories["txt"], "*.txt"),
            )
        }
        stems = sorted(set().union(*(files.keys() for files in indexed.values())))
        phrases.extend(
            PhraseFiles(
                split=split,
                stem=stem,
                wav=indexed["wav"].get(stem),
                json=indexed["json"].get(stem),
                txt=indexed["txt"].get(stem),
            )
            for stem in stems
        )
    return phrases


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")
    return data


def read_txt_timestamps(path: Path) -> tuple[list[float], int]:
    timestamps: list[float] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first_column = stripped.split()[0]
        try:
            timestamps.append(float(first_column))
        except (ValueError, IndexError):
            malformed += 1
    return timestamps, malformed


def json_event_bounds(data: dict[str, Any]) -> tuple[float | None, float | None]:
    events = data.get("events")
    if not isinstance(events, list):
        return None, None

    last_onset: float | None = None
    last_end: float | None = None
    for event in events:
        if not isinstance(event, dict):
            continue
        timestamp = numeric_value(event, ("timestamp", "time", "onset"))
        if timestamp is None:
            continue
        duration = numeric_value(event, ("duration",))
        event_end = timestamp + max(duration or 0.0, 0.0)
        last_onset = timestamp if last_onset is None else max(last_onset, timestamp)
        last_end = event_end if last_end is None else max(last_end, event_end)
    return last_onset, last_end


def format_range(values: list[float], digits: int = 3) -> str:
    if not values:
        return "n/a"
    return f"{min(values):.{digits}f} .. {max(values):.{digits}f}"


def main() -> None:
    args = parse_args()
    if args.duration_tolerance_ms < 0:
        raise SystemExit("--duration-tolerance-ms must be non-negative.")
    if args.min_tail_margin_ms < 0:
        raise SystemExit("--min-tail-margin-ms must be non-negative.")

    dataset_root = args.dataset_root.resolve()
    tolerance_seconds = args.duration_tolerance_ms / 1000.0
    min_tail_margin_seconds = args.min_tail_margin_ms / 1000.0
    phrases = collect_phrase_files(dataset_root)

    durations: list[float] = []
    bpms: list[float] = []
    duration_errors: list[float] = []
    grid_reference_errors: list[float] = []
    tail_margins: list[float] = []

    missing_triplets = 0
    unreadable_files = 0
    invalid_metadata = 0
    inconsistent_duration = 0
    grid_reference_inconsistent = 0
    outside_annotations = 0
    files_with_outside_annotations = 0
    malformed_txt_rows = 0
    low_tail_margin = 0
    details: list[str] = []
    checked = 0

    for phrase in phrases:
        missing = [
            kind
            for kind, path in (("WAV", phrase.wav), ("JSON", phrase.json), ("TXT", phrase.txt))
            if path is None
        ]
        if missing:
            missing_triplets += 1
            details.append(f"{phrase.split}/{phrase.stem}: missing {', '.join(missing)}")
            continue

        assert phrase.wav is not None
        assert phrase.json is not None
        assert phrase.txt is not None

        try:
            wav_info = sf.info(phrase.wav)
            data = read_json(phrase.json)
            txt_timestamps, malformed = read_txt_timestamps(phrase.txt)
        except (OSError, RuntimeError, json.JSONDecodeError, ValueError) as exc:
            unreadable_files += 1
            details.append(f"{phrase.split}/{phrase.stem}: unreadable data ({exc})")
            continue

        checked += 1
        duration = wav_info.frames / wav_info.samplerate
        durations.append(duration)
        malformed_txt_rows += malformed
        if malformed:
            details.append(f"{phrase.split}/{phrase.stem}: {malformed} malformed TXT rows")

        bpm = numeric_value(data, ("bpm", "tempo"))
        steps_value = numeric_value(data, ("steps", "num_steps", "grid_steps"))
        beats_value = numeric_value(
            data,
            ("beats_per_phrase", "beats_per_bar", "beats_per_measure", "beats"),
        )
        tail = numeric_value(data, ("tail_seconds", "tail_padding_seconds"))

        steps = int(steps_value) if steps_value is not None else DEFAULT_STEPS
        beats = beats_value if beats_value is not None else DEFAULT_BEATS
        tail = tail if tail is not None else 0.0

        if bpm is None or bpm <= 0 or steps <= 0 or beats <= 0 or tail < 0:
            invalid_metadata += 1
            details.append(
                f"{phrase.split}/{phrase.stem}: invalid temporal metadata "
                f"(bpm={bpm}, steps={steps}, beats={beats}, tail={tail})"
            )
            continue

        bpms.append(bpm)
        grid_duration = beats * 60.0 / bpm
        grid_expected_duration = grid_duration + tail
        _, last_json_event_end = json_event_bounds(data)

        # The current renderer keeps the complete final hit before appending tail padding.
        if last_json_event_end is not None:
            render_expected_duration = last_json_event_end + tail
        else:
            render_expected_duration = grid_expected_duration

        duration_error = abs(duration - render_expected_duration)
        duration_errors.append(duration_error)
        grid_reference_error = abs(duration - grid_expected_duration)
        grid_reference_errors.append(grid_reference_error)
        if grid_reference_error > tolerance_seconds:
            grid_reference_inconsistent += 1
        if duration_error > tolerance_seconds:
            inconsistent_duration += 1
            details.append(
                f"{phrase.split}/{phrase.stem}: duration error "
                f"{duration_error * 1000.0:.3f} ms "
                f"(actual={duration:.6f}s, expected={render_expected_duration:.6f}s, "
                f"grid={grid_duration:.6f}s, steps={steps}, beats={beats:g}, bpm={bpm:.3f})"
            )

        sample_tolerance = 1.0 / wav_info.samplerate
        outside = [
            timestamp
            for timestamp in txt_timestamps
            if timestamp < -sample_tolerance or timestamp > duration + sample_tolerance
        ]
        if outside:
            outside_annotations += len(outside)
            files_with_outside_annotations += 1
            details.append(
                f"{phrase.split}/{phrase.stem}: {len(outside)} TXT annotations outside audio"
            )

        if txt_timestamps:
            last_txt_onset = max(txt_timestamps)
            tail_margin = duration - last_txt_onset
            tail_margins.append(tail_margin)
            if tail_margin < min_tail_margin_seconds:
                low_tail_margin += 1
                details.append(
                    f"{phrase.split}/{phrase.stem}: final margin "
                    f"{tail_margin * 1000.0:.3f} ms"
                )

    average_duration = statistics.fmean(durations) if durations else 0.0
    max_duration_error = max(duration_errors, default=0.0)
    max_grid_reference_error = max(grid_reference_errors, default=0.0)

    print(f"Dataset root: {dataset_root}")
    print(f"Discovered phrase stems: {len(phrases)}")
    print(f"Complete phrases checked: {checked}")
    print(f"Missing WAV/JSON/TXT triplets: {missing_triplets}")
    print(
        "WAV duration seconds: "
        f"min={min(durations, default=0.0):.6f}, "
        f"max={max(durations, default=0.0):.6f}, "
        f"mean={average_duration:.6f}"
    )
    print(f"BPM range: {format_range(bpms)}")
    print(f"Maximum render-duration error: {max_duration_error * 1000.0:.3f} ms")
    print(
        "Maximum grid+tail reference difference: "
        f"{max_grid_reference_error * 1000.0:.3f} ms"
    )
    print(f"Files inconsistent with render duration: {inconsistent_duration}")
    print(f"Files different from grid+tail reference: {grid_reference_inconsistent}")
    print(f"TXT annotations outside audio: {outside_annotations}")
    print(f"Files with outside annotations: {files_with_outside_annotations}")
    print(f"Files with final margin below {args.min_tail_margin_ms:g} ms: {low_tail_margin}")
    print(f"Final-onset margin seconds: {format_range(tail_margins)}")
    print(f"Malformed TXT rows: {malformed_txt_rows}")
    print(f"Invalid temporal metadata files: {invalid_metadata}")
    print(f"Unreadable phrase triplets: {unreadable_files}")

    has_warnings = any(
        (
            not phrases,
            missing_triplets,
            unreadable_files,
            invalid_metadata,
            inconsistent_duration,
            outside_annotations,
            malformed_txt_rows,
            low_tail_margin,
        )
    )
    if has_warnings:
        print("\n[WARNING] Temporal inconsistencies were detected.")
        if details:
            print(f"First {min(len(details), DETAIL_LIMIT)} issue(s):")
            for detail in details[:DETAIL_LIMIT]:
                print(f"  - {detail}")
        raise SystemExit(1)

    print("\n[OK] Dataset temporal consistency checks passed.")


if __name__ == "__main__":
    main()
