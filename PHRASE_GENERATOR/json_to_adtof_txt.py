from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLITS = ("TRAIN", "VAL", "TEST")
VIBRO_SCHEMA_NAME = "KD, SD, T12, T14, T16"
CLASS_TO_MIDI = {
    "KD": 35,
    "SD": 38,
    "T12": 47,
    "T14": 45,
    "T16": 43,
    "TT": 47,
}
DEFAULT_VELOCITY = 1.0


@dataclass(frozen=True)
class ConvertedFile:
    split_name: str
    source_path: Path
    output_path: Path
    event_count: int


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning("Invalid JSON skipped: %s (%s)", path, exc)
    except OSError as exc:
        logging.warning("Could not read JSON skipped: %s (%s)", path, exc)
    return None


def get_event_class(event: dict[str, Any]) -> str | None:
    value = event.get("class", event.get("instrument"))
    if value is None:
        return None
    return str(value).strip().upper()


def get_numeric_field(event: dict[str, Any], field_name: str) -> float | None:
    value = event.get(field_name)
    if isinstance(value, bool) or value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_adtof_rows(data: dict[str, Any], source_path: Path) -> list[tuple[float, int, float]]:
    raw_events = data.get("events")
    if not isinstance(raw_events, list):
        logging.warning("JSON has no events list: %s", source_path)
        return []

    rows: list[tuple[float, int, float]] = []
    seen_events: set[tuple[float, int]] = set()
    for index, event in enumerate(raw_events):
        if not isinstance(event, dict):
            logging.warning("Ignoring non-object event %d in %s", index, source_path)
            continue

        timestamp = get_numeric_field(event, "timestamp")
        if timestamp is None:
            logging.warning("Ignoring event without numeric timestamp %d in %s", index, source_path)
            continue

        class_name = get_event_class(event)
        midi_class = CLASS_TO_MIDI.get(class_name or "")
        if midi_class is None:
            logging.warning(
                "Ignoring event with unsupported class %r at %.6f in %s",
                class_name,
                timestamp,
                source_path,
            )
            continue

        velocity = get_numeric_field(event, "velocity")
        if velocity is None:
            velocity = DEFAULT_VELOCITY

        event_key = (timestamp, midi_class)
        if event_key in seen_events:
            continue

        seen_events.add(event_key)
        rows.append((timestamp, midi_class, velocity))

    return sorted(rows, key=lambda row: row[0])


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_velocity(value: float) -> str:
    return f"{value:.3f}"


def write_annotation(path: Path, rows: Sequence[tuple[float, int, float]]) -> None:
    lines = [
        f"{format_float(timestamp)}\t{midi_class}\t{format_velocity(velocity)}"
        for timestamp, midi_class, velocity in rows
    ]
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def convert_split(split_name: str, root: Path) -> list[ConvertedFile]:
    split_dir = root / split_name
    labels_dir = split_dir / "LABELS"
    annotations_dir = split_dir / "ANNOTATIONS"

    if not labels_dir.is_dir():
        logging.warning("Labels directory not found, skipping split %s: %s", split_name, labels_dir)
        return []

    annotations_dir.mkdir(parents=True, exist_ok=True)
    converted: list[ConvertedFile] = []

    for json_path in sorted(labels_dir.glob("*.json")):
        data = read_json(json_path)
        if data is None:
            continue

        rows = extract_adtof_rows(data, json_path)
        if not rows:
            logging.warning("No valid events found in %s", json_path)

        output_path = annotations_dir / f"{json_path.stem}.txt"
        write_annotation(output_path, rows)
        converted.append(
            ConvertedFile(
                split_name=split_name,
                source_path=json_path,
                output_path=output_path,
                event_count=len(rows),
            )
        )

    return converted


def convert_dataset(root: Path, splits: Iterable[str]) -> list[ConvertedFile]:
    results: list[ConvertedFile] = []
    for split_name in splits:
        results.extend(convert_split(split_name, root))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert phrase_generator JSON labels to ADTOF tab-separated annotations "
            f"using the vibration 5-class schema: {VIBRO_SCHEMA_NAME}."
        )
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root containing TRAIN/VAL/TEST.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split folders to convert. Defaults to TRAIN VAL TEST.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    root = args.root.resolve()
    splits = [str(split).upper() for split in args.splits]

    logging.info("Using vibration 5-class schema: %s", VIBRO_SCHEMA_NAME)
    converted = convert_dataset(root, splits)
    total_events = sum(item.event_count for item in converted)

    by_split = {split_name: 0 for split_name in splits}
    for item in converted:
        by_split[item.split_name] = by_split.get(item.split_name, 0) + 1

    logging.info("Converted %d JSON files into ADTOF .txt annotations.", len(converted))
    logging.info("Converted %d total events.", total_events)
    for split_name in splits:
        logging.info("%s: %d files", split_name, by_split.get(split_name, 0))


if __name__ == "__main__":
    main()


