#!/usr/bin/env python
"""Analyze class balance in ADTOF dataset event-count exports."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adtof import config


CLASS_SCHEMA = "vibro_5_toms"
CLASS_ORDER = list(config.VIBRO_LABELS_5TXT)
TOM_CLASSES = ["T12", "T14", "T16"]
SPLIT_ORDER = ["TRAIN", "VAL", "TEST"]
MAX_PERCENTAGE_POINT_DIFFERENCE = 10.0
MIN_TOM_EVENTS_PER_SPLIT = 10
MIN_TOM_EVENTS_TOTAL = 50
MAX_CLASS_IMBALANCE_RATIO = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and diagnose ADTOF dataset event-count balance."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--counts-csv",
        type=Path,
        help="Direct path to dataset_event_counts.csv.",
    )
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing dataset_event_counts.csv.",
    )
    parser.add_argument(
        "--no-save-summary",
        action="store_true",
        help="Do not save summary CSV/JSON files when --run-dir is used.",
    )
    return parser.parse_args()


def resolve_counts_path(args):
    run_dir = args.run_dir.resolve() if args.run_dir else None
    counts_path = (
        run_dir / "dataset_event_counts.csv"
        if run_dir
        else args.counts_csv.resolve()
    )
    if not counts_path.is_file():
        raise FileNotFoundError(f"Event-count CSV does not exist: {counts_path}")
    return counts_path, run_dir


def normalized_columns(frame):
    """Return a copy with case-insensitive, whitespace-tolerant column names."""
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


def parse_long_format(frame):
    count_column = next(
        (name for name in ("count", "event_count") if name in frame.columns),
        None,
    )
    if not {"split", "class"}.issubset(frame.columns) or count_column is None:
        return None

    long_frame = frame[["split", "class", count_column]].copy()
    long_frame.columns = ["split", "class", "count"]
    return long_frame


def parse_wide_format(frame):
    class_columns = {
        column.upper(): column
        for column in frame.columns
        if column.upper() in CLASS_ORDER
    }
    if not class_columns:
        return None

    split_column = "split" if "split" in frame.columns else frame.columns[0]
    if split_column in class_columns.values():
        return None

    selected = frame[[split_column] + list(class_columns.values())].copy()
    selected = selected.rename(
        columns={split_column: "split", **{value: key for key, value in class_columns.items()}}
    )
    return selected.melt(id_vars="split", var_name="class", value_name="count")


def load_counts(counts_path):
    raw_frame = pd.read_csv(counts_path)
    frame = normalized_columns(raw_frame)
    long_frame = parse_long_format(frame)
    if long_frame is None:
        long_frame = parse_wide_format(frame)
    if long_frame is None:
        available = ", ".join(map(str, raw_frame.columns)) or "(no columns)"
        raise ValueError(
            "Unrecognized event-count CSV format. Expected columns "
            "'split,class,count' (or 'event_count'), or splits as rows and "
            f"classes as columns. Available columns: {available}"
        )

    long_frame["split"] = long_frame["split"].astype(str).str.strip().str.upper()
    long_frame["class"] = long_frame["class"].astype(str).str.strip().str.upper()
    long_frame["count"] = pd.to_numeric(long_frame["count"], errors="coerce")
    if long_frame["count"].isna().any():
        bad_rows = long_frame[long_frame["count"].isna()].to_dict(orient="records")
        raise ValueError(f"Non-numeric or missing event counts found: {bad_rows}")
    if (long_frame["count"] < 0).any():
        raise ValueError("Event counts must be non-negative.")

    long_frame = long_frame[long_frame["class"].isin(CLASS_ORDER)]
    if long_frame.empty:
        raise ValueError(
            f"No recognized classes found. Expected: {', '.join(CLASS_ORDER)}"
        )
    return long_frame


def build_summary(long_frame):
    observed_splits = list(dict.fromkeys(long_frame["split"]))
    split_order = SPLIT_ORDER + [
        split_name for split_name in observed_splits if split_name not in SPLIT_ORDER
    ]
    counts = long_frame.pivot_table(
        index="split",
        columns="class",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    counts = counts.reindex(index=split_order, columns=CLASS_ORDER, fill_value=0)
    counts = counts.loc[counts.index.isin(observed_splits)]
    counts = counts.astype(int)

    totals = counts.sum(axis=1)
    percentages = counts.div(totals.replace(0, pd.NA), axis=0).fillna(0) * 100
    return counts, percentages, totals


def diagnose(counts, percentages):
    warnings = []
    confirmations = []

    missing_splits = [split_name for split_name in SPLIT_ORDER if split_name not in counts.index]
    if missing_splits:
        warnings.append(f"Missing required splits: {', '.join(missing_splits)}.")

    for class_name in CLASS_ORDER:
        missing_positive = [
            split_name
            for split_name in SPLIT_ORDER
            if split_name in counts.index and counts.loc[split_name, class_name] <= 0
        ]
        if missing_positive:
            warnings.append(
                f"{class_name} has zero events in: {', '.join(missing_positive)}."
            )
        elif not missing_splits:
            confirmations.append(
                f"{class_name} has positive events in TRAIN, VAL, and TEST."
            )

    for class_name in TOM_CLASSES:
        total = int(counts[class_name].sum())
        if total < MIN_TOM_EVENTS_TOTAL:
            warnings.append(
                f"{class_name} has only {total} events across all splits "
                f"(guideline: at least {MIN_TOM_EVENTS_TOTAL})."
            )
        low_splits = [
            split_name
            for split_name in counts.index
            if counts.loc[split_name, class_name] < MIN_TOM_EVENTS_PER_SPLIT
        ]
        if low_splits:
            warnings.append(
                f"{class_name} has fewer than {MIN_TOM_EVENTS_PER_SPLIT} events "
                f"in: {', '.join(low_splits)}."
            )

    class_totals = counts[CLASS_ORDER].sum(axis=0)
    positive_totals = class_totals[class_totals > 0]
    if len(positive_totals) > 1:
        imbalance_ratio = positive_totals.max() / positive_totals.min()
        if imbalance_ratio > MAX_CLASS_IMBALANCE_RATIO:
            warnings.append(
                f"Global class imbalance ratio is {imbalance_ratio:.2f} "
                f"(guideline: at most {MAX_CLASS_IMBALANCE_RATIO:.0f})."
            )

    for split_name in counts.index:
        split_positive = counts.loc[split_name]
        split_positive = split_positive[split_positive > 0]
        if len(split_positive) > 1:
            imbalance_ratio = split_positive.max() / split_positive.min()
            if imbalance_ratio > MAX_CLASS_IMBALANCE_RATIO:
                warnings.append(
                    f"{split_name} class imbalance ratio is {imbalance_ratio:.2f} "
                    f"(guideline: at most {MAX_CLASS_IMBALANCE_RATIO:.0f})."
                )

    available_required_splits = [
        split_name for split_name in SPLIT_ORDER if split_name in percentages.index
    ]
    for class_name in CLASS_ORDER:
        class_percentages = percentages.loc[available_required_splits, class_name]
        if len(class_percentages) >= 2:
            spread = class_percentages.max() - class_percentages.min()
            if spread > MAX_PERCENTAGE_POINT_DIFFERENCE:
                warnings.append(
                    f"{class_name} changes by {spread:.2f} percentage points "
                    "across splits (guideline: at most 10)."
                )

    return confirmations, warnings


def summary_records(counts, percentages, totals):
    records = []
    for split_name in counts.index:
        for class_name in CLASS_ORDER:
            records.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "event_count": int(counts.loc[split_name, class_name]),
                    "split_total": int(totals.loc[split_name]),
                    "percentage": round(
                        float(percentages.loc[split_name, class_name]), 6
                    ),
                }
            )
    return records


def print_report(counts_path, counts, percentages, totals, confirmations, warnings):
    print(f"\nDataset event counts: {counts_path}")
    for split_name in counts.index:
        table = pd.DataFrame(
            {
                "count": counts.loc[split_name],
                "percentage": percentages.loc[split_name].map(
                    lambda value: f"{value:.2f}%"
                ),
            }
        )
        print(f"\n{split_name} - total events: {int(totals.loc[split_name])}")
        print(table.to_string())

    print("\nPercentage comparison by split")
    comparison = percentages.T.apply(
        lambda column: column.map(lambda value: f"{value:.2f}%")
    )
    print(comparison.to_string())

    print("\nDiagnostics")
    for message in confirmations:
        print(f"[OK] {message}")
    for message in warnings:
        print(f"[WARNING] {message}")

    print("\nConclusion")
    if warnings:
        print(
            "The current split may make some metrics unstable. Review the "
            "warnings above before comparing F1 scores."
        )
    else:
        print("The current split is suitable for interpreting onset-based metrics.")


def save_summary(run_dir, records, confirmations, warnings):
    csv_path = run_dir / "dataset_event_counts_summary.csv"
    json_path = run_dir / "dataset_event_counts_summary.json"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "class_schema": CLASS_SCHEMA,
                "class_names": CLASS_ORDER,
                "labels": list(config.VIBRO_LABELS_5),
                "summary": records,
                "confirmations": confirmations,
                "warnings": warnings,
                "suitable_for_onset_metrics": not warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved summary CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")


def main():
    args = parse_args()
    try:
        counts_path, run_dir = resolve_counts_path(args)
        long_frame = load_counts(counts_path)
        counts, percentages, totals = build_summary(long_frame)
        confirmations, warnings = diagnose(counts, percentages)
        records = summary_records(counts, percentages, totals)
        print_report(
            counts_path,
            counts,
            percentages,
            totals,
            confirmations,
            warnings,
        )
        if run_dir and not args.no_save_summary:
            save_summary(run_dir, records, confirmations, warnings)
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
