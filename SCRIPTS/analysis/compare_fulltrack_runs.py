"""
Compare two full-track calibration outputs produced by SCRIPTS/calibration/calibrate_fulltrack_thresholds.py.

The comparison is intentionally report-only: it reads selected inference configs
and writes a compact summary without touching datasets, generators, DataLoaders,
training code, or model checkpoints.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


EXPECTED_CLASS_NAMES = ["KD", "SD", "T12", "T14", "T16"]
EXPECTED_MIDI_LABELS = [35, 38, 47, 45, 43]
FOCUS_CLASSES = {"T14", "T16"}
EPS = 1e-12
ALTERNATIVE_CONFIG_RELATIVE_PATHS = [
    Path("fulltrack_calibration") / "fulltrack_selected_inference_config.json",
    Path("fulltrack_calibration_f1_nms") / "fulltrack_selected_inference_config.json",
    Path("fulltrack_calibration_f1_nms_refined") / "fulltrack_selected_inference_config.json",
    Path("fulltrack_calibration_perclass") / "fulltrack_selected_inference_config_per_class.json",
    Path("fulltrack_calibration_perclass") / "fulltrack_selected_inference_config.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate full-track selected inference configs."
    )
    parser.add_argument("--baseline-run", required=True, help="Baseline run directory.")
    parser.add_argument("--candidate-run", required=True, help="Candidate run directory.")
    parser.add_argument(
        "--baseline-config",
        default=None,
        help="Baseline fulltrack_selected_inference_config.json.",
    )
    parser.add_argument(
        "--candidate-config",
        default=None,
        help="Candidate fulltrack_selected_inference_config.json.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for comparison outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists.")
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List detected configs for baseline-run and candidate-run without comparing.",
    )
    return parser.parse_args()


def candidate_config_paths(run_dir):
    run_dir = Path(run_dir).resolve()
    return [run_dir / relative_path for relative_path in ALTERNATIVE_CONFIG_RELATIVE_PATHS]


def detected_config_paths(run_dir):
    return [path for path in candidate_config_paths(run_dir) if path.is_file()]


def format_path_list(paths):
    if not paths:
        return "  (none)"
    return "\n".join(f"  - {path}" for path in paths)


def resolve_config_path(requested_config, run_dir, label):
    run_dir = Path(run_dir).resolve()
    alternatives = candidate_config_paths(run_dir)
    requested_path = Path(requested_config).resolve() if requested_config else None

    if requested_path and requested_path.is_file():
        return requested_path

    matches = detected_config_paths(run_dir)
    if len(matches) == 1:
        match = matches[0]
        requested_display = requested_path if requested_path else "(not provided)"
        print(
            f"WARNING: {label} Config path not found; using detected alternative config: {match} "
            f"(requested: {requested_display})",
            file=sys.stderr,
        )
        return match

    if len(matches) > 1:
        requested_display = requested_path if requested_path else "(not provided)"
        raise SystemExit(
            f"{label} config path not found or not provided: {requested_display}\n"
            f"Multiple alternative configs were found for run-dir: {run_dir}\n"
            f"{format_path_list(matches)}\n"
            f"Please specify --{label}-config explicitly."
        )

    requested_display = requested_path if requested_path else "(not provided)"
    raise SystemExit(
        f"{label} config not found.\n"
        f"- requested_path: {requested_display}\n"
        f"- run_dir: {run_dir}\n"
        "- alternative paths checked:\n"
        f"{format_path_list(alternatives)}\n"
        "Suggestion: run SCRIPTS/calibration/calibrate_fulltrack_thresholds.py first."
    )


def print_detected_configs(run_dir, label):
    run_dir = Path(run_dir).resolve()
    matches = detected_config_paths(run_dir)
    print(f"{label}_run: {run_dir}")
    print(f"{label}_configs_found:")
    print(format_path_list(matches))
    print(f"{label}_alternative_paths_checked:")
    print(format_path_list(candidate_config_paths(run_dir)))


def load_json(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(
            f"Config not found: {path}\n"
            "This run may not have been calibrated with full-track + NMS yet.\n"
            "Run SCRIPTS/calibration/calibrate_fulltrack_thresholds.py first."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file), path


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_keys(mapping, keys, context):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise SystemExit(f"{context} is missing required keys: {', '.join(missing)}")


def validate_schema(config, label):
    class_names = config.get("class_names")
    midi_labels = config.get("midi_labels") or config.get("labels")
    if class_names != EXPECTED_CLASS_NAMES:
        raise SystemExit(
            f"{label} uses unexpected class_names: {class_names}; "
            f"expected {EXPECTED_CLASS_NAMES}."
        )
    if midi_labels != EXPECTED_MIDI_LABELS:
        raise SystemExit(
            f"{label} uses unexpected MIDI labels: {midi_labels}; "
            f"expected {EXPECTED_MIDI_LABELS}."
        )
    if config.get("test_not_used_for_selection") is not True:
        raise SystemExit(f"{label} does not explicitly mark test_not_used_for_selection=true.")
    if config.get("selection_split") != "VAL":
        raise SystemExit(f"{label} selection_split is not VAL: {config.get('selection_split')}")
    if config.get("final_report_split") != "TEST":
        raise SystemExit(f"{label} final_report_split is not TEST: {config.get('final_report_split')}")


def class_map(rows, label):
    output = {}
    for row in rows:
        class_name = row.get("class")
        if class_name in output:
            raise SystemExit(f"{label} has duplicated class row: {class_name}")
        output[class_name] = row
    missing = [class_name for class_name in EXPECTED_CLASS_NAMES if class_name not in output]
    if missing:
        raise SystemExit(f"{label} missing class metrics: {', '.join(missing)}")
    return output


def extract_run_summary(config, config_path, requested_run, label):
    validate_schema(config, label)
    require_keys(
        config,
        [
            "selected_checkpoint",
            "threshold_global",
            "onset_nms_ms",
            "onset_tolerance_ms",
            "test_metrics",
        ],
        label,
    )
    test_metrics = config["test_metrics"]
    require_keys(test_metrics, ["selected_config", "by_class_selected_config"], f"{label}.test_metrics")
    selected = test_metrics["selected_config"]
    require_keys(
        selected,
        ["TP", "FP", "FN", "micro_precision", "micro_recall", "micro_f1", "macro_f1"],
        f"{label}.test_metrics.selected_config",
    )
    by_class = class_map(test_metrics["by_class_selected_config"], label)
    requested_run = Path(requested_run).resolve()
    config_run = Path(config.get("run_dir", requested_run)).resolve()
    if requested_run != config_run:
        print(f"WARNING: {label} --run path differs from config run_dir: {requested_run} != {config_run}")

    dataset_root = config.get("dataset_root_resolved") or config.get("dataset_root")
    return {
        "label": label,
        "run_dir": str(requested_run),
        "config_path": str(config_path),
        "dataset_root": dataset_root,
        "dataset_root_original_from_run_config": config.get("dataset_root_original_from_run_config"),
        "dataset_root_resolved": config.get("dataset_root_resolved") or dataset_root,
        "dataset_root_source": config.get("dataset_root_source"),
        "checkpoint": config["selected_checkpoint"],
        "threshold_global": float(config["threshold_global"]),
        "threshold_mode": config.get("threshold_mode", "global"),
        "onset_nms_ms": float(config["onset_nms_ms"]),
        "onset_tolerance_ms": float(config["onset_tolerance_ms"]),
        "selection_split": config["selection_split"],
        "final_report_split": config["final_report_split"],
        "test_not_used_for_selection": bool(config["test_not_used_for_selection"]),
        "metrics": {
            "TP": int(selected["TP"]),
            "FP": int(selected["FP"]),
            "FN": int(selected["FN"]),
            "micro_precision": float(selected["micro_precision"]),
            "micro_recall": float(selected["micro_recall"]),
            "micro_f1": float(selected["micro_f1"]),
            "macro_f1": float(selected["macro_f1"]),
        },
        "by_class": {
            class_name: {
                "midi_pitch": int(by_class[class_name].get("midi_pitch", EXPECTED_MIDI_LABELS[index])),
                "threshold": float(by_class[class_name].get("threshold", config["threshold_global"])),
                "onset_nms_ms": float(by_class[class_name].get("onset_nms_ms", config["onset_nms_ms"])),
                "TP": int(by_class[class_name]["TP"]),
                "FP": int(by_class[class_name]["FP"]),
                "FN": int(by_class[class_name]["FN"]),
                "precision": float(by_class[class_name]["precision"]),
                "recall": float(by_class[class_name]["recall"]),
                "f1": float(by_class[class_name]["f1"]),
            }
            for index, class_name in enumerate(EXPECTED_CLASS_NAMES)
        },
    }


def delta(candidate, baseline):
    return candidate - baseline


def verdict_from_delta(value):
    if value > EPS:
        return "improved"
    if value < -EPS:
        return "worse"
    return "tied"


def build_comparison(baseline, candidate):
    metric_rows = []
    for metric in ("micro_precision", "micro_recall", "micro_f1", "macro_f1"):
        baseline_value = baseline["metrics"][metric]
        candidate_value = candidate["metrics"][metric]
        metric_rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta(candidate_value, baseline_value),
                "verdict": verdict_from_delta(delta(candidate_value, baseline_value)),
            }
        )

    class_rows = []
    for class_name in EXPECTED_CLASS_NAMES:
        baseline_row = baseline["by_class"][class_name]
        candidate_row = candidate["by_class"][class_name]
        f1_delta = delta(candidate_row["f1"], baseline_row["f1"])
        class_rows.append(
            {
                "class": class_name,
                "midi_pitch": baseline_row["midi_pitch"],
                "focus": class_name in FOCUS_CLASSES,
                "baseline_f1": baseline_row["f1"],
                "candidate_f1": candidate_row["f1"],
                "delta_f1": f1_delta,
                "verdict": verdict_from_delta(f1_delta),
                "baseline_TP": baseline_row["TP"],
                "baseline_FP": baseline_row["FP"],
                "baseline_FN": baseline_row["FN"],
                "candidate_TP": candidate_row["TP"],
                "candidate_FP": candidate_row["FP"],
                "candidate_FN": candidate_row["FN"],
                "delta_TP": candidate_row["TP"] - baseline_row["TP"],
                "delta_FP": candidate_row["FP"] - baseline_row["FP"],
                "delta_FN": candidate_row["FN"] - baseline_row["FN"],
                "baseline_precision": baseline_row["precision"],
                "candidate_precision": candidate_row["precision"],
                "delta_precision": candidate_row["precision"] - baseline_row["precision"],
                "baseline_recall": baseline_row["recall"],
                "candidate_recall": candidate_row["recall"],
                "delta_recall": candidate_row["recall"] - baseline_row["recall"],
            }
        )

    micro_f1_delta = candidate["metrics"]["micro_f1"] - baseline["metrics"]["micro_f1"]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "overall_verdict": {
            "primary_metric": "TEST full-track onset micro-F1 at 50 ms tolerance",
            "candidate_beats_baseline": micro_f1_delta > EPS,
            "micro_f1_delta": micro_f1_delta,
            "status": verdict_from_delta(micro_f1_delta),
        },
        "metric_comparison": metric_rows,
        "class_comparison": class_rows,
    }


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def signed_fmt(value, digits=4):
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{float(value):.{digits}f}"


def write_markdown(path, comparison):
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    verdict = comparison["overall_verdict"]
    status_text = "YES" if verdict["candidate_beats_baseline"] else "NO"

    lines = [
        "# Full-track run comparison",
        "",
        "## Verdict",
        "",
        f"- Candidate beats baseline on TEST micro-F1: `{status_text}`",
        f"- baseline micro-F1: `{fmt(baseline['metrics']['micro_f1'])}`",
        f"- candidate micro-F1: `{fmt(candidate['metrics']['micro_f1'])}`",
        f"- delta micro-F1: `{signed_fmt(verdict['micro_f1_delta'])}`",
        f"- selection_split: `{baseline['selection_split']}` / `{candidate['selection_split']}`",
        "- TEST was not used for selection in either config.",
        "",
        "## Selected Inference Config",
        "",
        "| run | checkpoint | threshold | NMS ms | tolerance ms |",
        "| --- | --- | ---: | ---: | ---: |",
        (
            f"| baseline | `{baseline['checkpoint']}` | {baseline['threshold_global']:g} | "
            f"{baseline['onset_nms_ms']:g} | {baseline['onset_tolerance_ms']:g} |"
        ),
        (
            f"| candidate | `{candidate['checkpoint']}` | {candidate['threshold_global']:g} | "
            f"{candidate['onset_nms_ms']:g} | {candidate['onset_tolerance_ms']:g} |"
        ),
        "",
        "## Dataset Roots",
        "",
        f"- baseline: `{baseline.get('dataset_root_resolved') or baseline.get('dataset_root')}`",
        f"- candidate: `{candidate.get('dataset_root_resolved') or candidate.get('dataset_root')}`",
        "",
        "## TEST Aggregate Metrics",
        "",
        "| metric | baseline | candidate | delta | verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparison["metric_comparison"]:
        lines.append(
            f"| {row['metric']} | {fmt(row['baseline'])} | {fmt(row['candidate'])} | "
            f"{signed_fmt(row['delta'])} | {row['verdict']} |"
        )

    lines.extend(
        [
            "",
            "## TEST Class Metrics",
            "",
            "| class | F1 baseline | F1 candidate | delta F1 | TP base/cand | FP base/cand | FN base/cand | verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparison["class_comparison"]:
        class_label = f"**{row['class']}**" if row["focus"] else row["class"]
        lines.append(
            f"| {class_label} | {fmt(row['baseline_f1'])} | {fmt(row['candidate_f1'])} | "
            f"{signed_fmt(row['delta_f1'])} | {row['baseline_TP']}/{row['candidate_TP']} | "
            f"{row['baseline_FP']}/{row['candidate_FP']} | {row['baseline_FN']}/{row['candidate_FN']} | "
            f"{row['verdict']} |"
        )

    focus_rows = [row for row in comparison["class_comparison"] if row["focus"]]
    lines.extend(["", "## Focus Classes", ""])
    for row in focus_rows:
        lines.append(
            f"- {row['class']}: F1 `{fmt(row['baseline_f1'])}` -> "
            f"`{fmt(row['candidate_f1'])}` (`{signed_fmt(row['delta_f1'])}`); "
            f"TP/FP/FN `{row['baseline_TP']}/{row['baseline_FP']}/{row['baseline_FN']}` -> "
            f"`{row['candidate_TP']}/{row['candidate_FP']}/{row['candidate_FN']}`."
        )

    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- baseline_config: `{baseline['config_path']}`",
            f"- candidate_config: `{candidate['config_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"Output dir already exists; use --overwrite: {output_dir}")
        if not output_dir.is_dir():
            raise SystemExit(f"Output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main():
    args = parse_args()
    if args.list_configs:
        print_detected_configs(args.baseline_run, "baseline")
        print()
        print_detected_configs(args.candidate_run, "candidate")
        return

    if not args.output_dir:
        raise SystemExit("--output-dir is required unless --list-configs is used.")

    baseline_path = resolve_config_path(args.baseline_config, args.baseline_run, "baseline")
    candidate_path = resolve_config_path(args.candidate_config, args.candidate_run, "candidate")
    baseline_config, baseline_path = load_json(baseline_path)
    candidate_config, candidate_path = load_json(candidate_path)
    baseline = extract_run_summary(baseline_config, baseline_path, args.baseline_run, "baseline")
    candidate = extract_run_summary(candidate_config, candidate_path, args.candidate_run, "candidate")
    comparison = build_comparison(baseline, candidate)

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    save_json(output_dir / "comparison_summary.json", comparison)
    write_csv(
        output_dir / "aggregate_metrics_comparison.csv",
        comparison["metric_comparison"],
        ["metric", "baseline", "candidate", "delta", "verdict"],
    )
    write_csv(
        output_dir / "class_metrics_comparison.csv",
        comparison["class_comparison"],
        [
            "class",
            "midi_pitch",
            "focus",
            "baseline_f1",
            "candidate_f1",
            "delta_f1",
            "verdict",
            "baseline_TP",
            "baseline_FP",
            "baseline_FN",
            "candidate_TP",
            "candidate_FP",
            "candidate_FN",
            "delta_TP",
            "delta_FP",
            "delta_FN",
            "baseline_precision",
            "candidate_precision",
            "delta_precision",
            "baseline_recall",
            "candidate_recall",
            "delta_recall",
        ],
    )
    write_markdown(output_dir / "README.md", comparison)
    write_markdown(output_dir / "comparison_summary.md", comparison)

    verdict = comparison["overall_verdict"]
    print(f"output_dir: {output_dir}")
    print(f"baseline_config: {baseline_path}")
    print(f"candidate_config: {candidate_path}")
    print(f"candidate_beats_baseline: {verdict['candidate_beats_baseline']}")
    print(f"baseline_micro_f1: {baseline['metrics']['micro_f1']:.6f}")
    print(f"candidate_micro_f1: {candidate['metrics']['micro_f1']:.6f}")
    print(f"delta_micro_f1: {verdict['micro_f1_delta']:+.6f}")
    for row in comparison["class_comparison"]:
        if row["focus"]:
            print(
                f"{row['class']}_f1: {row['baseline_f1']:.6f} -> "
                f"{row['candidate_f1']:.6f} ({row['delta_f1']:+.6f})"
            )


if __name__ == "__main__":
    main()
