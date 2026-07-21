"""
Full-track threshold calibration for calibrated custom_split runs.

Selection is performed on full VAL tracks only. TEST is evaluated afterwards
with the checkpoint and thresholds frozen from VAL.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

import numpy as np

from SCRIPTS.common.dataset_roots import resolve_dataset_root
from SCRIPTS.analysis.analyze_run_errors import (
    CHECKPOINT_METADATA_FILENAME,
    CLASS_NAMES,
    CLASS_SCHEMA,
    LABELS,
    build_model_and_data,
    counts_match,
    detect_onset_frames,
    detect_onset_frames_with_nms,
    detailed_match_events,
    empty_class_counts,
    ensure_output_dir,
    get_split_tracks,
    get_track_id,
    group_ground_truth_events,
    load_ground_truth_events_from_txt,
    load_json,
    require_vibro_5_toms,
    resolve_nms_ms,
    save_json,
    safe_metric,
    validate_checkpoint_schema,
    write_csv,
)


SELECTION_RULE = "maximize selected metric -> maximize precision -> minimize FP -> choose highest threshold"
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate onset thresholds on full VAL tracks and report full TEST metrics."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root override. Useful for old runs that point to PHRASE_GENERATOR.",
    )
    parser.add_argument("--onset-tolerance-ms", type=float, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument(
        "--threshold-mode",
        choices=["global", "per-class", "both"],
        default="global",
        help="Calibrate one global threshold, independent per-class thresholds, or both.",
    )
    parser.add_argument("--selection-metric", choices=["f1", "fbeta"], default="f1")
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--onset-nms-ms", type=float, default=None)
    parser.add_argument("--min-peak-distance-ms", type=float, default=None)
    parser.add_argument("--nms-grid-ms", type=float, nargs="*", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_thresholds(thresholds):
    unique = list(dict.fromkeys(float(threshold) for threshold in thresholds))
    invalid = [threshold for threshold in unique if threshold < 0.0 or threshold > 1.0]
    if invalid:
        raise ValueError(f"Thresholds must be between 0 and 1: {invalid}")
    return unique


def validate_nms_grid(nms_grid_ms):
    if nms_grid_ms is None:
        return []
    unique = list(dict.fromkeys(float(value) for value in nms_grid_ms))
    invalid = [value for value in unique if value < 0.0]
    if invalid:
        raise ValueError(f"NMS grid values must be non-negative: {invalid}")
    return unique


def load_json_optional(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return load_json(path)


def load_csv_optional(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} not found: {path}")
    return path


def require_dataset_split(dataset_root, split):
    split_dir = Path(dataset_root) / split.upper()
    audio_dir = split_dir / "AUDIO"
    annotation_dir = split_dir / "ANNOTATIONS"
    missing = [
        f"{name} ({path})"
        for name, path in (("AUDIO", audio_dir), ("ANNOTATIONS", annotation_dir))
        if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"Required {split.upper()} split folders are missing: {', '.join(missing)}"
        )


def previous_eval_artifacts(run_dir):
    artifacts = {
        "selected_thresholds_from_val": load_json_optional(
            run_dir / "selected_thresholds_from_val.json"
        ),
        "checkpoint_selection_from_val": load_json_optional(
            run_dir / "checkpoint_selection_from_val.json"
        ),
        "val_onset_metrics_by_threshold": load_csv_optional(
            run_dir / "val_onset_metrics_by_threshold.csv"
        ),
        "test_onset_metrics_by_threshold": load_csv_optional(
            run_dir / "test_onset_metrics_by_threshold.csv"
        ),
    }
    available = sorted(name for name, value in artifacts.items() if value is not None)
    return artifacts, available


def fbeta_score(precision, recall, beta):
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def checkpoint_paths(run_dir):
    checkpoint_dir = Path(run_dir) / "checkpoints"
    candidates = {
        "best": checkpoint_dir / "best_weights.weights.h5",
        "final": checkpoint_dir / "final_weights.weights.h5",
    }
    available = {}
    for label, path in candidates.items():
        if path.is_file():
            validate_checkpoint_schema(path)
            available[label] = path
    if not available:
        raise FileNotFoundError(f"No compatible checkpoints found in {checkpoint_dir}")
    return available


def count_events(events):
    counts = empty_class_counts()
    for event in events:
        counts[event["class_name"]] += 1
    return counts


def predict_split(model, hparams, data_access, split):
    loader, indexes, tracks = get_split_tracks(data_access, hparams, split, max_tracks=None)
    predictions = {}
    for track_number, track in enumerate(tracks, start=1):
        track_id = get_track_id(track)
        prediction = np.asarray(model.predict(track, **hparams))
        predictions[track_id] = prediction[:, : len(CLASS_NAMES)]
        print(f"{split} predicted {track_number}/{len(indexes)}: {track_id}")
    return predictions


def evaluate_predictions(
    predictions_by_track,
    gt_by_track,
    thresholds,
    sample_rate,
    tolerance_ms,
    checkpoint,
    split,
    onset_nms_ms=0.0,
):
    tolerance_frames = int(round(float(tolerance_ms) * sample_rate / 1000.0))
    counts = {
        threshold: {
            class_name: {"TP": 0, "FP": 0, "FN": 0}
            for class_name in CLASS_NAMES
        }
        for threshold in thresholds
    }

    for track_id, predictions in predictions_by_track.items():
        track_gt = gt_by_track.get(track_id, {class_name: [] for class_name in CLASS_NAMES})
        for class_index, class_name in enumerate(CLASS_NAMES):
            target_events = track_gt[class_name]
            values = predictions[:, class_index]
            for threshold in thresholds:
                predicted_frames = detect_onset_frames_with_nms(
                    values,
                    threshold,
                    sample_rate,
                    onset_nms_ms,
                )
                tp_pairs, fp_frames, fn_events = detailed_match_events(
                    predicted_frames,
                    target_events,
                    tolerance_frames,
                )
                class_counts = counts[threshold][class_name]
                class_counts["TP"] += len(tp_pairs)
                class_counts["FP"] += len(fp_frames)
                class_counts["FN"] += len(fn_events)

    rows = []
    for threshold in thresholds:
        class_f1 = []
        class_fbeta = []
        totals = {"TP": 0, "FP": 0, "FN": 0}
        threshold_rows = []
        for class_name in CLASS_NAMES:
            class_counts = counts[threshold][class_name]
            tp = int(class_counts["TP"])
            fp = int(class_counts["FP"])
            fn = int(class_counts["FN"])
            precision, recall, f1 = safe_metric(tp, fp, fn)
            fbeta = fbeta_score(precision, recall, beta=1.0)
            row = {
                "threshold": float(threshold),
                "class": class_name,
                "midi_pitch": LABELS[CLASS_NAMES.index(class_name)],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fbeta": float(fbeta),
                "tolerance_ms": float(tolerance_ms),
                "onset_nms_ms": float(onset_nms_ms),
                "split": split.upper(),
                "checkpoint": checkpoint,
            }
            totals["TP"] += tp
            totals["FP"] += fp
            totals["FN"] += fn
            class_f1.append(f1)
            class_fbeta.append(fbeta)
            threshold_rows.append(row)

        micro_precision, micro_recall, micro_f1 = safe_metric(
            totals["TP"],
            totals["FP"],
            totals["FN"],
        )
        micro_fbeta = fbeta_score(micro_precision, micro_recall, beta=1.0)
        for row in threshold_rows:
            row.update(
                {
                    "micro_precision": float(micro_precision),
                    "micro_recall": float(micro_recall),
                    "micro_f1": float(micro_f1),
                    "micro_fbeta": float(micro_fbeta),
                    "macro_f1": float(np.mean(class_f1)),
                    "macro_fbeta": float(np.mean(class_fbeta)),
                }
            )
            rows.append(row)
    return rows


def add_requested_fbeta(rows, beta):
    for row in rows:
        row["fbeta"] = float(fbeta_score(float(row["precision"]), float(row["recall"]), beta))
    aggregates = {}
    for row in rows:
        key = (row["checkpoint"], row["split"], row["threshold"])
        aggregates.setdefault(key, []).append(row)
    for grouped_rows in aggregates.values():
        tp = sum(int(row["TP"]) for row in grouped_rows)
        fp = sum(int(row["FP"]) for row in grouped_rows)
        fn = sum(int(row["FN"]) for row in grouped_rows)
        precision, recall, _ = safe_metric(tp, fp, fn)
        micro_fbeta = fbeta_score(precision, recall, beta)
        macro_fbeta = float(np.mean([float(row["fbeta"]) for row in grouped_rows]))
        for row in grouped_rows:
            row["micro_fbeta"] = float(micro_fbeta)
            row["macro_fbeta"] = macro_fbeta
    return rows


def aggregate_threshold_rows(rows, beta):
    aggregate = {}
    for row in rows:
        key = (row["checkpoint"], row["threshold"])
        item = aggregate.setdefault(
            key,
            {
                "checkpoint": row["checkpoint"],
                "threshold": row["threshold"],
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "class_f1": [],
                "class_fbeta": [],
            },
        )
        item["TP"] += int(row["TP"])
        item["FP"] += int(row["FP"])
        item["FN"] += int(row["FN"])
        item["class_f1"].append(float(row["f1"]))
        item["class_fbeta"].append(float(row["fbeta"]))

    output = []
    for item in aggregate.values():
        precision, recall, f1 = safe_metric(item["TP"], item["FP"], item["FN"])
        fbeta = fbeta_score(precision, recall, beta=beta)
        output.append(
            {
                "checkpoint": item["checkpoint"],
                "threshold": item["threshold"],
                "TP": item["TP"],
                "FP": item["FP"],
                "FN": item["FN"],
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "micro_fbeta": fbeta,
                "macro_f1": float(np.mean(item["class_f1"])),
                "macro_fbeta": float(np.mean(item["class_fbeta"])),
            }
        )
    return output


def aggregate_threshold_nms_rows(rows, beta):
    aggregate = {}
    for row in rows:
        key = (row["checkpoint"], row["threshold"], row.get("onset_nms_ms", 0.0))
        item = aggregate.setdefault(
            key,
            {
                "checkpoint": row["checkpoint"],
                "threshold": row["threshold"],
                "onset_nms_ms": row.get("onset_nms_ms", 0.0),
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "class_f1": [],
                "class_fbeta": [],
            },
        )
        item["TP"] += int(row["TP"])
        item["FP"] += int(row["FP"])
        item["FN"] += int(row["FN"])
        item["class_f1"].append(float(row["f1"]))
        item["class_fbeta"].append(float(row["fbeta"]))

    output = []
    for item in aggregate.values():
        precision, recall, f1 = safe_metric(item["TP"], item["FP"], item["FN"])
        fbeta = fbeta_score(precision, recall, beta=beta)
        output.append(
            {
                "checkpoint": item["checkpoint"],
                "threshold": item["threshold"],
                "onset_nms_ms": item["onset_nms_ms"],
                "TP": item["TP"],
                "FP": item["FP"],
                "FN": item["FN"],
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "micro_fbeta": fbeta,
                "macro_f1": float(np.mean(item["class_f1"])),
                "macro_fbeta": float(np.mean(item["class_fbeta"])),
            }
        )
    return output


def candidate_metric(row, selection_metric):
    return float(row["micro_fbeta"] if selection_metric == "fbeta" else row["micro_f1"])


def class_candidate_metric(row, selection_metric):
    return float(row["fbeta"] if selection_metric == "fbeta" else row["f1"])


def better_candidate(candidate, current, metric_value):
    if current is None:
        return True
    current_metric = current["_selection_metric_value"]
    if metric_value > current_metric + EPS:
        return True
    if metric_value < current_metric - EPS:
        return False
    if float(candidate["micro_precision"]) > float(current["micro_precision"]) + EPS:
        return True
    if float(candidate["micro_precision"]) < float(current["micro_precision"]) - EPS:
        return False
    if int(candidate["FP"]) < int(current["FP"]):
        return True
    if int(candidate["FP"]) > int(current["FP"]):
        return False
    return float(candidate["threshold"]) > float(current["threshold"])


def select_checkpoint_and_global_threshold(val_rows, selection_metric, beta):
    selected = None
    for row in aggregate_threshold_rows(val_rows, beta):
        metric_value = candidate_metric(row, selection_metric)
        candidate = dict(row)
        candidate["_selection_metric_value"] = metric_value
        if better_candidate(candidate, selected, metric_value):
            selected = candidate
    if selected is None:
        raise ValueError("No VAL rows available for checkpoint/threshold selection")
    selected.pop("_selection_metric_value", None)
    return selected


def select_checkpoint_threshold_nms(val_rows, selection_metric, beta):
    selected = None
    for row in aggregate_threshold_nms_rows(val_rows, beta):
        metric_value = candidate_metric(row, selection_metric)
        candidate = dict(row)
        candidate["_selection_metric_value"] = metric_value
        if better_candidate(candidate, selected, metric_value):
            selected = candidate
        elif selected is not None and abs(metric_value - selected["_selection_metric_value"]) <= EPS:
            if (
                abs(float(candidate["micro_precision"]) - float(selected["micro_precision"])) <= EPS
                and int(candidate["FP"]) == int(selected["FP"])
                and abs(float(candidate["threshold"]) - float(selected["threshold"])) <= EPS
                and float(candidate["onset_nms_ms"]) < float(selected["onset_nms_ms"])
            ):
                selected = candidate
    if selected is None:
        raise ValueError("No VAL rows available for NMS selection")
    selected.pop("_selection_metric_value", None)
    return selected


def select_per_class_thresholds(val_rows, checkpoint, selection_metric):
    selected = {}
    for class_name in CLASS_NAMES:
        best = None
        for row in val_rows:
            if row["checkpoint"] != checkpoint or row["class"] != class_name:
                continue
            metric_value = class_candidate_metric(row, selection_metric)
            candidate = dict(row)
            candidate["_selection_metric_value"] = metric_value
            if best is None:
                best = candidate
                continue
            current_metric = best["_selection_metric_value"]
            if metric_value > current_metric + EPS:
                best = candidate
            elif abs(metric_value - current_metric) <= EPS:
                if float(candidate["precision"]) > float(best["precision"]) + EPS:
                    best = candidate
                elif abs(float(candidate["precision"]) - float(best["precision"])) <= EPS:
                    if int(candidate["FP"]) < int(best["FP"]):
                        best = candidate
                    elif int(candidate["FP"]) == int(best["FP"]) and float(candidate["threshold"]) > float(best["threshold"]):
                        best = candidate
        if best is None:
            raise ValueError(f"No VAL rows for class {class_name}")
        best.pop("_selection_metric_value", None)
        selected[class_name] = best
    return selected


def select_per_class_thresholds_for_config(val_rows, checkpoint, onset_nms_ms, selection_metric):
    selected = {}
    for class_name in CLASS_NAMES:
        best = None
        for row in val_rows:
            if (
                row["checkpoint"] != checkpoint
                or row["class"] != class_name
                or abs(float(row.get("onset_nms_ms", 0.0)) - float(onset_nms_ms)) > 1e-12
            ):
                continue
            metric_value = class_candidate_metric(row, selection_metric)
            candidate = dict(row)
            candidate["_selection_metric_value"] = metric_value
            if best is None:
                best = candidate
                continue
            current_metric = best["_selection_metric_value"]
            if metric_value > current_metric + EPS:
                best = candidate
            elif abs(metric_value - current_metric) <= EPS:
                if float(candidate["precision"]) > float(best["precision"]) + EPS:
                    best = candidate
                elif abs(float(candidate["precision"]) - float(best["precision"])) <= EPS:
                    if int(candidate["FP"]) < int(best["FP"]):
                        best = candidate
                    elif int(candidate["FP"]) == int(best["FP"]) and float(candidate["threshold"]) > float(best["threshold"]):
                        best = candidate
        if best is None:
            raise ValueError(
                f"No VAL rows for class {class_name}, checkpoint {checkpoint}, NMS {onset_nms_ms}"
            )
        best.pop("_selection_metric_value", None)
        selected[class_name] = best
    return selected


def better_per_class_config(candidate, current, metric_value):
    if current is None:
        return True
    current_metric = current["_selection_metric_value"]
    if metric_value > current_metric + EPS:
        return True
    if metric_value < current_metric - EPS:
        return False
    if float(candidate["micro_precision"]) > float(current["micro_precision"]) + EPS:
        return True
    if float(candidate["micro_precision"]) < float(current["micro_precision"]) - EPS:
        return False
    if int(candidate["FP"]) < int(current["FP"]):
        return True
    if int(candidate["FP"]) > int(current["FP"]):
        return False
    if float(candidate["onset_nms_ms"]) < float(current["onset_nms_ms"]) - EPS:
        return True
    if float(candidate["onset_nms_ms"]) > float(current["onset_nms_ms"]) + EPS:
        return False
    candidate_threshold = min(float(value) for value in candidate["thresholds_per_class"].values())
    current_threshold = min(float(value) for value in current["thresholds_per_class"].values())
    return candidate_threshold > current_threshold


def select_checkpoint_nms_per_class(val_rows, selection_metric, beta):
    selected = None
    config_keys = sorted(
        {
            (row["checkpoint"], float(row.get("onset_nms_ms", 0.0)))
            for row in val_rows
        },
        key=lambda item: (item[0], item[1]),
    )
    for checkpoint, onset_nms_ms in config_keys:
        selected_by_class = select_per_class_thresholds_for_config(
            val_rows,
            checkpoint,
            onset_nms_ms,
            selection_metric,
        )
        thresholds_by_class = {
            class_name: float(row["threshold"])
            for class_name, row in selected_by_class.items()
        }
        selected_rows = filter_per_class_nms(
            val_rows,
            checkpoint,
            thresholds_by_class,
            onset_nms_ms,
            beta,
        )
        metrics = summarize_selected(selected_rows)
        metric_value = float(metrics["micro_fbeta"] if selection_metric == "fbeta" else metrics["micro_f1"])
        candidate = {
            "checkpoint": checkpoint,
            "onset_nms_ms": float(onset_nms_ms),
            "thresholds_per_class": thresholds_by_class,
            "selected_rows_by_class": selected_by_class,
            **metrics,
            "_selection_metric_value": metric_value,
        }
        if better_per_class_config(candidate, selected, metric_value):
            selected = candidate
    if selected is None:
        raise ValueError("No VAL rows available for per-class selection")
    selected.pop("_selection_metric_value", None)
    return selected


def filter_global(rows, checkpoint, threshold):
    selected = [
        dict(row)
        for row in rows
        if row["checkpoint"] == checkpoint and abs(float(row["threshold"]) - float(threshold)) < 1e-12
    ]
    return selected


def filter_global_nms(rows, checkpoint, threshold, onset_nms_ms):
    return [
        dict(row)
        for row in rows
        if row["checkpoint"] == checkpoint
        and abs(float(row["threshold"]) - float(threshold)) < 1e-12
        and abs(float(row.get("onset_nms_ms", 0.0)) - float(onset_nms_ms)) < 1e-12
    ]


def add_aggregate_metrics(selected, beta):
    if not selected:
        return []
    tp = sum(int(row["TP"]) for row in selected)
    fp = sum(int(row["FP"]) for row in selected)
    fn = sum(int(row["FN"]) for row in selected)
    precision, recall, f1 = safe_metric(tp, fp, fn)
    micro_fbeta = fbeta_score(precision, recall, beta=beta)
    macro_f1 = float(np.mean([float(row["f1"]) for row in selected]))
    macro_fbeta = float(np.mean([float(row["fbeta"]) for row in selected]))
    for row in selected:
        row.update(
            {
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "micro_fbeta": micro_fbeta,
                "macro_f1": macro_f1,
                "macro_fbeta": macro_fbeta,
            }
        )
    return selected


def filter_per_class(rows, checkpoint, thresholds_by_class, beta):
    selected = []
    for row in rows:
        if row["checkpoint"] != checkpoint:
            continue
        threshold = thresholds_by_class.get(row["class"])
        if threshold is not None and abs(float(row["threshold"]) - float(threshold)) < 1e-12:
            selected.append(dict(row))
    if not selected:
        return []
    return add_aggregate_metrics(selected, beta)


def filter_per_class_nms(rows, checkpoint, thresholds_by_class, onset_nms_ms, beta):
    selected = []
    for row in rows:
        if row["checkpoint"] != checkpoint:
            continue
        if abs(float(row.get("onset_nms_ms", 0.0)) - float(onset_nms_ms)) > 1e-12:
            continue
        threshold = thresholds_by_class.get(row["class"])
        if threshold is not None and abs(float(row["threshold"]) - float(threshold)) < 1e-12:
            selected.append(dict(row))
    return add_aggregate_metrics(selected, beta)


def metric_fields():
    return [
        "threshold",
        "class",
        "midi_pitch",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "fbeta",
        "tolerance_ms",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_fbeta",
        "macro_f1",
        "macro_fbeta",
        "split",
        "checkpoint",
    ]


def nms_metric_fields():
    return metric_fields()[:11] + ["onset_nms_ms"] + metric_fields()[11:]


def make_gt_check(split, dataset_root, gt_events, predictions_by_checkpoint):
    counts = count_events(gt_events)
    track_ids = sorted({event["track_id"] for event in gt_events})
    prediction_tracks = {
        checkpoint: sorted(predictions)
        for checkpoint, predictions in predictions_by_checkpoint.items()
    }
    return {
        "split": split,
        "dataset_root": str(dataset_root),
        "counts_from_annotations": counts,
        "n_annotation_tracks": len(track_ids),
        "n_prediction_tracks_by_checkpoint": {
            checkpoint: len(ids) for checkpoint, ids in prediction_tracks.items()
        },
        "missing_prediction_tracks_by_checkpoint": {
            checkpoint: sorted(set(track_ids) - set(ids))
            for checkpoint, ids in prediction_tracks.items()
        },
        "extra_prediction_tracks_by_checkpoint": {
            checkpoint: sorted(set(ids) - set(track_ids))
            for checkpoint, ids in prediction_tracks.items()
        },
        "status": "match"
        if all(set(ids) == set(track_ids) for ids in prediction_tracks.values())
        else "mismatch",
    }


def summarize_selected(rows):
    if not rows:
        return {}
    first = rows[0]
    return {
        "TP": sum(int(row["TP"]) for row in rows),
        "FP": sum(int(row["FP"]) for row in rows),
        "FN": sum(int(row["FN"]) for row in rows),
        "micro_precision": float(first["micro_precision"]),
        "micro_recall": float(first["micro_recall"]),
        "micro_f1": float(first["micro_f1"]),
        "micro_fbeta": float(first["micro_fbeta"]),
        "macro_f1": float(first["macro_f1"]),
        "macro_fbeta": float(first["macro_fbeta"]),
    }


def markdown_table(rows, columns):
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    path,
    run_dir,
    dataset_root,
    args,
    selected_global,
    selected_per_class,
    val_global_rows,
    val_per_class_rows,
    test_global_rows,
    test_per_class_rows,
    old_comparison,
    previous_artifacts_status,
    dataset_root_info=None,
):
    thresholds_by_class = {
        class_name: float(row["threshold"])
        for class_name, row in selected_per_class.items()
    }
    lines = [
        f"# Full-track calibration: {Path(run_dir).name}",
        "",
        f"- dataset: `{dataset_root}`",
        f"- dataset_root_original_from_run_config: `{(dataset_root_info or {}).get('dataset_root_original_from_run_config')}`",
        f"- dataset_root_resolved: `{(dataset_root_info or {}).get('dataset_root_resolved', dataset_root)}`",
        f"- dataset_root_source: `{(dataset_root_info or {}).get('dataset_root_source', 'config')}`",
        "- selection_split: `VAL`",
        "- final_report_split: `TEST`",
        f"- selection_metric: `{args.selection_metric}`",
        f"- beta: `{args.beta}`",
        f"- selection_rule: `{SELECTION_RULE}`",
        "- WARNING: TEST was not used for checkpoint or threshold selection.",
        "",
        "## Selected From Full VAL",
        f"- checkpoint: `{selected_global['checkpoint']}`",
        f"- global_threshold: `{selected_global['threshold']}`",
        f"- per_class_thresholds: `{thresholds_by_class}`",
        "",
        "## Full VAL Metrics At Selected Thresholds",
        f"- global: `{summarize_selected(val_global_rows)}`",
        f"- per-class: `{summarize_selected(val_per_class_rows)}`",
        "",
        "## Full TEST Metrics With Frozen VAL Selection",
        f"- global: `{summarize_selected(test_global_rows)}`",
        f"- per-class: `{summarize_selected(test_per_class_rows)}`",
        "",
        "## TEST By Class: VAL Global Threshold",
        markdown_table(
            test_global_rows,
            ["class", "threshold", "TP", "FP", "FN", "precision", "recall", "f1", "fbeta"],
        ),
        "",
        "## TEST By Class: VAL Per-Class Thresholds",
        markdown_table(
            sorted(test_per_class_rows, key=lambda row: CLASS_NAMES.index(row["class"])),
            ["class", "threshold", "TP", "FP", "FN", "precision", "recall", "f1", "fbeta"],
        ),
        "",
        "## Comparison Against Previous Run Thresholds",
        f"- previous_eval_artifacts: `{previous_artifacts_status}`",
        f"- comparison: `{old_comparison}`",
    ]
    if previous_artifacts_status == "not_available":
        lines.extend(
            [
                "",
                "No previous eval_steps calibration artifacts were found. This is expected for runs trained with the new protocol.",
            ]
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def selected_metric_value(rows, selection_metric):
    summary = summarize_selected(rows)
    if not summary:
        return float("-inf")
    return float(summary["micro_fbeta"] if selection_metric == "fbeta" else summary["micro_f1"])


def mode_wins(candidate_rows, current_rows, selection_metric):
    candidate_metric = selected_metric_value(candidate_rows, selection_metric)
    current_metric = selected_metric_value(current_rows, selection_metric)
    if candidate_metric > current_metric + EPS:
        return True
    if candidate_metric < current_metric - EPS:
        return False
    candidate = summarize_selected(candidate_rows)
    current = summarize_selected(current_rows)
    if float(candidate.get("micro_precision", 0.0)) > float(current.get("micro_precision", 0.0)) + EPS:
        return True
    if float(candidate.get("micro_precision", 0.0)) < float(current.get("micro_precision", 0.0)) - EPS:
        return False
    return int(candidate.get("FP", 0)) < int(current.get("FP", 0))


def class_delta_rows(global_rows, per_class_rows):
    by_global = {row["class"]: row for row in global_rows}
    by_per_class = {row["class"]: row for row in per_class_rows}
    output = []
    for class_name in CLASS_NAMES:
        global_row = by_global.get(class_name, {})
        per_class_row = by_per_class.get(class_name, {})
        output.append(
            {
                "class": class_name,
                "midi_pitch": LABELS[CLASS_NAMES.index(class_name)],
                "global_threshold": global_row.get("threshold"),
                "per_class_threshold": per_class_row.get("threshold"),
                "global_f1": global_row.get("f1"),
                "per_class_f1": per_class_row.get("f1"),
                "delta_f1": (
                    float(per_class_row["f1"]) - float(global_row["f1"])
                    if global_row and per_class_row
                    else None
                ),
                "global_FP": global_row.get("FP"),
                "per_class_FP": per_class_row.get("FP"),
                "global_FN": global_row.get("FN"),
                "per_class_FN": per_class_row.get("FN"),
            }
        )
    return output


def write_combined_summary(
    path,
    run_dir,
    dataset_root,
    args,
    selected_global,
    selected_per_class_config,
    val_global_rows,
    val_per_class_rows,
    test_global_rows,
    test_per_class_rows,
    selected_alias_mode,
    previous_artifacts_status,
    dataset_root_info=None,
):
    test_deltas = class_delta_rows(test_global_rows, test_per_class_rows)
    tom_deltas = [row for row in test_deltas if row["class"] in ("T12", "T14", "T16")]
    improved_toms = [
        row["class"]
        for row in tom_deltas
        if row["delta_f1"] is not None and row["delta_f1"] > EPS
    ]
    lines = [
        f"# Full-track calibration: {Path(run_dir).name}",
        "",
        f"- dataset: `{dataset_root}`",
        f"- dataset_root_original_from_run_config: `{(dataset_root_info or {}).get('dataset_root_original_from_run_config')}`",
        f"- dataset_root_resolved: `{(dataset_root_info or {}).get('dataset_root_resolved', dataset_root)}`",
        f"- dataset_root_source: `{(dataset_root_info or {}).get('dataset_root_source', 'config')}`",
        "- selection_split: `VAL`",
        "- final_report_split: `TEST`",
        "- test_not_used_for_selection: `true`",
        f"- threshold_mode_requested: `{args.threshold_mode}`",
        f"- selection_metric: `{args.selection_metric}`",
        f"- beta: `{args.beta}`",
        f"- selection_rule: `{SELECTION_RULE}`",
        f"- selected_inference_config_alias: `{selected_alias_mode}`",
        f"- previous_eval_artifacts: `{previous_artifacts_status}`",
        "",
        "## Best Global From VAL",
        f"- checkpoint: `{selected_global['checkpoint']}`",
        f"- threshold: `{selected_global['threshold']}`",
        f"- onset_nms_ms: `{selected_global.get('onset_nms_ms', 0.0)}`",
        f"- VAL: `{summarize_selected(val_global_rows)}`",
        f"- TEST frozen from VAL: `{summarize_selected(test_global_rows)}`",
        "",
        "## Best Per-Class From VAL",
        f"- checkpoint: `{selected_per_class_config['checkpoint']}`",
        f"- thresholds_per_class: `{selected_per_class_config['thresholds_per_class']}`",
        f"- onset_nms_ms: `{selected_per_class_config.get('onset_nms_ms', 0.0)}`",
        f"- VAL: `{summarize_selected(val_per_class_rows)}`",
        f"- TEST frozen from VAL: `{summarize_selected(test_per_class_rows)}`",
        "",
        "## TEST By Class: Global",
        markdown_table(
            sorted(test_global_rows, key=lambda row: CLASS_NAMES.index(row["class"])),
            ["class", "midi_pitch", "threshold", "onset_nms_ms", "TP", "FP", "FN", "precision", "recall", "f1", "fbeta"],
        ),
        "",
        "## TEST By Class: Per-Class",
        markdown_table(
            sorted(test_per_class_rows, key=lambda row: CLASS_NAMES.index(row["class"])),
            ["class", "midi_pitch", "threshold", "onset_nms_ms", "TP", "FP", "FN", "precision", "recall", "f1", "fbeta"],
        ),
        "",
        "## TEST Delta Per-Class Minus Global",
        markdown_table(
            test_deltas,
            [
                "class",
                "midi_pitch",
                "global_threshold",
                "per_class_threshold",
                "global_f1",
                "per_class_f1",
                "delta_f1",
                "global_FP",
                "per_class_FP",
                "global_FN",
                "per_class_FN",
            ],
        ),
        "",
        "## Toms",
        f"- toms_with_higher_TEST_f1_under_per_class: `{improved_toms}`",
        "- TEST is shown only as a frozen report after VAL selection; it did not choose the mode, checkpoint, thresholds, or NMS.",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def diff_summary(baseline_rows, selected_rows):
    baseline = summarize_selected(baseline_rows)
    selected = summarize_selected(selected_rows)
    keys = ["TP", "FP", "FN", "micro_precision", "micro_recall", "micro_f1", "micro_fbeta", "macro_f1", "macro_fbeta"]
    return {
        key: {
            "without_nms": baseline.get(key),
            "with_selected_nms": selected.get(key),
            "delta": selected.get(key, 0) - baseline.get(key, 0)
            if baseline.get(key) is not None and selected.get(key) is not None
            else None,
        }
        for key in keys
    }


def write_nms_summary(
    path,
    run_dir,
    dataset_root,
    args,
    selected_config,
    val_selected_rows,
    test_selected_rows,
    test_without_nms_rows,
    previous_artifacts_status="not_checked",
    dataset_root_info=None,
):
    lines = [
        f"# Full-track NMS calibration: {Path(run_dir).name}",
        "",
        f"- dataset: `{dataset_root}`",
        f"- dataset_root_original_from_run_config: `{(dataset_root_info or {}).get('dataset_root_original_from_run_config')}`",
        f"- dataset_root_resolved: `{(dataset_root_info or {}).get('dataset_root_resolved', dataset_root)}`",
        f"- dataset_root_source: `{(dataset_root_info or {}).get('dataset_root_source', 'config')}`",
        "- selection_split: `VAL`",
        "- final_report_split: `TEST`",
        f"- selection_metric: `{args.selection_metric}`",
        f"- beta: `{args.beta}`",
        f"- selection_rule: `{SELECTION_RULE}`",
        "- WARNING: TEST was not used for checkpoint, threshold, or NMS selection.",
        "- TEST no participa en seleccion.",
        "- Esta configuracion es la recomendada para inferencia.",
        f"- previous_eval_artifacts: `{previous_artifacts_status}`",
        "",
        "## Selected From Full VAL",
        f"- checkpoint: `{selected_config['checkpoint']}`",
        f"- threshold: `{selected_config['threshold']}`",
        f"- onset_nms_ms: `{selected_config['onset_nms_ms']}`",
        "",
        "## Full VAL Metrics At Selected Config",
        f"- selected: `{summarize_selected(val_selected_rows)}`",
        "",
        "## Full TEST Metrics With Frozen VAL Config",
        f"- without NMS at same checkpoint/threshold: `{summarize_selected(test_without_nms_rows)}`",
        f"- with selected NMS: `{summarize_selected(test_selected_rows)}`",
        f"- change: `{diff_summary(test_without_nms_rows, test_selected_rows)}`",
        "",
        "## TEST By Class: Selected VAL Config",
        markdown_table(
            test_selected_rows,
            ["class", "threshold", "onset_nms_ms", "TP", "FP", "FN", "precision", "recall", "f1", "fbeta"],
        ),
        "",
        "## Interpretation",
        "- NMS is applied per track and per class before onset matching.",
        "- The selected NMS value comes only from VAL complete tracks.",
        "- TEST is reported only after freezing checkpoint, threshold, and NMS from VAL.",
        "- This frozen VAL-selected configuration is recommended for inference.",
    ]
    if previous_artifacts_status == "not_available":
        lines.extend(
            [
                "",
                "No previous eval_steps calibration artifacts were found. This is expected for runs trained with the new protocol.",
            ]
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def make_inference_config(
    run_dir,
    dataset_root,
    checkpoints,
    threshold_mode,
    selected_checkpoint,
    global_threshold,
    per_class_thresholds,
    onset_nms_ms,
    tolerance_ms,
    val_metrics,
    test_metrics,
    selection_metric,
    beta,
    dataset_root_info=None,
):
    config = {
        "run_dir": str(Path(run_dir).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "class_schema": CLASS_SCHEMA,
        "class_names": CLASS_NAMES,
        "midi_labels": LABELS,
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_path": str(checkpoints[selected_checkpoint].resolve()),
        "threshold_global": float(global_threshold),
        "thresholds_per_class": per_class_thresholds,
        "threshold_mode": threshold_mode,
        "onset_nms_ms": float(onset_nms_ms),
        "min_peak_distance_ms": float(onset_nms_ms),
        "onset_tolerance_ms": float(tolerance_ms),
        "selection_split": "VAL",
        "final_report_split": "TEST",
        "selection_metric": selection_metric,
        "beta": float(beta),
        "test_not_used_for_selection": True,
        "recommended_for_inference": True,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if dataset_root_info:
        config.update(dataset_root_info)
    return config


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = ensure_output_dir(args.output_dir, args.overwrite)
    thresholds = validate_thresholds(args.thresholds)
    fixed_nms_ms = resolve_nms_ms(args.onset_nms_ms, args.min_peak_distance_ms)
    nms_grid_ms = validate_nms_grid(args.nms_grid_ms)
    if args.beta <= 0:
        raise ValueError("--beta must be positive")

    run_config_path = require_file(run_dir / "config.json", "run config.json")
    run_config = load_json(run_config_path)
    previous_artifacts, available_previous_artifacts = previous_eval_artifacts(run_dir)
    previous_thresholds = previous_artifacts["selected_thresholds_from_val"]
    previous_checkpoint_selection = previous_artifacts["checkpoint_selection_from_val"] or {}
    previous_artifacts_status = (
        ", ".join(available_previous_artifacts)
        if available_previous_artifacts
        else "not_available"
    )
    schema_reference = previous_thresholds or {
        "class_schema": CLASS_SCHEMA,
    }
    require_vibro_5_toms(run_config, schema_reference)
    if "dataset_root" not in run_config:
        raise KeyError(f"Required dataset_root missing from {run_config_path}")
    dataset_root_info = resolve_dataset_root(
        run_config["dataset_root"],
        cli_dataset_root=args.dataset_root,
        repo_root=REPO_ROOT,
    )
    dataset_root = Path(dataset_root_info["dataset_root_resolved"]).resolve()
    for split in ("VAL", "TEST"):
        require_dataset_split(dataset_root, split)
    if args.onset_tolerance_ms is None and "onset_tolerance_ms" not in run_config:
        raise KeyError(
            f"Required onset_tolerance_ms missing from {run_config_path}; "
            "pass --onset-tolerance-ms explicitly."
        )
    if "sample_rate" not in run_config:
        raise KeyError(f"Required sample_rate missing from {run_config_path}")
    tolerance_ms = float(args.onset_tolerance_ms if args.onset_tolerance_ms is not None else run_config["onset_tolerance_ms"])
    sample_rate = int(run_config["sample_rate"])

    checkpoints = checkpoint_paths(run_dir)
    model, hparams, data_access = build_model_and_data(run_config, dataset_root)

    gt_events_by_split = {
        split: load_ground_truth_events_from_txt(dataset_root, split)
        for split in ("VAL", "TEST")
    }
    gt_by_track = {
        split: group_ground_truth_events(events, sample_rate)
        for split, events in gt_events_by_split.items()
    }

    val_rows = []
    test_rows = []
    nms_val_rows = []
    nms_test_rows = []
    predictions_track_check = {"VAL": {}, "TEST": {}}
    for checkpoint, weights_path in checkpoints.items():
        print(f"\nLoading checkpoint {checkpoint}: {weights_path}")
        model.model.load_weights(str(weights_path))
        for split in ("VAL", "TEST"):
            predictions = predict_split(model, hparams, data_access, split)
            predictions_track_check[split][checkpoint] = predictions
            rows = evaluate_predictions(
                predictions,
                gt_by_track[split],
                thresholds,
                sample_rate,
                tolerance_ms,
                checkpoint,
                split,
                fixed_nms_ms,
            )
            add_requested_fbeta(rows, args.beta)
            if split == "VAL":
                val_rows.extend(rows)
            else:
                test_rows.extend(rows)
            for onset_nms_ms in nms_grid_ms:
                nms_rows = evaluate_predictions(
                    predictions,
                    gt_by_track[split],
                    thresholds,
                    sample_rate,
                    tolerance_ms,
                    checkpoint,
                    split,
                    onset_nms_ms,
                )
                add_requested_fbeta(nms_rows, args.beta)
                if split == "VAL":
                    nms_val_rows.extend(nms_rows)
                else:
                    nms_test_rows.extend(nms_rows)

    selection_val_rows = nms_val_rows if nms_grid_ms else val_rows
    selection_test_rows = nms_test_rows if nms_grid_ms else test_rows

    if nms_grid_ms:
        selected_global = select_checkpoint_threshold_nms(
            selection_val_rows,
            args.selection_metric,
            args.beta,
        )
    else:
        selected_global = select_checkpoint_and_global_threshold(
            selection_val_rows,
            args.selection_metric,
            args.beta,
        )
        selected_global["onset_nms_ms"] = float(fixed_nms_ms)
    selected_checkpoint = selected_global["checkpoint"]

    val_global_rows = filter_global_nms(
        selection_val_rows,
        selected_global["checkpoint"],
        selected_global["threshold"],
        selected_global.get("onset_nms_ms", fixed_nms_ms),
    )
    test_global_rows = filter_global_nms(
        selection_test_rows,
        selected_global["checkpoint"],
        selected_global["threshold"],
        selected_global.get("onset_nms_ms", fixed_nms_ms),
    )

    selected_per_class_config = select_checkpoint_nms_per_class(
        selection_val_rows,
        args.selection_metric,
        args.beta,
    )
    selected_per_class = selected_per_class_config["selected_rows_by_class"]
    selected_per_class_thresholds = selected_per_class_config["thresholds_per_class"]
    val_per_class_rows = filter_per_class_nms(
        selection_val_rows,
        selected_per_class_config["checkpoint"],
        selected_per_class_thresholds,
        selected_per_class_config["onset_nms_ms"],
        args.beta,
    )
    test_per_class_rows = filter_per_class_nms(
        selection_test_rows,
        selected_per_class_config["checkpoint"],
        selected_per_class_thresholds,
        selected_per_class_config["onset_nms_ms"],
        args.beta,
    )
    selected_alias_mode = "global"
    if args.threshold_mode == "per-class":
        selected_alias_mode = "per-class"
    elif args.threshold_mode == "both" and mode_wins(
        val_per_class_rows,
        val_global_rows,
        args.selection_metric,
    ):
        selected_alias_mode = "per-class"

    old_comparison = {
        "previous_eval_artifacts": previous_artifacts_status,
        "message": (
            "No previous eval_steps calibration artifacts were found. "
            "This is expected for runs trained with the new protocol."
        ),
        "fulltrack_test_with_new_global": summarize_selected(test_global_rows),
        "fulltrack_test_with_new_per_class": summarize_selected(test_per_class_rows),
    }
    if previous_thresholds is not None:
        try:
            previous_checkpoint = previous_thresholds.get(
                "selected_checkpoint",
                previous_checkpoint_selection.get("selected_checkpoint", selected_checkpoint),
            )
            previous_global_threshold = float(previous_thresholds["selected_global_threshold"])
            previous_per_class_thresholds = {
                key: float(value)
                for key, value in previous_thresholds["selected_per_class_thresholds"].items()
            }
            old_global_rows = filter_global(test_rows, previous_checkpoint, previous_global_threshold)
            old_per_class_rows = filter_per_class(
                test_rows, previous_checkpoint, previous_per_class_thresholds, args.beta
            )
            old_comparison = {
                "previous_eval_artifacts": previous_artifacts_status,
                "previous_checkpoint": previous_checkpoint,
                "previous_global_threshold": previous_global_threshold,
                "previous_per_class_thresholds": previous_per_class_thresholds,
                "fulltrack_test_with_previous_global": summarize_selected(old_global_rows),
                "fulltrack_test_with_previous_per_class": summarize_selected(old_per_class_rows),
                "fulltrack_test_with_new_global": summarize_selected(test_global_rows),
                "fulltrack_test_with_new_per_class": summarize_selected(test_per_class_rows),
            }
        except KeyError as error:
            old_comparison["message"] = (
                f"Previous eval_steps artifacts are incomplete ({error}); "
                "skipping historical threshold comparison."
            )

    selected_thresholds_payload = {
        "dataset_root": str(dataset_root),
        **dataset_root_info,
        "run_name": run_dir.name,
        "class_schema": CLASS_SCHEMA,
        "class_names": CLASS_NAMES,
        "labels": LABELS,
        "onset_tolerance_ms": tolerance_ms,
        "threshold_grid": thresholds,
        "selection_split": "VAL",
        "selection_scope": "full-track",
        "selection_metric": args.selection_metric,
        "beta": args.beta,
        "selection_rule": SELECTION_RULE,
        "selected_checkpoint": selected_checkpoint,
        "selected_global_threshold": float(selected_global["threshold"]),
        "selected_per_class_thresholds": selected_per_class_thresholds,
        "val_metrics_for_selected_global_threshold": val_global_rows,
        "val_metrics_for_selected_per_class_thresholds": selected_per_class,
    }
    checkpoint_selection_payload = {
        "dataset_root": str(dataset_root),
        **dataset_root_info,
        "run_name": run_dir.name,
        "class_schema": CLASS_SCHEMA,
        "checkpoints_evaluated": list(checkpoints),
        "selection_split": "VAL",
        "selection_scope": "full-track",
        "selection_metric": args.selection_metric,
        "beta": args.beta,
        "selection_rule": SELECTION_RULE,
        "selected_checkpoint": selected_checkpoint,
        "selected_global_threshold": float(selected_global["threshold"]),
        "val_metrics_selected_checkpoint_global": selected_global,
        "test_metrics_using_val_global_threshold": test_global_rows,
        "test_metrics_using_val_per_class_thresholds": test_per_class_rows,
    }
    consistency_payload = {
        "note": "Full-track calibration uses TXT annotations as GT and selects only from VAL.",
        **dataset_root_info,
        "val_ground_truth_check": make_gt_check("VAL", dataset_root, gt_events_by_split["VAL"], predictions_track_check["VAL"]),
        "test_ground_truth_check": make_gt_check("TEST", dataset_root, gt_events_by_split["TEST"], predictions_track_check["TEST"]),
        "previous_eval_artifacts": previous_artifacts_status,
        "previous_run_threshold_comparison": old_comparison,
        "test_not_used_for_selection": True,
    }

    write_csv(output_dir / "fulltrack_val_onset_metrics_by_threshold.csv", val_rows, nms_metric_fields())
    write_csv(output_dir / "fulltrack_test_onset_metrics_by_threshold.csv", test_rows, nms_metric_fields())
    write_csv(output_dir / "val_fulltrack_metrics_global.csv", val_global_rows, nms_metric_fields())
    write_csv(output_dir / "test_fulltrack_metrics_global.csv", test_global_rows, nms_metric_fields())
    write_csv(output_dir / "val_fulltrack_metrics_per_class.csv", val_per_class_rows, nms_metric_fields())
    write_csv(output_dir / "test_fulltrack_metrics_per_class.csv", test_per_class_rows, nms_metric_fields())
    write_csv(output_dir / "fulltrack_test_metrics_using_val_global_threshold.csv", test_global_rows, nms_metric_fields())
    write_csv(output_dir / "fulltrack_test_metrics_using_val_per_class_thresholds.csv", test_per_class_rows, nms_metric_fields())
    save_json(output_dir / "fulltrack_selected_thresholds_from_val.json", selected_thresholds_payload)
    save_json(output_dir / "fulltrack_checkpoint_selection_from_val.json", checkpoint_selection_payload)
    save_json(output_dir / "fulltrack_metric_consistency_check.json", consistency_payload)
    global_inference_config = make_inference_config(
        run_dir,
        dataset_root,
        checkpoints,
        "global",
        selected_global["checkpoint"],
        selected_global["threshold"],
        None,
        selected_global.get("onset_nms_ms", fixed_nms_ms),
        tolerance_ms,
        {
            "selected_config": summarize_selected(val_global_rows),
            "by_class_selected_config": val_global_rows,
        },
        {
            "selected_config": summarize_selected(test_global_rows),
            "by_class_selected_config": test_global_rows,
        },
        args.selection_metric,
        args.beta,
        dataset_root_info,
    )
    per_class_fallback_global_threshold = min(selected_per_class_thresholds.values())
    per_class_inference_config = make_inference_config(
        run_dir,
        dataset_root,
        checkpoints,
        "per-class",
        selected_per_class_config["checkpoint"],
        per_class_fallback_global_threshold,
        selected_per_class_thresholds,
        selected_per_class_config["onset_nms_ms"],
        tolerance_ms,
        {
            "selected_config": summarize_selected(val_per_class_rows),
            "by_class_selected_config": val_per_class_rows,
        },
        {
            "selected_config": summarize_selected(test_per_class_rows),
            "by_class_selected_config": test_per_class_rows,
            "delta_vs_global": class_delta_rows(test_global_rows, test_per_class_rows),
        },
        args.selection_metric,
        args.beta,
        dataset_root_info,
    )
    selected_thresholds_per_class_payload = {
        "dataset_root": str(dataset_root),
        **dataset_root_info,
        "run_name": run_dir.name,
        "class_schema": CLASS_SCHEMA,
        "class_names": CLASS_NAMES,
        "midi_labels": LABELS,
        "labels": LABELS,
        "checkpoint_seleccionado": selected_per_class_config["checkpoint"],
        "selected_checkpoint": selected_per_class_config["checkpoint"],
        "checkpoint_path": str(checkpoints[selected_per_class_config["checkpoint"]].resolve()),
        "thresholds_per_class": selected_per_class_thresholds,
        "selected_per_class_thresholds": selected_per_class_thresholds,
        "nms_ms": float(selected_per_class_config["onset_nms_ms"]),
        "onset_nms_ms": float(selected_per_class_config["onset_nms_ms"]),
        "onset_tolerance_ms": float(tolerance_ms),
        "threshold_grid": thresholds,
        "nms_grid_ms": nms_grid_ms if nms_grid_ms else [float(fixed_nms_ms)],
        "selection_split": "VAL",
        "selection_metric": args.selection_metric,
        "beta": args.beta,
        "selection_rule": SELECTION_RULE,
        "val_metrics_for_selected_per_class_thresholds": val_per_class_rows,
        "test_metrics_using_val_per_class_thresholds": test_per_class_rows,
        "test_not_used_for_selection": True,
    }
    save_json(output_dir / "selected_thresholds_per_class_from_val.json", selected_thresholds_per_class_payload)
    save_json(output_dir / "fulltrack_selected_inference_config_global.json", global_inference_config)
    save_json(output_dir / "fulltrack_selected_inference_config_per_class.json", per_class_inference_config)
    selected_inference_config = (
        per_class_inference_config
        if selected_alias_mode == "per-class"
        else global_inference_config
    )
    save_json(output_dir / "fulltrack_selected_inference_config.json", selected_inference_config)
    write_combined_summary(
        output_dir / "fulltrack_calibration_summary.md",
        run_dir,
        dataset_root,
        args,
        selected_global,
        selected_per_class_config,
        val_global_rows,
        val_per_class_rows,
        test_global_rows,
        test_per_class_rows,
        selected_alias_mode,
        previous_artifacts_status,
        dataset_root_info,
    )

    generated_files = [
        "fulltrack_val_onset_metrics_by_threshold.csv",
        "fulltrack_test_onset_metrics_by_threshold.csv",
        "val_fulltrack_metrics_global.csv",
        "test_fulltrack_metrics_global.csv",
        "val_fulltrack_metrics_per_class.csv",
        "test_fulltrack_metrics_per_class.csv",
        "fulltrack_selected_thresholds_from_val.json",
        "selected_thresholds_per_class_from_val.json",
        "fulltrack_checkpoint_selection_from_val.json",
        "fulltrack_test_metrics_using_val_global_threshold.csv",
        "fulltrack_test_metrics_using_val_per_class_thresholds.csv",
        "fulltrack_calibration_summary.md",
        "fulltrack_metric_consistency_check.json",
        "fulltrack_selected_inference_config_global.json",
        "fulltrack_selected_inference_config_per_class.json",
        "fulltrack_selected_inference_config.json",
    ]

    if nms_grid_ms:
        selected_nms_config = selected_global
        nms_val_selected_rows = val_global_rows
        nms_test_selected_rows = test_global_rows
        nms_test_without_nms_rows = filter_global_nms(
            nms_test_rows,
            selected_nms_config["checkpoint"],
            selected_nms_config["threshold"],
            0.0,
        )
        if not nms_test_without_nms_rows:
            nms_test_without_nms_rows = filter_global(
                test_rows,
                selected_nms_config["checkpoint"],
                selected_nms_config["threshold"],
            )

        nms_payload = {
            "dataset_root": str(dataset_root),
            **dataset_root_info,
            "run_name": run_dir.name,
            "class_schema": CLASS_SCHEMA,
            "class_names": CLASS_NAMES,
            "labels": LABELS,
            "onset_tolerance_ms": tolerance_ms,
            "threshold_grid": thresholds,
            "nms_grid_ms": nms_grid_ms,
            "selection_split": "VAL",
            "selection_scope": "full-track global threshold plus per-class NMS post-processing",
            "selection_metric": args.selection_metric,
            "beta": args.beta,
            "selection_rule": SELECTION_RULE,
            "selected_checkpoint": selected_nms_config["checkpoint"],
            "selected_threshold": float(selected_nms_config["threshold"]),
            "selected_onset_nms_ms": float(selected_nms_config["onset_nms_ms"]),
            "val_metrics_for_selected_config": nms_val_selected_rows,
            "test_metrics_using_val_config": nms_test_selected_rows,
            "test_metrics_without_nms_same_checkpoint_threshold": nms_test_without_nms_rows,
            "test_delta_with_selected_nms": diff_summary(nms_test_without_nms_rows, nms_test_selected_rows),
            "test_not_used_for_selection": True,
        }

        write_csv(output_dir / "fulltrack_nms_val_metrics.csv", nms_val_rows, nms_metric_fields())
        write_csv(output_dir / "fulltrack_nms_test_metrics.csv", nms_test_rows, nms_metric_fields())
        write_csv(
            output_dir / "fulltrack_nms_test_metrics_using_val_config.csv",
            nms_test_selected_rows,
            nms_metric_fields(),
        )
        save_json(output_dir / "fulltrack_nms_selected_config_from_val.json", nms_payload)
        write_nms_summary(
            output_dir / "fulltrack_nms_summary.md",
            run_dir,
            dataset_root,
            args,
            selected_nms_config,
            nms_val_selected_rows,
            nms_test_selected_rows,
            nms_test_without_nms_rows,
            previous_artifacts_status,
            dataset_root_info,
        )
        generated_files.extend(
            [
                "fulltrack_nms_val_metrics.csv",
                "fulltrack_nms_test_metrics.csv",
                "fulltrack_nms_selected_config_from_val.json",
                "fulltrack_nms_test_metrics_using_val_config.csv",
                "fulltrack_nms_summary.md",
                "fulltrack_selected_inference_config.json",
            ]
        )
    else:
        fixed_config = {
            "checkpoint": selected_checkpoint,
            "threshold": float(selected_global["threshold"]),
            "onset_nms_ms": float(fixed_nms_ms),
        }
        test_without_nms_rows = (
            filter_global(test_rows, selected_checkpoint, selected_global["threshold"])
            if fixed_nms_ms != 0.0
            else test_global_rows
        )
        write_nms_summary(
            output_dir / "fulltrack_nms_summary.md",
            run_dir,
            dataset_root,
            args,
            fixed_config,
            val_global_rows,
            test_global_rows,
            test_without_nms_rows,
            previous_artifacts_status,
            dataset_root_info,
        )
        generated_files.append("fulltrack_nms_summary.md")

    print(f"\nfulltrack_calibration_dir: {output_dir}")
    print("\nBest global config on VAL:")
    print(f"  checkpoint: {selected_global['checkpoint']}")
    print(f"  threshold: {selected_global['threshold']}")
    print(f"  onset_nms_ms: {selected_global.get('onset_nms_ms', fixed_nms_ms)}")
    print(f"  VAL: {summarize_selected(val_global_rows)}")
    print(f"  TEST frozen from VAL: {summarize_selected(test_global_rows)}")
    print("\nBest per-class config on VAL:")
    print(f"  checkpoint: {selected_per_class_config['checkpoint']}")
    print(f"  thresholds_per_class: {selected_per_class_thresholds}")
    print(f"  onset_nms_ms: {selected_per_class_config['onset_nms_ms']}")
    print(f"  VAL: {summarize_selected(val_per_class_rows)}")
    print(f"  TEST frozen from VAL: {summarize_selected(test_per_class_rows)}")
    print(f"\nselected inference config alias: {selected_alias_mode}")
    print("generated_files:")
    for filename in generated_files:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
