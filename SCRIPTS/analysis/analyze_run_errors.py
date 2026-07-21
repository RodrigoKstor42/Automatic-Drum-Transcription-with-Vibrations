"""
Per-track onset error analysis for calibrated custom_split runs.

This script does not train, modify datasets, or select thresholds. It reloads
the checkpoint and VAL-selected thresholds already stored in a run directory,
then reconstructs TEST/VAL event errors track by track.
"""
import argparse
import collections
import collections.abc
import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

import numpy as np

collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool

from adtof import config
from adtof.model.dataLoader import DataLoader
from adtof.model.model import Model
from SCRIPTS.common.dataset_roots import resolve_dataset_root
from SCRIPTS.training.train_custom_split import (
    CHECKPOINT_METADATA_FILENAME,
    CLASS_NAMES,
    CLASS_SCHEMA,
    LABELS,
    MIDI_TO_CLASS,
    detect_onset_frames,
    validate_checkpoint_schema,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze event-level onset errors for a calibrated custom_split run."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory to analyze.")
    parser.add_argument(
        "--inference-config",
        default=None,
        help="Optional fulltrack_selected_inference_config.json. Autodetected if omitted.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root override. Useful for old runs that point to PHRASE_GENERATOR.",
    )
    parser.add_argument("--split", choices=["VAL", "TEST"], default="TEST")
    parser.add_argument("--threshold-mode", choices=["global", "per-class", "both"], default="both")
    parser.add_argument("--onset-tolerance-ms", type=float, default=None)
    parser.add_argument("--onset-nms-ms", type=float, default=None)
    parser.add_argument("--min-peak-distance-ms", type=float, default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json_optional(path):
    path = Path(path)
    if not path.is_file():
        return None
    return load_json(path)


def save_json(path, data):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_output_dir(path, overwrite):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already exists and is not empty. Use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_vibro_5_toms(run_config, threshold_artifact=None, artifact_name="threshold artifact"):
    expected = {
        "class_schema": CLASS_SCHEMA,
        "class_names": CLASS_NAMES,
        "labels": LABELS,
    }
    actual = {
        "class_schema": run_config.get("class_schema"),
        "class_names": run_config.get("class_names"),
        "labels": run_config.get("labels"),
    }
    if actual != expected:
        raise ValueError(f"Run schema mismatch: expected {expected}, got {actual}")
    if threshold_artifact is None:
        return
    artifact_labels = threshold_artifact.get("midi_labels") or threshold_artifact.get("labels")
    if threshold_artifact.get("class_schema") != CLASS_SCHEMA:
        raise ValueError(
            f"{artifact_name} schema mismatch: expected {CLASS_SCHEMA}, "
            f"got {threshold_artifact.get('class_schema')}"
        )
    artifact_class_names = threshold_artifact.get("class_names")
    if artifact_class_names is not None and artifact_class_names != CLASS_NAMES:
        raise ValueError(
            f"{artifact_name} class_names mismatch: expected {CLASS_NAMES}, "
            f"got {artifact_class_names}"
        )
    if artifact_labels is not None and artifact_labels != LABELS:
        raise ValueError(
            f"{artifact_name} MIDI labels mismatch: expected {LABELS}, got {artifact_labels}"
        )


def autodetect_inference_config(run_dir, cli_inference_config=None):
    if cli_inference_config:
        path = Path(cli_inference_config).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Inference config not found: {path}")
        return load_json(path), path

    candidates = [
        run_dir / "latest_fulltrack_selected_inference_config.json",
        run_dir / "fulltrack_calibration" / "fulltrack_selected_inference_config.json",
        run_dir / "fulltrack_calibration_f1_nms" / "fulltrack_selected_inference_config.json",
    ]
    for path in candidates:
        if path.is_file():
            return load_json(path), path.resolve()
    return None, None


def resolve_checkpoint_from_artifacts(run_dir, inference_config, selected_thresholds, checkpoint_selection):
    if inference_config:
        selected_checkpoint = inference_config["selected_checkpoint"]
        checkpoint_path = inference_config.get("checkpoint_path")
        if checkpoint_path:
            path = Path(checkpoint_path).resolve()
            if not path.is_file():
                fallback = Path(run_dir) / "checkpoints" / Path(checkpoint_path).name
                if fallback.is_file():
                    path = fallback.resolve()
                else:
                    path = selected_checkpoint_path(run_dir, selected_checkpoint)
            validate_checkpoint_schema(path)
            return selected_checkpoint, path
        return selected_checkpoint, selected_checkpoint_path(run_dir, selected_checkpoint)

    selected_checkpoint = selected_thresholds.get("selected_checkpoint") or checkpoint_selection.get("selected_checkpoint")
    return selected_checkpoint, selected_checkpoint_path(run_dir, selected_checkpoint)


def thresholds_from_artifacts(inference_config, selected_thresholds, threshold_modes):
    if inference_config:
        global_threshold = float(inference_config["threshold_global"])
        per_class_thresholds = inference_config.get("thresholds_per_class")
        if per_class_thresholds:
            per_class_thresholds = {
                class_name: float(per_class_thresholds[class_name])
                for class_name in CLASS_NAMES
            }
        elif "per-class" in threshold_modes:
            warnings.warn(
                "Inference config has no per-class thresholds; using global threshold for per-class mode.",
                RuntimeWarning,
            )
            per_class_thresholds = {class_name: global_threshold for class_name in CLASS_NAMES}
        else:
            per_class_thresholds = {class_name: global_threshold for class_name in CLASS_NAMES}
        return {
            "global": global_threshold,
            "per_class": per_class_thresholds,
        }

    return {
        "global": float(selected_thresholds["selected_global_threshold"]),
        "per_class": {
            key: float(value)
            for key, value in selected_thresholds["selected_per_class_thresholds"].items()
        },
    }


def empty_class_counts():
    return {class_name: 0 for class_name in CLASS_NAMES}


def load_ground_truth_events_from_txt(dataset_root, split, class_schema=CLASS_SCHEMA):
    if class_schema != CLASS_SCHEMA:
        raise ValueError(f"Unsupported class schema: {class_schema}")

    annotation_dir = Path(dataset_root) / split.upper() / "ANNOTATIONS"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    events = []
    for annotation_path in sorted(annotation_dir.glob("*.txt")):
        track_id = annotation_path.stem
        with annotation_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                fields = line.strip().split()
                if len(fields) < 2:
                    continue
                try:
                    time_sec = float(fields[0])
                    midi_pitch = int(float(fields[1]))
                except ValueError:
                    print(f"WARNING: invalid annotation row in {annotation_path}:{line_number}")
                    continue
                class_name = MIDI_TO_CLASS.get(midi_pitch)
                if class_name is None:
                    continue
                events.append(
                    {
                        "track_id": track_id,
                        "annotation_path": str(annotation_path),
                        "class_name": class_name,
                        "midi_pitch": midi_pitch,
                        "time_sec": time_sec,
                    }
                )
    return events


def group_ground_truth_events(events, sample_rate):
    grouped = {
        track_id: {class_name: [] for class_name in CLASS_NAMES}
        for track_id in sorted({event["track_id"] for event in events})
    }
    for event in events:
        item = dict(event)
        item["frame"] = int(round(float(event["time_sec"]) * sample_rate))
        grouped[item["track_id"]][item["class_name"]].append(item)

    for classes in grouped.values():
        for class_events in classes.values():
            class_events.sort(key=lambda event: (event["frame"], event["time_sec"]))
    return grouped


def count_ground_truth_from_annotations(dataset_root, split):
    counts = empty_class_counts()
    for event in load_ground_truth_events_from_txt(dataset_root, split):
        counts[event["class_name"]] += 1
    return counts


def count_ground_truth_from_dataset_event_counts_csv(dataset_root, split):
    path = Path(dataset_root) / "dataset_event_counts.csv"
    if not path.is_file():
        return None

    counts = empty_class_counts()
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("split", "").upper() != split.upper():
                continue
            class_name = row.get("class") or row.get("class_name")
            if class_name not in counts:
                continue
            event_count = row.get("event_count") or row.get("count") or 0
            counts[class_name] += int(float(event_count))
    return counts


def subtract_counts(left, right):
    if left is None or right is None:
        return None
    return {
        class_name: int(left.get(class_name, 0)) - int(right.get(class_name, 0))
        for class_name in CLASS_NAMES
    }


def counts_match(left, right):
    return left is not None and right is not None and all(
        int(left.get(class_name, 0)) == int(right.get(class_name, 0))
        for class_name in CLASS_NAMES
    )


def count_ground_truth_used_by_analyzer_by_mode(class_rows, threshold_modes):
    counts_by_mode = {mode: empty_class_counts() for mode in threshold_modes}
    for row in class_rows:
        mode = row["threshold_mode"]
        if mode not in counts_by_mode:
            continue
        class_name = row["class_name"]
        counts_by_mode[mode][class_name] += int(row["TP"]) + int(row["FN"])
    return counts_by_mode


def make_ground_truth_count_check(
    split,
    dataset_root,
    counts_from_annotations,
    counts_from_dataset_event_counts_csv,
    counts_used_by_analyzer_by_mode,
    max_tracks,
):
    annotation_vs_csv_match = (
        counts_from_dataset_event_counts_csv is None
        or counts_match(counts_from_annotations, counts_from_dataset_event_counts_csv)
    )
    analyzer_mode_matches = {
        mode: counts_match(counts, counts_from_annotations)
        for mode, counts in counts_used_by_analyzer_by_mode.items()
    }
    analyzer_vs_annotations_match = all(analyzer_mode_matches.values())
    if analyzer_vs_annotations_match and annotation_vs_csv_match:
        status = "match"
    elif max_tracks is not None and not analyzer_vs_annotations_match:
        status = "partial_due_to_max_tracks"
    else:
        status = "mismatch"

    return {
        "split": split.upper(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "counts_from_annotations": counts_from_annotations,
        "counts_from_dataset_event_counts_csv": counts_from_dataset_event_counts_csv,
        "counts_used_by_analyzer_by_mode": counts_used_by_analyzer_by_mode,
        "delta_annotations_vs_csv": subtract_counts(
            counts_from_annotations,
            counts_from_dataset_event_counts_csv,
        ),
        "delta_analyzer_vs_annotations_by_mode": {
            mode: subtract_counts(counts, counts_from_annotations)
            for mode, counts in counts_used_by_analyzer_by_mode.items()
        },
        "analyzer_mode_matches_annotations": analyzer_mode_matches,
        "status": status,
        "max_tracks": max_tracks,
    }


def selected_checkpoint_path(run_dir, selected_checkpoint):
    checkpoint_dir = run_dir / "checkpoints"
    filename_by_label = {
        "best": "best_weights.weights.h5",
        "final": "final_weights.weights.h5",
    }
    filename = filename_by_label.get(selected_checkpoint)
    if filename is None:
        raise ValueError(f"Unsupported selected checkpoint: {selected_checkpoint}")
    path = checkpoint_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {path}")
    validate_checkpoint_schema(path)
    return path


def build_model_and_data(run_config, dataset_root):
    common_kwargs = {
        "sampleRate": int(run_config["sample_rate"]),
        "trainingSequence": int(run_config["training_sequence"]),
        "batchSize": int(run_config["batch_size"]),
        "context": int(run_config["context"]),
        "labels": config.VIBRO_LABELS_5,
        "sampleWeight": config.VIBRO_WEIGHTS_5,
        "prefetch": None,
        "n_channels": 1,
        "samePadding": bool(run_config.get("same_padding", False)),
    }
    model, hparams = Model.modelFactory(
        modelName=run_config.get("model_name", "Frame_RNN"),
        scenario="custom_split",
        fold=0,
        **common_kwargs,
    )
    if model.weightLoadedFlag:
        raise RuntimeError("Model factory loaded default weights unexpectedly; refusing analysis.")

    data_access = DataLoader.factoryMixedDatasets(
        folderPath=str(dataset_root),
        scenario="custom_split",
        cachePreprocessFolders=str(dataset_root / "PREPROCESS_MINI_TRAIN"),
        **hparams,
    )
    return model, hparams, data_access


def get_split_tracks(data_access, hparams, split, max_tracks=None):
    loader = data_access.trainDataLoaders[0]
    index_attr = {"VAL": "valIndexes", "TEST": "testIndexes"}[split]
    indexes = list(getattr(loader, index_attr))
    if max_tracks is not None:
        indexes = indexes[:max_tracks]
    generator = loader.getGen(indexes, training=False, **hparams)()
    return loader, indexes, generator


def get_track_id(track):
    return Path(getattr(track, "path", getattr(track, "title", ""))).stem


def get_annotation_path(loader, track):
    basename = get_track_id(track)
    for path in getattr(loader, "annotationPaths", []) or []:
        if Path(path).stem == basename:
            return Path(path)
    return None


def infer_profile(track_id, annotation_path):
    candidates = []
    if annotation_path is not None:
        candidates.extend(
            [
                annotation_path.with_suffix(".json"),
                annotation_path.parent.parent / "METADATA" / f"{annotation_path.stem}.json",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            try:
                metadata = load_json(candidate)
            except Exception:
                continue
            for key in ("profile", "profile_name", "source_profile", "kit_profile", "pattern_type"):
                if metadata.get(key):
                    return str(metadata[key])

    match = re.search(r"^(?:train|val|test)_([^_]+(?:_[^_]+)*)_\d{6}_", track_id)
    if match:
        return match.group(1)
    return "unknown"


def detailed_match(predicted_frames, target_frames, tolerance_frames):
    matched_targets = set()
    tp_pairs = []
    fp_frames = []
    for predicted_frame in predicted_frames:
        candidates = [
            (abs(int(target_frame) - int(predicted_frame)), target_index, int(target_frame))
            for target_index, target_frame in enumerate(target_frames)
            if target_index not in matched_targets
            and abs(int(target_frame) - int(predicted_frame)) <= tolerance_frames
        ]
        if candidates:
            _, target_index, target_frame = min(candidates)
            matched_targets.add(target_index)
            tp_pairs.append((int(predicted_frame), target_frame))
        else:
            fp_frames.append(int(predicted_frame))
    fn_frames = [
        int(target_frame)
        for target_index, target_frame in enumerate(target_frames)
        if target_index not in matched_targets
    ]
    return tp_pairs, fp_frames, fn_frames


def detailed_match_events(predicted_frames, target_events, tolerance_frames):
    matched_targets = set()
    tp_pairs = []
    fp_frames = []
    for predicted_frame in predicted_frames:
        candidates = [
            (abs(int(target_event["frame"]) - int(predicted_frame)), target_index, target_event)
            for target_index, target_event in enumerate(target_events)
            if target_index not in matched_targets
            and abs(int(target_event["frame"]) - int(predicted_frame)) <= tolerance_frames
        ]
        if candidates:
            _, target_index, target_event = min(candidates)
            matched_targets.add(target_index)
            tp_pairs.append((int(predicted_frame), target_event))
        else:
            fp_frames.append(int(predicted_frame))
    fn_events = [
        target_event
        for target_index, target_event in enumerate(target_events)
        if target_index not in matched_targets
    ]
    return tp_pairs, fp_frames, fn_events


def resolve_nms_ms(onset_nms_ms=None, min_peak_distance_ms=None):
    if onset_nms_ms is None and min_peak_distance_ms is None:
        return 0.0
    if onset_nms_ms is None:
        onset_nms_ms = min_peak_distance_ms
    if min_peak_distance_ms is None:
        min_peak_distance_ms = onset_nms_ms
    onset_nms_ms = float(onset_nms_ms)
    min_peak_distance_ms = float(min_peak_distance_ms)
    if onset_nms_ms < 0 or min_peak_distance_ms < 0:
        raise ValueError("NMS/min-peak-distance values must be non-negative")
    if abs(onset_nms_ms - min_peak_distance_ms) > 1e-12:
        raise ValueError("--onset-nms-ms and --min-peak-distance-ms must match when both are provided")
    return onset_nms_ms


def apply_onset_nms(predicted_frames, confidence_values, nms_frames):
    predicted_frames = [int(frame) for frame in predicted_frames]
    if nms_frames <= 0 or len(predicted_frames) <= 1:
        return sorted(predicted_frames)

    ranked_frames = sorted(
        predicted_frames,
        key=lambda frame: (-float(confidence_values[frame]), frame),
    )
    kept = []
    for frame in ranked_frames:
        if all(abs(frame - kept_frame) > nms_frames for kept_frame in kept):
            kept.append(frame)
    return sorted(kept)


def detect_onset_frames_with_nms(values, threshold, sample_rate, onset_nms_ms=0.0):
    predicted_frames = detect_onset_frames(values, threshold)
    nms_frames = int(round(float(onset_nms_ms) * sample_rate / 1000.0))
    return apply_onset_nms(predicted_frames, values, nms_frames)


def safe_metric(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def percentile_or_blank(values, percentile):
    if not values:
        return ""
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def analyze_track(
    track,
    loader,
    model,
    hparams,
    ground_truth_events_by_track,
    split,
    threshold_mode,
    thresholds_by_class,
    selected_checkpoint,
    tolerance_frames,
    tolerance_ms,
    sample_rate,
    onset_nms_ms=0.0,
):
    predictions = np.asarray(model.predict(track, **hparams))
    frame_count = predictions.shape[0]
    predictions = predictions[:, : len(CLASS_NAMES)]

    track_id = get_track_id(track)
    audio_path = Path(getattr(track, "path", ""))
    annotation_path = get_annotation_path(loader, track)
    track_gt_events = ground_truth_events_by_track.get(
        track_id,
        {class_name: [] for class_name in CLASS_NAMES},
    )
    if annotation_path is None:
        for class_events in track_gt_events.values():
            if class_events:
                annotation_path = Path(class_events[0]["annotation_path"])
                break
    profile = infer_profile(track_id, annotation_path)
    duration_sec = float(getattr(track, "samplesCardinality", frame_count)) / sample_rate
    n_total_events_track = int(sum(len(track_gt_events[class_name]) for class_name in CLASS_NAMES))
    event_density = n_total_events_track / duration_sec if duration_sec else 0.0

    event_rows = []
    track_rows = []
    confusion_fp = []
    confusion_fn = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        midi_pitch = LABELS[class_index]
        threshold = float(thresholds_by_class[class_name])
        target_events = track_gt_events[class_name]
        predicted_frames = detect_onset_frames_with_nms(
            predictions[:, class_index],
            threshold,
            sample_rate,
            onset_nms_ms,
        )
        tp_pairs, fp_frames, fn_events = detailed_match_events(
            predicted_frames, target_events, tolerance_frames
        )
        abs_errors = []

        base = {
            "split": split,
            "track_id": track_id,
            "audio_path": str(audio_path),
            "annotation_path": str(annotation_path) if annotation_path else "",
            "profile": profile,
            "class_name": class_name,
            "midi_pitch": midi_pitch,
            "threshold": threshold,
            "threshold_mode": threshold_mode,
            "checkpoint": selected_checkpoint,
            "onset_tolerance_ms": float(tolerance_ms),
            "onset_nms_ms": float(onset_nms_ms),
        }

        for predicted_frame, target_event in tp_pairs:
            target_frame = int(target_event["frame"])
            target_time_sec = float(target_event["time_sec"])
            predicted_time_sec = predicted_frame / sample_rate
            time_error_ms = (predicted_time_sec - target_time_sec) * 1000.0
            abs_error = abs(time_error_ms)
            abs_errors.append(abs_error)
            event_rows.append(
                {
                    **base,
                    "event_type": "TP",
                    "target_time_sec": target_time_sec,
                    "pred_time_sec": predicted_time_sec,
                    "time_error_ms": time_error_ms,
                    "abs_time_error_ms": abs_error,
                    "confidence": float(predictions[predicted_frame, class_index]),
                }
            )
        for predicted_frame in fp_frames:
            event_rows.append(
                {
                    **base,
                    "event_type": "FP",
                    "target_time_sec": "",
                    "pred_time_sec": predicted_frame / sample_rate,
                    "time_error_ms": "",
                    "abs_time_error_ms": "",
                    "confidence": float(predictions[predicted_frame, class_index]),
                }
            )
            confusion_fp.append(
                {
                    "class_name": class_name,
                    "frame": predicted_frame,
                    "time_sec": predicted_frame / sample_rate,
                    "confidence": float(predictions[predicted_frame, class_index]),
                    "threshold": threshold,
                }
            )
        for target_event in fn_events:
            target_frame = int(target_event["frame"])
            event_rows.append(
                {
                    **base,
                    "event_type": "FN",
                    "target_time_sec": float(target_event["time_sec"]),
                    "pred_time_sec": "",
                    "time_error_ms": "",
                    "abs_time_error_ms": "",
                    "confidence": "",
                }
            )
            confusion_fn.append(
                {
                    "class_name": class_name,
                    "frame": target_frame,
                    "time_sec": target_frame / sample_rate,
                    "threshold": threshold,
                }
            )

        tp, fp, fn = len(tp_pairs), len(fp_frames), len(fn_events)
        precision, recall, f1 = safe_metric(tp, fp, fn)
        track_rows.append(
            {
                "split": split,
                "track_id": track_id,
                "profile": profile,
                "duration_sec": duration_sec,
                "n_total_events_track": n_total_events_track,
                "event_density_per_sec": event_density,
                "class_name": class_name,
                "threshold": threshold,
                "threshold_mode": threshold_mode,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_abs_time_error_ms": float(np.mean(abs_errors)) if abs_errors else "",
                "median_abs_time_error_ms": float(np.median(abs_errors)) if abs_errors else "",
                "p90_abs_time_error_ms": percentile_or_blank(abs_errors, 90),
            }
        )

    confusion_rows = []
    for fp_event in confusion_fp:
        for fn_event in confusion_fn:
            if fp_event["class_name"] == fn_event["class_name"]:
                continue
            delta_frames = fp_event["frame"] - fn_event["frame"]
            if abs(delta_frames) <= tolerance_frames:
                confusion_rows.append(
                    {
                        "split": split,
                        "track_id": track_id,
                        "profile": profile,
                        "fp_class_name": fp_event["class_name"],
                        "fn_class_name": fn_event["class_name"],
                        "fp_time_sec": fp_event["time_sec"],
                        "fn_time_sec": fn_event["time_sec"],
                        "delta_ms": delta_frames * 1000.0 / sample_rate,
                        "abs_delta_ms": abs(delta_frames) * 1000.0 / sample_rate,
                        "fp_confidence": fp_event["confidence"],
                        "fp_threshold": fp_event["threshold"],
                        "fn_threshold": fn_event["threshold"],
                        "threshold_mode": threshold_mode,
                        "checkpoint": selected_checkpoint,
                        "onset_tolerance_ms": float(tolerance_ms),
                    }
                )
    return event_rows, track_rows, confusion_rows


def aggregate_rows(track_rows, keys):
    grouped = collections.defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "errors": [], "tracks": set(), "error_tracks": set()})
    for row in track_rows:
        key = tuple(row[k] for k in keys)
        item = grouped[key]
        item["TP"] += int(row["TP"])
        item["FP"] += int(row["FP"])
        item["FN"] += int(row["FN"])
        item["tracks"].add(row["track_id"])
        if int(row["FP"]) or int(row["FN"]):
            item["error_tracks"].add(row["track_id"])
        for error_key in ("mean_abs_time_error_ms", "median_abs_time_error_ms", "p90_abs_time_error_ms"):
            if row[error_key] != "":
                item["errors"].append(float(row[error_key]))

    rows = []
    for key, item in grouped.items():
        tp, fp, fn = item["TP"], item["FP"], item["FN"]
        precision, recall, f1 = safe_metric(tp, fp, fn)
        output = {field: value for field, value in zip(keys, key)}
        output.update(
            {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_abs_time_error_ms": float(np.mean(item["errors"])) if item["errors"] else "",
                "median_abs_time_error_ms": float(np.median(item["errors"])) if item["errors"] else "",
                "p90_abs_time_error_ms": percentile_or_blank(item["errors"], 90),
                "n_tracks_with_class": len(item["tracks"]),
                "n_tracks_with_errors": len(item["error_tracks"]),
            }
        )
        rows.append(output)
    return rows


def aggregate_worst_tracks(track_rows):
    grouped = collections.defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "duration_sec": 0.0, "density": 0.0, "profile": ""})
    for row in track_rows:
        key = (row["threshold_mode"], row["track_id"])
        item = grouped[key]
        item["TP"] += int(row["TP"])
        item["FP"] += int(row["FP"])
        item["FN"] += int(row["FN"])
        item["duration_sec"] = float(row["duration_sec"])
        item["density"] = float(row["event_density_per_sec"])
        item["profile"] = row["profile"]

    rows = []
    for (threshold_mode, track_id), item in grouped.items():
        precision, recall, f1 = safe_metric(item["TP"], item["FP"], item["FN"])
        rows.append(
            {
                "threshold_mode": threshold_mode,
                "track_id": track_id,
                "profile": item["profile"],
                "duration_sec": item["duration_sec"],
                "event_density_per_sec": item["density"],
                "TP": item["TP"],
                "FP": item["FP"],
                "FN": item["FN"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def compute_global_metrics(class_rows):
    by_mode = {}
    for mode in sorted({row["threshold_mode"] for row in class_rows}):
        rows = [row for row in class_rows if row["threshold_mode"] == mode]
        tp = sum(int(row["TP"]) for row in rows)
        fp = sum(int(row["FP"]) for row in rows)
        fn = sum(int(row["FN"]) for row in rows)
        precision, recall, f1 = safe_metric(tp, fp, fn)
        by_mode[mode] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "macro_f1": float(np.mean([float(row["f1"]) for row in rows])) if rows else 0.0,
        }
    return by_mode


def make_track_coverage_check(
    split,
    ground_truth_events_by_track,
    track_rows,
    track_iteration_counts,
    track_iteration_counts_by_mode,
    threshold_modes,
    max_tracks,
):
    annotation_track_ids = set(ground_truth_events_by_track)
    used_track_ids = set(track_iteration_counts)

    used_counts_by_track_mode = collections.defaultdict(
        lambda: {mode: empty_class_counts() for mode in threshold_modes}
    )
    for row in track_rows:
        track_id = row["track_id"]
        mode = row["threshold_mode"]
        class_name = row["class_name"]
        used_counts_by_track_mode[track_id][mode][class_name] += int(row["TP"]) + int(row["FN"])

    coverage_rows = []
    tracks_with_gt_mismatch = []
    for track_id in sorted(annotation_track_ids | used_track_ids):
        annotation_counts = empty_class_counts()
        for class_name in CLASS_NAMES:
            annotation_counts[class_name] = len(
                ground_truth_events_by_track.get(track_id, {}).get(class_name, [])
            )

        row = {
            "split": split.upper(),
            "track_id": track_id,
            "annotation_path": "",
            "times_seen_by_analyzer": int(track_iteration_counts.get(track_id, 0)),
        }
        for class_events in ground_truth_events_by_track.get(track_id, {}).values():
            if class_events:
                row["annotation_path"] = class_events[0]["annotation_path"]
                break

        for class_name in CLASS_NAMES:
            row[f"n_gt_annotation_{class_name}"] = annotation_counts[class_name]
        for mode in ("global", "per-class"):
            mode_key = "per_class" if mode == "per-class" else mode
            mode_counts = used_counts_by_track_mode[track_id].get(mode, empty_class_counts())
            for class_name in CLASS_NAMES:
                row[f"n_gt_used_{mode_key}_{class_name}"] = mode_counts[class_name]

        mode_mismatches = []
        for mode in threshold_modes:
            mode_counts = used_counts_by_track_mode[track_id][mode]
            if any(mode_counts[class_name] != annotation_counts[class_name] for class_name in CLASS_NAMES):
                mode_mismatches.append(mode)

        if track_id not in annotation_track_ids:
            status = "extra"
        elif track_id not in used_track_ids:
            status = "missing_due_to_max_tracks" if max_tracks is not None else "missing"
        elif row["times_seen_by_analyzer"] > 1:
            status = "duplicated"
        elif mode_mismatches:
            status = "gt_mismatch"
            tracks_with_gt_mismatch.append(track_id)
        else:
            status = "ok"
        row["status"] = status
        coverage_rows.append(row)

    duplicated_track_ids = sorted(
        track_id for track_id, count in track_iteration_counts.items() if count > 1
    )
    missing_track_ids = sorted(annotation_track_ids - used_track_ids)
    extra_track_ids = sorted(used_track_ids - annotation_track_ids)
    total_iterations_by_mode = {
        mode: sum(track_iteration_counts_by_mode.get(mode, {}).values())
        for mode in threshold_modes
    }

    if max_tracks is not None and missing_track_ids:
        status = "partial_due_to_max_tracks"
    elif duplicated_track_ids or missing_track_ids or extra_track_ids or tracks_with_gt_mismatch:
        status = "mismatch"
    else:
        status = "match"

    payload = {
        "split": split.upper(),
        "n_annotation_tracks": len(annotation_track_ids),
        "n_unique_tracks_used_by_analyzer": len(used_track_ids),
        "n_total_track_iterations_by_analyzer": sum(track_iteration_counts.values()),
        "n_total_track_iterations_by_analyzer_by_mode": total_iterations_by_mode,
        "duplicated_track_ids": duplicated_track_ids,
        "missing_track_ids": missing_track_ids,
        "extra_track_ids": extra_track_ids,
        "tracks_with_gt_mismatch": sorted(set(tracks_with_gt_mismatch)),
        "status": status,
        "max_tracks": max_tracks,
    }
    return coverage_rows, payload


def read_official_metrics(run_dir, split, mode):
    if split == "TEST":
        filename = {
            "global": "final_test_onset_metrics_using_val_global_threshold.csv",
            "per-class": "final_test_onset_metrics_using_val_per_class_thresholds.csv",
        }[mode]
        path = run_dir / filename
    else:
        path = run_dir / f"{split.lower()}_onset_metrics_by_threshold.csv"
    if not path.is_file():
        return None, []
    with path.open("r", newline="", encoding="utf-8") as file:
        return path, list(csv.DictReader(file))


def compare_official(run_dir, split, class_rows, selected_checkpoint, max_tracks):
    checks = {}
    for mode in sorted({row["threshold_mode"] for row in class_rows}):
        rows = [row for row in class_rows if row["threshold_mode"] == mode]
        path, official_rows = read_official_metrics(run_dir, split, mode)
        if not official_rows:
            checks[mode] = {"status": "official_file_missing", "official_path": str(path) if path else ""}
            continue

        reconstructed = compute_global_metrics(rows)[mode]
        official_for_classes = []
        for class_row in rows:
            candidates = [
                official
                for official in official_rows
                if official.get("class") == class_row["class_name"]
                and official.get("checkpoint", selected_checkpoint) == selected_checkpoint
                and abs(float(official.get("threshold", -1)) - float(class_row["threshold"])) < 1e-12
            ]
            if candidates:
                official_for_classes.append(candidates[0])

        if not official_for_classes:
            checks[mode] = {
                "status": "no_matching_official_rows",
                "official_path": str(path),
                "note": "Rows may be checkpoint/threshold filtered differently.",
            }
            continue

        official_tp = sum(int(float(row["TP"])) for row in official_for_classes)
        official_fp = sum(int(float(row["FP"])) for row in official_for_classes)
        official_fn = sum(int(float(row["FN"])) for row in official_for_classes)
        op, or_, of1 = safe_metric(official_tp, official_fp, official_fn)
        official_macro = float(np.mean([float(row["f1"]) for row in official_for_classes]))
        official = {
            "TP": official_tp,
            "FP": official_fp,
            "FN": official_fn,
            "micro_precision": op,
            "micro_recall": or_,
            "micro_f1": of1,
            "macro_f1": official_macro,
        }
        diffs = {
            key: abs(float(reconstructed[key]) - float(official[key]))
            for key in ("micro_precision", "micro_recall", "micro_f1", "macro_f1")
        }
        status = "match" if max(diffs.values()) <= 1e-6 and max_tracks is None else "different"
        note = ""
        if max_tracks is not None:
            note = "--max-tracks was used, so mismatch with official full-run metrics is expected."
        elif status == "different":
            note = (
                "This script analyzes full tracks; the original run metrics may have been "
                "computed from eval_steps batches."
            )
        checks[mode] = {
            "status": status,
            "official_path": str(path),
            "reconstructed": reconstructed,
            "official": official,
            "absolute_differences": diffs,
            "note": note,
        }
    return checks


def markdown_table(rows, columns, limit=None):
    rows = rows[:limit] if limit else rows
    if not rows:
        return "_Sin filas._"
    output = [
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
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def write_summary(
    path,
    run_dir,
    split,
    selected_checkpoint,
    threshold_modes,
    thresholds,
    global_metrics,
    class_rows,
    worst_f1,
    worst_fp,
    worst_fn,
    profile_rows,
    max_tracks,
    consistency,
    ground_truth_check,
    track_coverage_check,
    onset_nms_ms,
    threshold_source,
    inference_config_path=None,
):
    lines = [
        f"# Error analysis: {run_dir.name}",
        "",
        f"- split: `{split}`",
        f"- checkpoint: `{selected_checkpoint}`",
        f"- threshold source: `{threshold_source}`",
        f"- inference_config: `{inference_config_path}`",
        f"- threshold modes: `{', '.join(threshold_modes)}`",
        f"- global threshold: `{thresholds['global']}`",
        f"- per-class thresholds: `{thresholds['per_class']}`",
        f"- nms_ms: `{onset_nms_ms}`",
    ]
    if max_tracks is not None:
        lines.append(f"- WARNING: analysis limited with `--max-tracks {max_tracks}`.")
    lines.extend(["", "## Reconstructed Metrics"])
    for mode, metrics in global_metrics.items():
        lines.append(
            f"- `{mode}`: micro-P={metrics['micro_precision']:.4f}, "
            f"micro-R={metrics['micro_recall']:.4f}, micro-F1={metrics['micro_f1']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}, TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}"
        )
    lines.extend(
        [
            "",
            "## Ground-truth count check",
            f"- status: `{ground_truth_check['status']}`",
            f"- counts_from_annotations: `{ground_truth_check['counts_from_annotations']}`",
            f"- counts_used_by_analyzer_by_mode: `{ground_truth_check['counts_used_by_analyzer_by_mode']}`",
        ]
    )
    if ground_truth_check["counts_from_dataset_event_counts_csv"] is None:
        lines.append("- counts_from_dataset_event_counts_csv: `not found`")
    else:
        lines.append(
            f"- counts_from_dataset_event_counts_csv: `{ground_truth_check['counts_from_dataset_event_counts_csv']}`"
        )
    lines.append(
        f"- delta_analyzer_vs_annotations_by_mode: `{ground_truth_check['delta_analyzer_vs_annotations_by_mode']}`"
    )
    if ground_truth_check["status"] == "partial_due_to_max_tracks":
        lines.append(
            "- note: `--max-tracks was used, so analyzer counts intentionally cover only the analyzed subset.`"
        )
    elif ground_truth_check["status"] != "match":
        lines.append(
            "- WARNING: analyzer ground-truth counts differ from annotation counts"
        )
    lines.extend(
        [
            "",
            "## Track coverage check",
            f"- status: `{track_coverage_check['status']}`",
            f"- n_annotation_tracks: `{track_coverage_check['n_annotation_tracks']}`",
            f"- n_unique_tracks_used_by_analyzer: `{track_coverage_check['n_unique_tracks_used_by_analyzer']}`",
            f"- n_total_track_iterations_by_analyzer: `{track_coverage_check['n_total_track_iterations_by_analyzer']}`",
            f"- n_total_track_iterations_by_analyzer_by_mode: `{track_coverage_check['n_total_track_iterations_by_analyzer_by_mode']}`",
            f"- duplicated_track_ids: `{track_coverage_check['duplicated_track_ids']}`",
            f"- missing_track_ids_count: `{len(track_coverage_check['missing_track_ids'])}`",
            f"- extra_track_ids: `{track_coverage_check['extra_track_ids']}`",
            f"- tracks_with_gt_mismatch: `{track_coverage_check['tracks_with_gt_mismatch']}`",
        ]
    )
    lines.extend(
        [
            "",
            "## By Class",
            markdown_table(
                sorted(class_rows, key=lambda row: (row["threshold_mode"], row["class_name"])),
                ["threshold_mode", "class_name", "threshold", "TP", "FP", "FN", "precision", "recall", "f1"],
            ),
            "",
            "## Worst Tracks By F1",
            markdown_table(worst_f1, ["threshold_mode", "track_id", "profile", "TP", "FP", "FN", "precision", "recall", "f1"], 10),
            "",
            "## Worst Tracks By False Positives",
            markdown_table(worst_fp, ["threshold_mode", "track_id", "profile", "TP", "FP", "FN", "precision", "recall", "f1"], 10),
            "",
            "## Worst Tracks By False Negatives",
            markdown_table(worst_fn, ["threshold_mode", "track_id", "profile", "TP", "FP", "FN", "precision", "recall", "f1"], 10),
            "",
            "## Problematic Profiles",
            markdown_table(
                sorted(profile_rows, key=lambda row: (row["threshold_mode"], row["f1"], -row["FP"] - row["FN"])),
                ["threshold_mode", "profile", "class_name", "TP", "FP", "FN", "precision", "recall", "f1"],
                15,
            ),
            "",
            "## Consistency Check",
        ]
    )
    for mode, check in consistency.items():
        lines.append(f"- `{mode}`: {check.get('status')} {check.get('note', '')}".rstrip())
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = ensure_output_dir(args.output_dir, args.overwrite)
    run_config = load_json(run_dir / "config.json")
    inference_config, inference_config_path = autodetect_inference_config(
        run_dir,
        args.inference_config,
    )
    selected_thresholds_path = run_dir / "selected_thresholds_from_val.json"
    selected_thresholds = load_json_optional(selected_thresholds_path) or {}
    checkpoint_selection_path = run_dir / "checkpoint_selection_from_val.json"
    checkpoint_selection = load_json_optional(checkpoint_selection_path) or {}
    if inference_config is None and not selected_thresholds:
        raise FileNotFoundError(
            "No full-track inference config or legacy selected_thresholds_from_val.json found.\n"
            "Tried:\n"
            f"- {run_dir / 'latest_fulltrack_selected_inference_config.json'}\n"
            f"- {run_dir / 'fulltrack_calibration' / 'fulltrack_selected_inference_config.json'}\n"
            f"- {run_dir / 'fulltrack_calibration_f1_nms' / 'fulltrack_selected_inference_config.json'}\n"
            f"- {selected_thresholds_path}\n"
            "Suggestion: pass --inference-config with a fulltrack_selected_inference_config.json."
        )

    threshold_artifact = inference_config or selected_thresholds
    threshold_source = "fulltrack_inference_config" if inference_config else "legacy_selected_thresholds"
    require_vibro_5_toms(run_config, threshold_artifact, threshold_source)
    config_dataset_root = (
        inference_config.get("dataset_root_resolved")
        if inference_config and inference_config.get("dataset_root_resolved")
        else run_config["dataset_root"]
    )
    dataset_root_info = resolve_dataset_root(
        config_dataset_root,
        cli_dataset_root=args.dataset_root,
        repo_root=REPO_ROOT,
    )
    if inference_config and not args.dataset_root:
        for key in (
            "dataset_root_original_from_run_config",
            "dataset_root_source",
            "dataset_root_migrated_candidate",
        ):
            if inference_config.get(key):
                dataset_root_info[key] = inference_config[key]
    dataset_root = Path(dataset_root_info["dataset_root_resolved"]).resolve()
    ground_truth_events = load_ground_truth_events_from_txt(dataset_root, args.split)
    counts_from_annotations = empty_class_counts()
    for event in ground_truth_events:
        counts_from_annotations[event["class_name"]] += 1
    counts_from_dataset_event_counts_csv = count_ground_truth_from_dataset_event_counts_csv(
        dataset_root,
        args.split,
    )
    selected_checkpoint, weights_path = resolve_checkpoint_from_artifacts(
        run_dir,
        inference_config,
        selected_thresholds,
        checkpoint_selection,
    )
    artifact_tolerance_ms = (
        inference_config.get("onset_tolerance_ms")
        if inference_config
        else selected_thresholds["onset_tolerance_ms"]
    )
    tolerance_ms = float(args.onset_tolerance_ms if args.onset_tolerance_ms is not None else artifact_tolerance_ms)
    sample_rate = int(run_config["sample_rate"])
    tolerance_frames = int(round(tolerance_ms * sample_rate / 1000.0))
    artifact_nms_ms = inference_config.get("onset_nms_ms") if inference_config else None
    onset_nms_ms = resolve_nms_ms(
        args.onset_nms_ms if args.onset_nms_ms is not None else artifact_nms_ms,
        args.min_peak_distance_ms,
    )
    ground_truth_events_by_track = group_ground_truth_events(ground_truth_events, sample_rate)

    threshold_modes = ["global", "per-class"] if args.threshold_mode == "both" else [args.threshold_mode]
    thresholds = thresholds_from_artifacts(inference_config, selected_thresholds, threshold_modes)

    model, hparams, data_access = build_model_and_data(run_config, dataset_root)
    model.model.load_weights(str(weights_path))
    loader, indexes, tracks = get_split_tracks(data_access, hparams, args.split, args.max_tracks)

    event_rows = []
    track_rows = []
    confusion_rows = []
    track_iteration_counts = collections.Counter()
    track_iteration_counts_by_mode = {mode: collections.Counter() for mode in threshold_modes}
    for track_number, track in enumerate(tracks, start=1):
        if args.max_tracks is not None and track_number > args.max_tracks:
            break
        track_id = get_track_id(track)
        track_iteration_counts[track_id] += 1
        for mode in threshold_modes:
            track_iteration_counts_by_mode[mode][track_id] += 1
            if mode == "global":
                thresholds_by_class = {class_name: thresholds["global"] for class_name in CLASS_NAMES}
            else:
                thresholds_by_class = thresholds["per_class"]
            events, tracks_by_class, confusions = analyze_track(
                track,
                loader,
                model,
                hparams,
                ground_truth_events_by_track,
                args.split,
                mode,
                thresholds_by_class,
                selected_checkpoint,
                tolerance_frames,
                tolerance_ms,
                sample_rate,
                onset_nms_ms,
            )
            event_rows.extend(events)
            track_rows.extend(tracks_by_class)
            confusion_rows.extend(confusions)
        print(f"analyzed {track_number}/{len(indexes)} tracks: {track_id}")

    class_rows = aggregate_rows(
        track_rows,
        ["split", "threshold_mode", "class_name", "threshold"],
    )
    profile_rows = aggregate_rows(
        track_rows,
        ["split", "threshold_mode", "profile", "class_name", "threshold"],
    )
    worst_rows = aggregate_worst_tracks(track_rows)
    worst_f1 = sorted(worst_rows, key=lambda row: (row["f1"], -row["FP"] - row["FN"], row["track_id"]))
    worst_fp = sorted(worst_rows, key=lambda row: (-row["FP"], row["f1"], row["track_id"]))
    worst_fn = sorted(worst_rows, key=lambda row: (-row["FN"], row["f1"], row["track_id"]))
    global_metrics = compute_global_metrics(class_rows)
    consistency = compare_official(run_dir, args.split, class_rows, selected_checkpoint, args.max_tracks)
    counts_used_by_analyzer_by_mode = count_ground_truth_used_by_analyzer_by_mode(
        class_rows,
        threshold_modes,
    )
    ground_truth_check = make_ground_truth_count_check(
        args.split,
        dataset_root,
        counts_from_annotations,
        counts_from_dataset_event_counts_csv,
        counts_used_by_analyzer_by_mode,
        args.max_tracks,
    )
    track_coverage_rows, track_coverage_check = make_track_coverage_check(
        args.split,
        ground_truth_events_by_track,
        track_rows,
        track_iteration_counts,
        track_iteration_counts_by_mode,
        threshold_modes,
        args.max_tracks,
    )
    if not all(
        counts_match(counts, counts_from_annotations)
        for counts in counts_used_by_analyzer_by_mode.values()
    ):
        print("WARNING: analyzer ground-truth counts differ from annotation counts")

    event_fields = [
        "split",
        "track_id",
        "audio_path",
        "annotation_path",
        "profile",
        "class_name",
        "midi_pitch",
        "event_type",
        "target_time_sec",
        "pred_time_sec",
        "time_error_ms",
        "abs_time_error_ms",
        "confidence",
        "threshold",
        "threshold_mode",
        "checkpoint",
        "onset_tolerance_ms",
        "onset_nms_ms",
    ]
    track_fields = [
        "split",
        "track_id",
        "profile",
        "duration_sec",
        "n_total_events_track",
        "event_density_per_sec",
        "class_name",
        "threshold",
        "threshold_mode",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "mean_abs_time_error_ms",
        "median_abs_time_error_ms",
        "p90_abs_time_error_ms",
    ]
    class_fields = [
        "split",
        "threshold_mode",
        "class_name",
        "threshold",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "mean_abs_time_error_ms",
        "median_abs_time_error_ms",
        "p90_abs_time_error_ms",
        "n_tracks_with_class",
        "n_tracks_with_errors",
    ]
    profile_fields = ["split", "threshold_mode", "profile"] + class_fields[2:]
    worst_fields = [
        "threshold_mode",
        "track_id",
        "profile",
        "duration_sec",
        "event_density_per_sec",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
    ]
    confusion_fields = [
        "split",
        "track_id",
        "profile",
        "fp_class_name",
        "fn_class_name",
        "fp_time_sec",
        "fn_time_sec",
        "delta_ms",
        "abs_delta_ms",
        "fp_confidence",
        "fp_threshold",
        "fn_threshold",
        "threshold_mode",
        "checkpoint",
        "onset_tolerance_ms",
    ]
    track_coverage_fields = [
        "split",
        "track_id",
        "annotation_path",
        "n_gt_annotation_KD",
        "n_gt_annotation_SD",
        "n_gt_annotation_T12",
        "n_gt_annotation_T14",
        "n_gt_annotation_T16",
        "n_gt_used_global_KD",
        "n_gt_used_global_SD",
        "n_gt_used_global_T12",
        "n_gt_used_global_T14",
        "n_gt_used_global_T16",
        "n_gt_used_per_class_KD",
        "n_gt_used_per_class_SD",
        "n_gt_used_per_class_T12",
        "n_gt_used_per_class_T14",
        "n_gt_used_per_class_T16",
        "times_seen_by_analyzer",
        "status",
    ]

    write_csv(output_dir / "event_level_errors.csv", event_rows, event_fields)
    write_csv(output_dir / "error_analysis_by_track.csv", track_rows, track_fields)
    write_csv(output_dir / "error_analysis_by_class.csv", class_rows, class_fields)
    write_csv(output_dir / "error_analysis_by_profile.csv", profile_rows, profile_fields)
    write_csv(output_dir / "worst_tracks_by_f1.csv", worst_f1, worst_fields)
    write_csv(output_dir / "worst_tracks_by_false_positives.csv", worst_fp, worst_fields)
    write_csv(output_dir / "worst_tracks_by_false_negatives.csv", worst_fn, worst_fields)
    write_csv(output_dir / "confusion_candidates.csv", confusion_rows, confusion_fields)
    write_csv(output_dir / "track_coverage_check.csv", track_coverage_rows, track_coverage_fields)

    analysis_config = {
        "run_dir": run_dir,
        "dataset_root": dataset_root,
        **dataset_root_info,
        "split": args.split,
        "inference_config_path": inference_config_path,
        "threshold_source": threshold_source,
        "selected_thresholds_from_val_path": selected_thresholds_path if selected_thresholds else None,
        "checkpoint_selection_from_val_path": checkpoint_selection_path if checkpoint_selection else None,
        "test_not_used_for_selection": (
            bool(inference_config.get("test_not_used_for_selection"))
            if inference_config
            else True
        ),
        "threshold_mode": args.threshold_mode,
        "threshold_modes_evaluated": threshold_modes,
        "max_tracks": args.max_tracks,
        "class_schema": CLASS_SCHEMA,
        "class_names": CLASS_NAMES,
        "labels": LABELS,
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_path": weights_path,
        "checkpoint_metadata_path": weights_path.parent / CHECKPOINT_METADATA_FILENAME,
        "selected_global_threshold": thresholds["global"],
        "selected_per_class_thresholds": thresholds["per_class"],
        "onset_tolerance_ms": tolerance_ms,
        "onset_tolerance_frames": tolerance_frames,
        "onset_nms_ms": onset_nms_ms,
        "nms_ms": onset_nms_ms,
        "sample_rate": sample_rate,
        "n_tracks_available": len(indexes),
        "n_tracks_analyzed": min(len(indexes), args.max_tracks) if args.max_tracks else len(indexes),
    }
    consistency_payload = {
        "note": (
            "The official run metrics were generated by the training script. "
            "This analyzer reconstructs per-track full-track errors using the same onset "
            "detection and one-to-one matching logic."
        ),
        "max_tracks": args.max_tracks,
        "checks": consistency,
    }
    save_json(output_dir / "error_analysis_config.json", analysis_config)
    save_json(output_dir / "metric_consistency_check.json", consistency_payload)
    save_json(output_dir / "ground_truth_count_check.json", ground_truth_check)
    save_json(output_dir / "track_coverage_check.json", track_coverage_check)
    write_summary(
        output_dir / "error_summary.md",
        run_dir,
        args.split,
        selected_checkpoint,
        threshold_modes,
        thresholds,
        global_metrics,
        class_rows,
        worst_f1,
        worst_fp,
        worst_fn,
        profile_rows,
        args.max_tracks,
        consistency,
        ground_truth_check,
        track_coverage_check,
        onset_nms_ms,
        threshold_source,
        inference_config_path,
    )

    print(f"error_analysis_dir: {output_dir}")
    print("generated_files:")
    for filename in [
        "event_level_errors.csv",
        "error_analysis_by_track.csv",
        "error_analysis_by_class.csv",
        "error_analysis_by_profile.csv",
        "worst_tracks_by_f1.csv",
        "worst_tracks_by_false_positives.csv",
        "worst_tracks_by_false_negatives.csv",
        "confusion_candidates.csv",
        "track_coverage_check.csv",
        "metric_consistency_check.json",
        "ground_truth_count_check.json",
        "track_coverage_check.json",
        "error_summary.md",
        "error_analysis_config.json",
    ]:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
