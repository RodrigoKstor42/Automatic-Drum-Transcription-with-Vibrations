"""
Minimal custom_split training smoke test for ADTOF.

This script reuses Model.modelFactory(), which builds and compiles the Keras
model in adtof/model/model.py. It keeps training intentionally tiny: the goal is
only to validate DataLoader -> model -> loss -> backpropagation.
"""
import argparse
import csv
import collections
import collections.abc
import datetime
import json
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must be set before TensorFlow/Keras are imported by adtof.model.model.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "True")

import numpy as np

# Compatibility shims used by madmom in newer Python/NumPy environments.
collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool

import tensorflow as tf

from adtof import config
from adtof.model.dataLoader import DataLoader
from adtof.model.model import Model

CLASS_SCHEMA = "vibro_5_toms"
CLASS_NAMES = list(config.VIBRO_LABELS_5TXT)
LABELS = list(config.VIBRO_LABELS_5)
MIDI_TO_CLASS = dict(zip(LABELS, CLASS_NAMES))
LEGACY_CLASS_NAMES = ["KD", "SD", "TT", "HH", "CY"]
CHECKPOINT_METADATA_FILENAME = "checkpoint_metadata.json"
DEFAULT_THRESHOLDS = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a minimal ADTOF training pass on CUSTOM_DEBUG custom_split."
    )
    parser.add_argument("--dataset-root", default="CUSTOM_DEBUG", help="Root with TRAIN/VAL/TEST folders.")
    parser.add_argument("--model-name", default="Frame_RNN", help="Model key from adtof.model.hyperparameters.models.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--training-sequence", type=int, default=64)
    parser.add_argument("--context", type=int, default=9)
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=2)
    parser.add_argument("--validation-steps", type=int, default=1)
    parser.add_argument("--same-padding", action="store_true", help="Use same padding from the chosen model hparams.")
    parser.add_argument("--no-val", action="store_true", help="Skip validation if needed.")
    parser.add_argument("--output-dir", default="RUNS", help="Root directory for experiment outputs.")
    parser.add_argument("--run-name", default=None, help="Experiment name. Defaults to a timestamp.")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save final and best validation weights.")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate frame-wise metrics on the TEST split.")
    parser.add_argument("--eval-steps", type=int, default=20, help="Number of TEST batches to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used to binarize predictions.")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Thresholds used for multi-threshold TEST evaluation.",
    )
    parser.add_argument("--eval-onsets", action="store_true", help="Evaluate onset events on TEST batches.")
    parser.add_argument("--onset-tolerance-ms", type=float, default=50.0)
    parser.add_argument(
        "--onset-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Thresholds for onset evaluation. Defaults to --thresholds.",
    )
    return parser.parse_args()


def make_run_dir(output_dir, run_name):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / base_name

    if run_name is None:
        suffix = 1
        while run_dir.exists():
            run_dir = output_dir / f"{base_name}_{suffix:02d}"
            suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_value(data), file, indent=2)


def save_history(run_dir, history):
    history_data = history.history
    save_json(run_dir / "history.json", history_data)

    metric_names = list(history_data.keys())
    epoch_count = max((len(values) for values in history_data.values()), default=0)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch"] + metric_names)
        writer.writeheader()
        for epoch_index in range(epoch_count):
            row = {"epoch": epoch_index + 1}
            for metric_name, values in history_data.items():
                row[metric_name] = values[epoch_index] if epoch_index < len(values) else ""
            writer.writerow(row)


class SafeModelCheckpoint(tf.keras.callbacks.Callback):
    """Proxy a Keras ModelCheckpoint without allowing save errors to stop fit."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.disabled = False

    def set_model(self, model):
        super().set_model(model)
        self.callback.set_model(model)

    def set_params(self, params):
        super().set_params(params)
        self.callback.set_params(params)

    def on_train_begin(self, logs=None):
        self._call("on_train_begin", logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._call("on_epoch_begin", epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._call("on_epoch_end", epoch, logs)

    def on_train_end(self, logs=None):
        self._call("on_train_end", logs)

    def _call(self, method_name, *args):
        if self.disabled:
            return
        try:
            getattr(self.callback, method_name)(*args)
        except Exception as error:
            self.disabled = True
            print(f"WARNING: best checkpoint disabled after a compatibility error: {error}")


def build_callbacks(model, checkpoint_dir, use_validation):
    if not use_validation:
        return []

    best_path = checkpoint_dir / "best_weights.weights.h5"
    try:
        callback = model.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        return [SafeModelCheckpoint(callback)]
    except Exception as error:
        print(f"WARNING: could not create best checkpoint callback: {error}")
        return []


def save_final_weights(keras_model, checkpoint_dir):
    final_path = checkpoint_dir / "final_weights.weights.h5"
    try:
        keras_model.save_weights(str(final_path))
        print(f"final_weights: {final_path}")
        return final_path
    except Exception as error:
        print(f"WARNING: could not save final weights to {final_path}: {error}")
        return None


def checkpoint_metadata():
    return {
        "class_schema": CLASS_SCHEMA,
        "class_names": CLASS_NAMES,
        "labels": LABELS,
    }


def save_checkpoint_metadata(checkpoint_dir):
    metadata_path = checkpoint_dir / CHECKPOINT_METADATA_FILENAME
    save_json(metadata_path, checkpoint_metadata())
    return metadata_path


def validate_checkpoint_schema(weights_path):
    metadata_path = weights_path.parent / CHECKPOINT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(
            f"Refusing to load checkpoint without {CHECKPOINT_METADATA_FILENAME}: "
            f"{weights_path}. It may use the legacy classes {LEGACY_CLASS_NAMES}."
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    expected = checkpoint_metadata()
    actual = {
        "class_schema": metadata.get("class_schema"),
        "class_names": metadata.get("class_names"),
        "labels": metadata.get("labels"),
    }
    if actual != expected:
        raise ValueError(
            f"Checkpoint class schema mismatch for {weights_path}: "
            f"expected {expected}, got {actual}."
        )


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_dataset_events(dataset_root):
    rows = []
    for split in ("TRAIN", "VAL", "TEST"):
        counts = {class_name: 0 for class_name in CLASS_NAMES}
        annotation_dir = dataset_root / split / "ANNOTATIONS"
        for annotation_path in sorted(annotation_dir.glob("*.txt")):
            with annotation_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    fields = line.strip().split()
                    if not fields:
                        continue
                    if len(fields) < 2:
                        print(f"WARNING: malformed annotation {annotation_path}:{line_number}")
                        continue
                    try:
                        midi_pitch = int(float(fields[1]))
                    except ValueError:
                        print(f"WARNING: invalid MIDI pitch {annotation_path}:{line_number}")
                        continue
                    class_name = MIDI_TO_CLASS.get(midi_pitch)
                    if class_name is not None:
                        counts[class_name] += 1
        rows.extend(
            {
                "split": split,
                "class": class_name,
                "midi_pitch": midi_pitch,
                "event_count": counts[class_name],
            }
            for midi_pitch, class_name in zip(LABELS, CLASS_NAMES)
        )
    return rows


def save_dataset_event_counts(run_dir, rows):
    write_csv(
        run_dir / "dataset_event_counts.csv",
        rows,
        ["split", "class", "midi_pitch", "event_count"],
    )
    save_json(run_dir / "dataset_event_counts.json", {"counts": rows})
    print("\nAnnotated dataset events")
    print("split class midi  events")
    for row in rows:
        print(
            f"{row['split']:<5} {row['class']:<5} {row['midi_pitch']:4d}"
            f" {row['event_count']:7d}"
        )


def validate_thresholds(thresholds, argument_name):
    if not thresholds:
        raise ValueError(f"{argument_name} requires at least one value")
    invalid = [threshold for threshold in thresholds if not 0.0 <= threshold <= 1.0]
    if invalid:
        raise ValueError(f"{argument_name} values must be between 0 and 1: {invalid}")
    return list(dict.fromkeys(thresholds))


def capture_test_batches(data_access, hparams, eval_steps):
    if eval_steps <= 0:
        raise ValueError("--eval-steps must be greater than zero")

    test_loader = data_access.trainDataLoaders[0]
    if len(test_loader.testIndexes) == 0:
        raise ValueError("custom_split TEST contains no paired tracks")

    test_gen = test_loader.getGen(test_loader.testIndexes, **hparams)
    test_dataset = DataLoader._getDataset(test_gen, **hparams)
    test_iterator = iter(test_dataset)
    batches = []
    for _ in range(eval_steps):
        batch = next(test_iterator)
        inputs, targets = batch[0], batch[1]
        batches.append((inputs, np.asarray(targets)))
    return batches


def predict_test_batches(keras_model, batches):
    predicted_batches = []
    for inputs, targets in batches:
        predictions = keras_model.predict_on_batch(inputs)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"prediction/target shape mismatch: {predictions.shape} vs {targets.shape}"
            )
        if predictions.shape[-1] != len(CLASS_NAMES):
            raise ValueError(
                f"expected {len(CLASS_NAMES)} output classes, got {predictions.shape[-1]}"
            )
        predicted_batches.append((predictions, targets))
    return predicted_batches


def calculate_frame_metrics(predicted_batches, thresholds):
    counts_by_threshold = {
        threshold: {
            "tp": np.zeros(len(CLASS_NAMES), dtype=np.int64),
            "fp": np.zeros(len(CLASS_NAMES), dtype=np.int64),
            "fn": np.zeros(len(CLASS_NAMES), dtype=np.int64),
            "predicted_positive_frames": np.zeros(len(CLASS_NAMES), dtype=np.int64),
        }
        for threshold in thresholds
    }
    target_positive_frames = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    prediction_sum = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    max_prediction = np.full(len(CLASS_NAMES), -np.inf, dtype=np.float64)
    prediction_frame_count = 0

    for predictions, targets in predicted_batches:
        target_binary = targets > 0
        axes = tuple(range(target_binary.ndim - 1))
        target_positive_frames += np.sum(target_binary, axis=axes, dtype=np.int64)
        prediction_sum += np.sum(predictions, axis=axes, dtype=np.float64)
        max_prediction = np.maximum(max_prediction, np.max(predictions, axis=axes))
        prediction_frame_count += int(np.prod(predictions.shape[:-1]))

        for threshold, counts in counts_by_threshold.items():
            predicted_binary = predictions >= threshold
            counts["tp"] += np.sum(predicted_binary & target_binary, axis=axes, dtype=np.int64)
            counts["fp"] += np.sum(predicted_binary & ~target_binary, axis=axes, dtype=np.int64)
            counts["fn"] += np.sum(~predicted_binary & target_binary, axis=axes, dtype=np.int64)
            counts["predicted_positive_frames"] += np.sum(
                predicted_binary, axis=axes, dtype=np.int64
            )

    metrics = []
    prediction_stats = []
    mean_prediction = prediction_sum / prediction_frame_count
    for threshold, counts in counts_by_threshold.items():
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_tp = counts["tp"][class_index]
            class_fp = counts["fp"][class_index]
            class_fn = counts["fn"][class_index]
            precision = class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0
            recall = class_tp / (class_tp + class_fn) if class_tp + class_fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics.append(
                {
                    "threshold": float(threshold),
                    "class": class_name,
                    "TP": int(class_tp),
                    "FP": int(class_fp),
                    "FN": int(class_fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )
            prediction_stats.append(
                {
                    "threshold": float(threshold),
                    "class": class_name,
                    "target_positive_frames": int(target_positive_frames[class_index]),
                    "predicted_positive_frames": int(
                        counts["predicted_positive_frames"][class_index]
                    ),
                    "max_prediction": float(max_prediction[class_index]),
                    "mean_prediction": float(mean_prediction[class_index]),
                }
            )
    return metrics, prediction_stats


def detect_onset_frames(values, threshold):
    values = np.asarray(values)
    above = values >= threshold
    crossings = above & np.concatenate(([True], ~above[:-1]))
    previous = np.concatenate(([-np.inf], values[:-1]))
    following = np.concatenate((values[1:], [-np.inf]))
    local_maxima = above & (values >= previous) & (values > following)
    return np.flatnonzero(crossings | local_maxima)


def match_onsets(predicted_frames, target_frames, tolerance_frames):
    matched_targets = set()
    true_positives = 0
    for predicted_frame in predicted_frames:
        candidates = [
            (abs(int(target_frame) - int(predicted_frame)), target_index)
            for target_index, target_frame in enumerate(target_frames)
            if target_index not in matched_targets
            and abs(int(target_frame) - int(predicted_frame)) <= tolerance_frames
        ]
        if candidates:
            _, target_index = min(candidates)
            matched_targets.add(target_index)
            true_positives += 1
    return true_positives, len(predicted_frames) - true_positives, len(target_frames) - true_positives


def calculate_onset_metrics(predicted_batches, thresholds, sample_rate, tolerance_ms):
    if sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than zero for onset evaluation")
    if tolerance_ms < 0:
        raise ValueError("--onset-tolerance-ms cannot be negative")
    tolerance_frames = int(round(tolerance_ms * sample_rate / 1000.0))
    counts = {
        threshold: {
            "tp": np.zeros(len(CLASS_NAMES), dtype=np.int64),
            "fp": np.zeros(len(CLASS_NAMES), dtype=np.int64),
            "fn": np.zeros(len(CLASS_NAMES), dtype=np.int64),
        }
        for threshold in thresholds
    }

    for predictions, targets in predicted_batches:
        for batch_index in range(predictions.shape[0]):
            for class_index in range(len(CLASS_NAMES)):
                target_frames = detect_onset_frames(targets[batch_index, :, class_index], 1e-12)
                for threshold in thresholds:
                    predicted_frames = detect_onset_frames(
                        predictions[batch_index, :, class_index], threshold
                    )
                    tp, fp, fn = match_onsets(
                        predicted_frames, target_frames, tolerance_frames
                    )
                    counts[threshold]["tp"][class_index] += tp
                    counts[threshold]["fp"][class_index] += fp
                    counts[threshold]["fn"][class_index] += fn

    rows = []
    for threshold, threshold_counts in counts.items():
        for class_index, class_name in enumerate(CLASS_NAMES):
            tp = int(threshold_counts["tp"][class_index])
            fp = int(threshold_counts["fp"][class_index])
            fn = int(threshold_counts["fn"][class_index])
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "threshold": float(threshold),
                    "class": class_name,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "tolerance_ms": float(tolerance_ms),
                }
            )
    return rows


def save_test_metrics(run_dir, metrics, eval_steps, threshold, print_summary=True):
    threshold_metrics = [row for row in metrics if row["threshold"] == threshold]
    save_json(
        run_dir / "test_metrics.json",
        {"eval_steps": eval_steps, "threshold": threshold, "classes": threshold_metrics},
    )
    with (run_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["threshold", "class", "TP", "FP", "FN", "precision", "recall", "f1"]
        )
        writer.writeheader()
        writer.writerows(threshold_metrics)

    if print_summary:
        print(f"\nTEST frame-wise metrics (threshold={threshold:g})")
        print("class       TP       FP       FN  precision     recall         f1")
        for row in threshold_metrics:
            print(
                f"{row['class']:<5} {row['TP']:8d} {row['FP']:8d} {row['FN']:8d}"
                f" {row['precision']:10.4f} {row['recall']:10.4f} {row['f1']:10.4f}"
            )


def print_metric_summary(title, metrics):
    print(f"\n{title}")
    print("threshold class       TP       FP       FN  precision     recall         f1")
    for row in metrics:
        print(
            f"{row['threshold']:9g} {row['class']:<5} {row['TP']:8d}"
            f" {row['FP']:8d} {row['FN']:8d} {row['precision']:10.4f}"
            f" {row['recall']:10.4f} {row['f1']:10.4f}"
        )


def print_prediction_stats(title, prediction_stats):
    print(f"\n{title}")
    print("threshold class   target_pos  predicted_pos   max_pred  mean_pred")
    for row in prediction_stats:
        print(
            f"{row['threshold']:9g} {row['class']:<5}"
            f" {row['target_positive_frames']:11d}"
            f" {row['predicted_positive_frames']:14d}"
            f" {row['max_prediction']:10.4f}"
            f" {row['mean_prediction']:10.4f}"
        )


def save_model_evaluation(run_dir, model_label, metrics, prediction_stats, onset_metrics=None):
    metric_fields = ["threshold", "class", "TP", "FP", "FN", "precision", "recall", "f1"]
    metrics_path = run_dir / f"{model_label}_test_metrics_by_threshold.csv"
    stats_path = run_dir / f"{model_label}_prediction_stats.csv"
    write_csv(metrics_path, metrics, metric_fields)

    stats_fields = [
        "threshold",
        "class",
        "target_positive_frames",
        "predicted_positive_frames",
        "max_prediction",
        "mean_prediction",
    ]
    write_csv(stats_path, prediction_stats, stats_fields)
    generated = [metrics_path, stats_path]
    print_metric_summary(f"{model_label.upper()} frame-wise metrics", metrics)
    print_prediction_stats(f"{model_label.upper()} prediction statistics", prediction_stats)

    if onset_metrics is not None:
        onset_path = run_dir / f"{model_label}_onset_metrics_by_threshold.csv"
        write_csv(onset_path, onset_metrics, metric_fields + ["tolerance_ms"])
        generated.append(onset_path)
        print_metric_summary(f"{model_label.upper()} onset metrics", onset_metrics)
    return generated


def best_thresholds_by_class(metrics):
    best = {}
    for class_name in CLASS_NAMES:
        class_rows = [row for row in metrics if row["class"] == class_name]
        if class_rows:
            best[class_name] = max(class_rows, key=lambda row: (row["f1"], -row["threshold"]))
    return best


def print_best_thresholds(title, metrics):
    print(f"\n{title}")
    for class_name, row in best_thresholds_by_class(metrics).items():
        print(f"{class_name}: threshold={row['threshold']:g}, f1={row['f1']:.4f}")


def save_generic_evaluation(run_dir, metrics, prediction_stats, eval_steps, threshold):
    metric_fields = ["threshold", "class", "TP", "FP", "FN", "precision", "recall", "f1"]
    stats_fields = [
        "threshold",
        "class",
        "target_positive_frames",
        "predicted_positive_frames",
        "max_prediction",
        "mean_prediction",
    ]
    write_csv(run_dir / "test_metrics_by_threshold.csv", metrics, metric_fields)
    save_json(
        run_dir / "test_metrics_by_threshold.json",
        {"eval_steps": eval_steps, "thresholds": sorted({row["threshold"] for row in metrics}), "classes": metrics},
    )
    write_csv(run_dir / "prediction_stats.csv", prediction_stats, stats_fields)
    save_test_metrics(run_dir, metrics, eval_steps, threshold, print_summary=False)


def evaluate_loaded_model(
    keras_model,
    batches,
    thresholds,
    onset_thresholds,
    sample_rate,
    onset_tolerance_ms,
    eval_onsets,
):
    predicted_batches = predict_test_batches(keras_model, batches)
    frame_metrics, prediction_stats = calculate_frame_metrics(predicted_batches, thresholds)
    onset_metrics = None
    onset_error = None
    if eval_onsets:
        try:
            onset_metrics = calculate_onset_metrics(
                predicted_batches,
                onset_thresholds,
                sample_rate,
                onset_tolerance_ms,
            )
        except Exception as error:
            onset_error = error
    return frame_metrics, prediction_stats, onset_metrics, onset_error


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    thresholds = validate_thresholds(args.thresholds, "--thresholds")
    onset_thresholds = validate_thresholds(
        args.onset_thresholds or thresholds, "--onset-thresholds"
    )
    run_dir = make_run_dir(args.output_dir, args.run_name)
    generated_files = []
    execution_datetime = datetime.datetime.now().astimezone().isoformat()
    config_data = {
        "class_schema": CLASS_SCHEMA,
        "class_names": CLASS_NAMES,
        "labels": LABELS,
        "dataset_root": str(dataset_root),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "validation_steps": args.validation_steps,
        "batch_size": args.batch_size,
        "model_name": args.model_name,
        "training_sequence": args.training_sequence,
        "context": args.context,
        "sample_rate": args.sample_rate,
        "same_padding": args.same_padding,
        "no_val": args.no_val,
        "threshold": args.threshold,
        "thresholds": thresholds,
        "eval_onsets": args.eval_onsets,
        "onset_tolerance_ms": args.onset_tolerance_ms,
        "onset_thresholds": onset_thresholds,
        "execution_datetime": execution_datetime,
    }
    save_json(run_dir / "config.json", config_data)
    generated_files.append(run_dir / "config.json")

    print(f"dataset_root: {dataset_root}")
    print(f"run_dir: {run_dir}")
    print(f"epochs: {args.epochs}")
    print(f"steps_per_epoch: {args.steps_per_epoch}")
    print(f"validation_steps: {args.validation_steps}")
    print(f"batch_size: {args.batch_size}")
    print(f"save_checkpoints: {args.save_checkpoints}")
    print(f"eval_test: {args.eval_test}")

    try:
        event_count_rows = count_dataset_events(dataset_root)
        save_dataset_event_counts(run_dir, event_count_rows)
        generated_files.extend(
            [run_dir / "dataset_event_counts.csv", run_dir / "dataset_event_counts.json"]
        )
    except Exception as error:
        print(f"WARNING: dataset event counting failed: {error}")

    cache_root = dataset_root / "PREPROCESS_MINI_TRAIN"
    common_kwargs = {
        "sampleRate": args.sample_rate,
        "trainingSequence": args.training_sequence,
        "batchSize": args.batch_size,
        "context": args.context,
        "labels": config.VIBRO_LABELS_5,
        "sampleWeight": config.VIBRO_WEIGHTS_5,
        "prefetch": None,
        "n_channels": 1,
        # The validated custom_split smoke batch is input 72 -> target 64, which
        # corresponds to valid padding with context=9.
        "samePadding": bool(args.same_padding),
    }

    model, hparams = Model.modelFactory(
        modelName=args.model_name,
        scenario="custom_split",
        fold=0,
        **common_kwargs,
    )
    if model.weightLoadedFlag:
        raise RuntimeError(
            "Refusing to continue with automatically loaded pre-existing weights. "
            f"The vibro schema is {CLASS_NAMES}; legacy checkpoints may use "
            f"{LEGACY_CLASS_NAMES}."
        )

    data_access = DataLoader.factoryMixedDatasets(
        folderPath=str(dataset_root),
        scenario="custom_split",
        cachePreprocessFolders=str(cache_root),
        **hparams,
    )

    first_batch = next(iter(data_access.train_dataset))
    inputs, targets, weights = first_batch
    print(f"input shape: {tuple(inputs['x'].shape)}")
    print(f"target shape: {tuple(targets.shape)}")
    print(f"weight shape: {tuple(weights.shape)}")

    validation_data = None if args.no_val else data_access.val_dataset
    validation_steps = None if args.no_val else args.validation_steps
    callbacks = []
    checkpoint_dir = run_dir / "checkpoints"
    if args.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        generated_files.append(save_checkpoint_metadata(checkpoint_dir))
        callbacks = build_callbacks(tf.keras, checkpoint_dir, not args.no_val)

    history = model.model.fit(
        data_access.train_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    save_history(run_dir, history)
    generated_files.extend([run_dir / "history.csv", run_dir / "history.json"])

    loss = history.history.get("loss", [None])[-1]
    val_loss = None if args.no_val else history.history.get("val_loss", [None])[-1]
    print(f"loss: {loss}")
    if val_loss is not None:
        print(f"val_loss: {val_loss}")

    best_checkpoint_path = checkpoint_dir / "best_weights.weights.h5"
    final_checkpoint_path = None
    if args.save_checkpoints:
        final_checkpoint_path = save_final_weights(model.model, checkpoint_dir)

    frame_results = {}
    onset_results = {}
    if args.eval_test:
        try:
            batches = capture_test_batches(data_access, hparams, args.eval_steps)
            evaluation_targets = []
            if args.save_checkpoints:
                if best_checkpoint_path.is_file():
                    evaluation_targets.append(("best", best_checkpoint_path))
                else:
                    print(f"WARNING: best checkpoint not found: {best_checkpoint_path}")
                if final_checkpoint_path and final_checkpoint_path.is_file():
                    evaluation_targets.append(("final", final_checkpoint_path))
                else:
                    print("WARNING: final checkpoint is unavailable; evaluating current model weights")
                    evaluation_targets.append(("final", None))
            else:
                evaluation_targets.append(("final", None))

            for model_label, weights_path in evaluation_targets:
                try:
                    if weights_path is not None:
                        validate_checkpoint_schema(weights_path)
                        model.model.load_weights(str(weights_path))
                    metrics, prediction_stats, onset_metrics, onset_error = evaluate_loaded_model(
                        model.model,
                        batches,
                        thresholds,
                        onset_thresholds,
                        args.sample_rate,
                        args.onset_tolerance_ms,
                        args.eval_onsets,
                    )
                    generated_files.extend(
                        save_model_evaluation(
                            run_dir,
                            model_label,
                            metrics,
                            prediction_stats,
                            onset_metrics,
                        )
                    )
                    frame_results[model_label] = metrics
                    if onset_metrics is not None:
                        onset_results[model_label] = onset_metrics
                    if onset_error is not None:
                        print(
                            f"ERROR: {model_label} onset evaluation failed; "
                            f"frame-wise results remain valid: {onset_error}"
                        )
                    if model_label == "final":
                        save_generic_evaluation(
                            run_dir,
                            metrics,
                            prediction_stats,
                            args.eval_steps,
                            args.threshold,
                        )
                        generated_files.extend(
                            [
                                run_dir / "test_metrics.csv",
                                run_dir / "test_metrics.json",
                                run_dir / "test_metrics_by_threshold.csv",
                                run_dir / "test_metrics_by_threshold.json",
                                run_dir / "prediction_stats.csv",
                            ]
                        )
                except Exception as error:
                    print(f"ERROR: {model_label} TEST evaluation failed: {error}")
        except Exception as error:
            print(f"ERROR: TEST evaluation failed: {error}")

    print("\nExecution summary")
    print(f"run_dir: {run_dir}")
    if best_checkpoint_path.is_file():
        print(f"best_checkpoint: {best_checkpoint_path}")
    else:
        print("best_checkpoint: unavailable")
    if final_checkpoint_path and final_checkpoint_path.is_file():
        print(f"final_checkpoint: {final_checkpoint_path}")
    else:
        print("final_checkpoint: unavailable")
    print("generated_files:")
    for path in dict.fromkeys(generated_files):
        if path.is_file():
            print(f"  {path.name}")

    for model_label, metrics in frame_results.items():
        print_best_thresholds(
            f"Best frame-wise threshold by class ({model_label})", metrics
        )
    if args.eval_onsets:
        for model_label, metrics in onset_results.items():
            print_best_thresholds(
                f"Best onset threshold by class ({model_label})", metrics
            )


if __name__ == "__main__":
    main()
