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

REPO_ROOT = Path(__file__).resolve().parents[2]
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
THRESHOLD_SELECTION_EPS = 1e-9
THRESHOLD_SELECTION_RULE = (
    "maximize F1 -> maximize precision -> minimize FP -> choose highest threshold"
)
CHECKPOINT_SELECTION_RULE = (
    "select checkpoint by VAL onset micro-F1 using the threshold selection rule; "
    "prefer final only if all VAL criteria tie"
)
DEFAULT_CLASS_WEIGHTS = [1.0] * len(CLASS_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a controlled ADTOF training pass on a custom_split dataset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Root with TRAIN/VAL/TEST folders. Recommended: "
            r".\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL"
        ),
    )
    parser.add_argument("--model-name", default="Frame_RNN", help="Model key from adtof.model.hyperparameters.models.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--training-sequence", type=int, default=64)
    parser.add_argument("--context", type=int, default=9)
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=len(CLASS_NAMES),
        default=DEFAULT_CLASS_WEIGHTS,
        metavar=tuple(CLASS_NAMES),
        help=(
            "Per-class training sample weights in the fixed order "
            f"{' '.join(CLASS_NAMES)}. Defaults to all 1.0."
        ),
    )
    parser.add_argument(
        "--loss-type",
        choices=("bce", "focal"),
        default="bce",
        help="Training loss. Use bce for the original binary crossentropy or focal for binary focal loss.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma used only when --loss-type focal.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Positive-class alpha used only when --loss-type focal.",
    )
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
    parser.add_argument(
        "--calibrate-thresholds-on-val",
        action="store_true",
        help=(
            "Evaluate VAL and TEST onsets, select checkpoint and thresholds only from VAL, "
            "and report final TEST metrics with those fixed thresholds."
        ),
    )
    return parser.parse_args()


def validate_class_weights(class_weights):
    weights = [float(weight) for weight in class_weights]
    if len(weights) != len(CLASS_NAMES):
        raise SystemExit(
            f"--class-weights requires {len(CLASS_NAMES)} values in order "
            f"{' '.join(CLASS_NAMES)}; got {len(weights)}."
        )
    if any(weight <= 0 for weight in weights):
        raise SystemExit("--class-weights values must be positive.")
    return weights


def validate_loss_args(loss_type, focal_gamma, focal_alpha):
    if loss_type not in ("bce", "focal"):
        raise SystemExit("--loss-type must be one of: bce, focal")
    if focal_gamma < 0:
        raise SystemExit("--focal-gamma must be greater than or equal to 0.")
    if not 0 <= focal_alpha <= 1:
        raise SystemExit("--focal-alpha must be between 0 and 1.")
    return loss_type, float(focal_gamma), float(focal_alpha)


def binary_focal_loss(gamma=2.0, alpha=0.25):
    gamma = float(gamma)
    alpha = float(alpha)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return alpha_factor * modulating_factor * cross_entropy

    loss.__name__ = f"binary_focal_loss_gamma_{gamma:g}_alpha_{alpha:g}"
    return loss


def configure_model_loss(keras_model, hparams, loss_type, focal_gamma, focal_alpha):
    if loss_type == "bce":
        return

    keras_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=hparams.get("learningRate", 0.0005)
        ),
        loss=binary_focal_loss(gamma=focal_gamma, alpha=focal_alpha),
        weighted_metrics=[],
    )


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


def checkpoint_metadata(
    class_weights=None,
    loss_type="bce",
    focal_gamma=2.0,
    focal_alpha=0.25,
):
    return {
        "class_schema": CLASS_SCHEMA,
        "class_names": CLASS_NAMES,
        "labels": LABELS,
        "class_weights_order": CLASS_NAMES,
        "class_weights": class_weights or DEFAULT_CLASS_WEIGHTS,
        "loss_type": loss_type,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
    }


def save_checkpoint_metadata(
    checkpoint_dir,
    class_weights=None,
    loss_type="bce",
    focal_gamma=2.0,
    focal_alpha=0.25,
):
    metadata_path = checkpoint_dir / CHECKPOINT_METADATA_FILENAME
    save_json(
        metadata_path,
        checkpoint_metadata(class_weights, loss_type, focal_gamma, focal_alpha),
    )
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
    expected_schema = {
        "class_schema": expected["class_schema"],
        "class_names": expected["class_names"],
        "labels": expected["labels"],
    }
    if actual != expected_schema:
        raise ValueError(
            f"Checkpoint class schema mismatch for {weights_path}: "
            f"expected {expected_schema}, got {actual}."
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


def capture_split_batches(data_access, hparams, split, eval_steps):
    if eval_steps <= 0:
        raise ValueError("--eval-steps must be greater than zero")

    split = split.upper()
    index_attr = {
        "TRAIN": "trainIndexes",
        "VAL": "valIndexes",
        "TEST": "testIndexes",
    }.get(split)
    if index_attr is None:
        raise ValueError(f"Unsupported split: {split}")

    test_loader = data_access.trainDataLoaders[0]
    indexes = getattr(test_loader, index_attr)
    if len(indexes) == 0:
        raise ValueError(f"custom_split {split} contains no paired tracks")

    test_gen = test_loader.getGen(indexes, **hparams)
    test_dataset = DataLoader._getDataset(test_gen, **hparams)
    test_iterator = iter(test_dataset)
    batches = []
    for _ in range(eval_steps):
        batch = next(test_iterator)
        inputs, targets = batch[0], batch[1]
        batches.append((inputs, np.asarray(targets)))
    return batches


def capture_test_batches(data_access, hparams, eval_steps):
    return capture_split_batches(data_access, hparams, "TEST", eval_steps)


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


def add_selection_columns(metrics, split, checkpoint):
    """Add aggregate metrics required by the calibrated VAL/TEST protocol."""
    if not metrics:
        return []

    totals_by_threshold = {}
    f1_by_threshold = {}
    for row in metrics:
        threshold = row["threshold"]
        totals = totals_by_threshold.setdefault(threshold, {"TP": 0, "FP": 0, "FN": 0})
        totals["TP"] += int(row["TP"])
        totals["FP"] += int(row["FP"])
        totals["FN"] += int(row["FN"])
        f1_by_threshold.setdefault(threshold, []).append(float(row["f1"]))

    aggregate_by_threshold = {}
    for threshold, totals in totals_by_threshold.items():
        tp = totals["TP"]
        fp = totals["FP"]
        fn = totals["FN"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate_by_threshold[threshold] = {
            "micro_precision": float(precision),
            "micro_recall": float(recall),
            "micro_f1": float(f1),
            "macro_f1": float(np.mean(f1_by_threshold[threshold])),
        }

    enriched = []
    for row in metrics:
        output_row = dict(row)
        output_row.update(aggregate_by_threshold[row["threshold"]])
        output_row["split"] = split.upper()
        output_row["checkpoint"] = checkpoint
        enriched.append(output_row)
    return enriched


def onset_metric_fields(include_tolerance=True):
    fields = [
        "threshold",
        "class",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
    ]
    if include_tolerance:
        fields.append("tolerance_ms")
    fields.extend(
        [
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_f1",
            "split",
            "checkpoint",
        ]
    )
    return fields


def aggregate_threshold_rows(metrics):
    by_threshold = {}
    for row in metrics:
        threshold = row["threshold"]
        totals = by_threshold.setdefault(
            threshold,
            {
                "threshold": threshold,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "class_f1": [],
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "macro_f1": 0.0,
            },
        )
        totals["TP"] += int(row["TP"])
        totals["FP"] += int(row["FP"])
        totals["FN"] += int(row["FN"])
        totals["class_f1"].append(float(row["f1"]))

    rows = []
    for row in by_threshold.values():
        tp = row["TP"]
        fp = row["FP"]
        fn = row["FN"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(row["threshold"]),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "micro_precision": float(precision),
                "micro_recall": float(recall),
                "micro_f1": float(f1),
                "macro_f1": float(np.mean(row["class_f1"])),
            }
        )
    return sorted(rows, key=lambda row: row["threshold"])


def is_better_threshold_candidate(
    candidate,
    current,
    f1_key,
    precision_key,
    eps=THRESHOLD_SELECTION_EPS,
):
    if current is None:
        return True

    candidate_f1 = float(candidate[f1_key])
    current_f1 = float(current[f1_key])
    if candidate_f1 > current_f1 + eps:
        return True
    if candidate_f1 < current_f1 - eps:
        return False

    candidate_precision = float(candidate[precision_key])
    current_precision = float(current[precision_key])
    if candidate_precision > current_precision + eps:
        return True
    if candidate_precision < current_precision - eps:
        return False

    candidate_fp = int(candidate["FP"])
    current_fp = int(current["FP"])
    if candidate_fp < current_fp:
        return True
    if candidate_fp > current_fp:
        return False

    return float(candidate["threshold"]) > float(current["threshold"])


def select_threshold_candidate(rows, f1_key, precision_key):
    selected = None
    for row in rows:
        if is_better_threshold_candidate(row, selected, f1_key, precision_key):
            selected = row
    return selected


def select_global_threshold(metrics):
    aggregate_rows = aggregate_threshold_rows(metrics)
    if not aggregate_rows:
        raise ValueError("No onset metrics available for global threshold selection")
    return select_threshold_candidate(aggregate_rows, "micro_f1", "micro_precision")


def select_per_class_thresholds(metrics):
    selected = {}
    for class_name in CLASS_NAMES:
        class_rows = [row for row in metrics if row["class"] == class_name]
        if not class_rows:
            continue
        selected[class_name] = select_threshold_candidate(class_rows, "f1", "precision")
    return selected


def is_better_checkpoint_candidate(candidate, current):
    if current is None:
        return True

    _, candidate_row, _ = candidate
    _, current_row, _ = current
    if is_better_threshold_candidate(
        candidate_row,
        current_row,
        "micro_f1",
        "micro_precision",
    ):
        return True
    if is_better_threshold_candidate(
        current_row,
        candidate_row,
        "micro_f1",
        "micro_precision",
    ):
        return False

    return candidate[0] == "final" and current[0] != "final"


def select_checkpoint_from_val(val_onset_results):
    candidates = []
    for checkpoint, metrics in val_onset_results.items():
        best_global = select_global_threshold(metrics)
        candidates.append((checkpoint, best_global, metrics))
    if not candidates:
        raise ValueError("No VAL onset metrics available for checkpoint selection")

    selected = None
    for candidate in candidates:
        if is_better_checkpoint_candidate(candidate, selected):
            selected = candidate
    return selected


def filter_metrics_for_global_threshold(metrics, threshold):
    selected = [row for row in metrics if row["threshold"] == threshold]
    return add_selection_columns(selected, "TEST", selected[0]["checkpoint"] if selected else "")


def filter_metrics_for_per_class_thresholds(metrics, thresholds_by_class):
    selected = []
    for row in metrics:
        class_threshold = thresholds_by_class.get(row["class"])
        if class_threshold is not None and row["threshold"] == class_threshold:
            selected.append(row)
    if not selected:
        return []

    totals = {
        "TP": sum(int(row["TP"]) for row in selected),
        "FP": sum(int(row["FP"]) for row in selected),
        "FN": sum(int(row["FN"]) for row in selected),
    }
    precision = totals["TP"] / (totals["TP"] + totals["FP"]) if totals["TP"] + totals["FP"] else 0.0
    recall = totals["TP"] / (totals["TP"] + totals["FN"]) if totals["TP"] + totals["FN"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_f1 = float(np.mean([float(row["f1"]) for row in selected]))

    output = []
    for row in selected:
        output_row = dict(row)
        output_row.update(
            {
                "micro_precision": float(precision),
                "micro_recall": float(recall),
                "micro_f1": float(f1),
                "macro_f1": macro_f1,
                "split": "TEST",
                "checkpoint": row.get("checkpoint", ""),
            }
        )
        output.append(output_row)
    return output


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
        write_csv(onset_path, onset_metrics, onset_metric_fields())
        generated.append(onset_path)
        print_metric_summary(f"{model_label.upper()} onset metrics", onset_metrics)
    return generated


def best_thresholds_by_class(metrics):
    best = {}
    for class_name in CLASS_NAMES:
        class_rows = [row for row in metrics if row["class"] == class_name]
        if class_rows:
            best[class_name] = select_threshold_candidate(class_rows, "f1", "precision")
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


def save_split_onset_metrics(run_dir, split, checkpoint_metrics):
    generated = []
    all_rows = []
    for checkpoint, rows in checkpoint_metrics.items():
        all_rows.extend(rows)
        path = run_dir / f"{checkpoint}_{split.lower()}_onset_metrics_by_threshold.csv"
        write_csv(path, rows, onset_metric_fields())
        generated.append(path)

    if all_rows:
        path = run_dir / f"{split.lower()}_onset_metrics_by_threshold.csv"
        write_csv(path, all_rows, onset_metric_fields())
        generated.append(path)
    return generated


def save_calibration_outputs(
    run_dir,
    dataset_root,
    run_name,
    threshold_grid,
    checkpoints_evaluated,
    selected_checkpoint,
    selected_global_row,
    selected_per_class_rows,
    val_metrics,
    test_global_rows,
    test_per_class_rows,
    onset_tolerance_ms,
):
    selected_per_class_thresholds = {
        class_name: row["threshold"] for class_name, row in selected_per_class_rows.items()
    }
    selected_per_class_metrics = {
        class_name: row for class_name, row in selected_per_class_rows.items()
    }
    selected_global_threshold = selected_global_row["threshold"]
    val_global_rows = [
        row for row in val_metrics if row["threshold"] == selected_global_threshold
    ]

    selected_thresholds_path = run_dir / "selected_thresholds_from_val.json"
    save_json(
        selected_thresholds_path,
        {
            "dataset_root": str(dataset_root),
            "run_name": run_name,
            "class_schema": CLASS_SCHEMA,
            "class_names": CLASS_NAMES,
            "onset_tolerance_ms": onset_tolerance_ms,
            "threshold_grid": threshold_grid,
            "selection_split": "VAL",
            "selected_checkpoint": selected_checkpoint,
            "selection_metric": "onset_micro_f1_for_global_threshold_and_checkpoint; onset_f1_for_per_class_thresholds",
            "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
            "threshold_selection_eps": THRESHOLD_SELECTION_EPS,
            "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
            "selected_global_threshold": selected_global_threshold,
            "selected_per_class_thresholds": selected_per_class_thresholds,
            "val_metrics_for_selected_global_threshold": val_global_rows,
            "val_metrics_for_selected_per_class_thresholds": selected_per_class_metrics,
        },
    )

    checkpoint_selection_path = run_dir / "checkpoint_selection_from_val.json"
    save_json(
        checkpoint_selection_path,
        {
            "dataset_root": str(dataset_root),
            "run_name": run_name,
            "class_schema": CLASS_SCHEMA,
            "class_names": CLASS_NAMES,
            "selection_split": "VAL",
            "selection_metric": "onset_micro_f1",
            "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
            "threshold_selection_eps": THRESHOLD_SELECTION_EPS,
            "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
            "checkpoints_evaluated": checkpoints_evaluated,
            "selected_checkpoint": selected_checkpoint,
            "selected_global_threshold": selected_global_threshold,
            "selected_per_class_thresholds": selected_per_class_thresholds,
            "val_metrics_selected_checkpoint_best_global": selected_global_row,
            "test_metrics_using_val_global_threshold": test_global_rows,
            "test_metrics_using_val_per_class_thresholds": test_per_class_rows,
        },
    )

    return [selected_thresholds_path, checkpoint_selection_path]


def aggregate_selected_rows(rows):
    if not rows:
        return {}
    return {
        "micro_precision": rows[0].get("micro_precision", 0.0),
        "micro_recall": rows[0].get("micro_recall", 0.0),
        "micro_f1": rows[0].get("micro_f1", 0.0),
        "macro_f1": rows[0].get("macro_f1", 0.0),
        "TP": sum(int(row["TP"]) for row in rows),
        "FP": sum(int(row["FP"]) for row in rows),
        "FN": sum(int(row["FN"]) for row in rows),
    }


def format_threshold_map(thresholds_by_class):
    if not thresholds_by_class:
        return "No disponible"
    return ", ".join(
        f"{class_name}={thresholds_by_class[class_name]:g}"
        for class_name in CLASS_NAMES
        if class_name in thresholds_by_class
    )


def write_run_readme(
    run_dir,
    config_data,
    checkpoint_paths,
    calibrate_on_val,
    selected_checkpoint=None,
    selected_global_threshold=None,
    selected_per_class_thresholds=None,
    test_global_rows=None,
    test_per_class_rows=None,
    eval_steps=None,
    split_track_counts=None,
):
    test_global_summary = aggregate_selected_rows(test_global_rows or [])
    test_per_class_summary = aggregate_selected_rows(test_per_class_rows or [])
    checkpoint_lines = []
    for label, path in checkpoint_paths.items():
        status = "available" if path and path.is_file() else "unavailable"
        checkpoint_lines.append(f"- {label}: {status} ({path})")

    warning_lines = []
    if eval_steps is not None and split_track_counts:
        for split in ("VAL", "TEST"):
            track_count = split_track_counts.get(split)
            if track_count and eval_steps < track_count:
                warning_lines.append(
                    f"- WARNING: eval_steps={eval_steps} is lower than {split} track count "
                    f"({track_count}); reported metrics are based on sampled batches, not the full split."
                )
        if not warning_lines:
            warning_lines.append(
                "- Note: evaluation is limited by eval_steps batches; confirm coverage manually if full-split reporting is required."
            )

    lines = [
        "# Run summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- dataset_root: `{config_data['dataset_root']}`",
        f"- class_schema: `{config_data['class_schema']}`",
        f"- class_names: `{', '.join(config_data['class_names'])}`",
        f"- labels: `{', '.join(str(label) for label in config_data['labels'])}`",
        "",
        "## Training configuration",
        "",
        f"- model_name: `{config_data['model_name']}`",
        f"- epochs: `{config_data['epochs']}`",
        f"- steps_per_epoch: `{config_data['steps_per_epoch']}`",
        f"- validation_steps: `{config_data['validation_steps']}`",
        f"- batch_size: `{config_data['batch_size']}`",
        f"- training_sequence: `{config_data['training_sequence']}`",
        f"- context: `{config_data['context']}`",
        f"- sample_rate: `{config_data['sample_rate']}`",
        f"- class_weights_order: `{', '.join(config_data['class_weights_order'])}`",
        f"- class_weights: `{', '.join(str(weight) for weight in config_data['class_weights'])}`",
        f"- loss_type: `{config_data.get('loss_type', 'bce')}`",
        f"- focal_gamma: `{config_data.get('focal_gamma', 2.0)}`",
        f"- focal_alpha: `{config_data.get('focal_alpha', 0.25)}`",
        f"- onset_tolerance_ms: `{config_data['onset_tolerance_ms']}`",
        f"- threshold_grid: `{', '.join(str(threshold) for threshold in config_data['thresholds'])}`",
        "",
        "## Checkpoints",
        "",
        *checkpoint_lines,
        "",
        "## VAL threshold calibration",
        "",
        f"- calibration_enabled: `{bool(calibrate_on_val)}`",
    ]
    if calibrate_on_val and selected_checkpoint:
        lines.extend(
            [
                f"- selection_split: `VAL`",
                f"- selected_checkpoint: `{selected_checkpoint}`",
                f"- threshold_selection_rule: `{THRESHOLD_SELECTION_RULE}`",
                f"- threshold_selection_eps: `{THRESHOLD_SELECTION_EPS}`",
                f"- selected_global_threshold_from_VAL: `{selected_global_threshold:g}`",
                f"- selected_per_class_thresholds_from_VAL: `{format_threshold_map(selected_per_class_thresholds or {})}`",
                "- TEST was used only after threshold/checkpoint selection.",
            ]
        )
    else:
        lines.append("- No VAL calibration was requested for this run.")

    lines.extend(["", "## Final TEST metrics using VAL thresholds", ""])
    if test_global_summary:
        lines.extend(
            [
                "Global VAL threshold applied to TEST:",
                "",
                f"- micro_precision: `{test_global_summary['micro_precision']:.4f}`",
                f"- micro_recall: `{test_global_summary['micro_recall']:.4f}`",
                f"- micro_f1: `{test_global_summary['micro_f1']:.4f}`",
                f"- macro_f1: `{test_global_summary['macro_f1']:.4f}`",
                f"- TP/FP/FN: `{test_global_summary['TP']}/{test_global_summary['FP']}/{test_global_summary['FN']}`",
                "",
            ]
        )
    if test_per_class_summary:
        lines.extend(
            [
                "Per-class VAL thresholds applied to TEST:",
                "",
                f"- micro_precision: `{test_per_class_summary['micro_precision']:.4f}`",
                f"- micro_recall: `{test_per_class_summary['micro_recall']:.4f}`",
                f"- micro_f1: `{test_per_class_summary['micro_f1']:.4f}`",
                f"- macro_f1: `{test_per_class_summary['macro_f1']:.4f}`",
                f"- TP/FP/FN: `{test_per_class_summary['TP']}/{test_per_class_summary['FP']}/{test_per_class_summary['FN']}`",
                "",
            ]
        )
    if not test_global_summary and not test_per_class_summary:
        lines.append("No calibrated TEST metrics were generated.")

    if warning_lines:
        lines.extend(["", "## Evaluation coverage", "", *warning_lines])

    readme_path = run_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


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
    if args.calibrate_thresholds_on_val and args.no_val:
        raise SystemExit("--calibrate-thresholds-on-val requires validation; remove --no-val")

    class_weights = validate_class_weights(args.class_weights)
    loss_type, focal_gamma, focal_alpha = validate_loss_args(
        args.loss_type,
        args.focal_gamma,
        args.focal_alpha,
    )
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    thresholds = validate_thresholds(args.thresholds, "--thresholds")
    onset_thresholds = validate_thresholds(
        args.onset_thresholds or thresholds, "--onset-thresholds"
    )
    eval_onsets = bool(args.eval_onsets or args.calibrate_thresholds_on_val)
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
        "class_weights_order": CLASS_NAMES,
        "class_weights": class_weights,
        "loss_type": loss_type,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "same_padding": args.same_padding,
        "no_val": args.no_val,
        "threshold": args.threshold,
        "thresholds": thresholds,
        "eval_onsets": eval_onsets,
        "calibrate_thresholds_on_val": args.calibrate_thresholds_on_val,
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
    print(
        "class_weights: "
        + ", ".join(
            f"{class_name}={weight:g}"
            for class_name, weight in zip(CLASS_NAMES, class_weights)
        )
    )
    print(f"loss_type: {loss_type}")
    print(f"focal_gamma: {focal_gamma}")
    print(f"focal_alpha: {focal_alpha}")
    print(f"save_checkpoints: {args.save_checkpoints}")
    print(f"eval_test: {args.eval_test}")
    print(f"calibrate_thresholds_on_val: {args.calibrate_thresholds_on_val}")

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
        "sampleWeight": class_weights,
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
    configure_model_loss(model.model, hparams, loss_type, focal_gamma, focal_alpha)
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
    custom_loader = data_access.trainDataLoaders[0]
    split_track_counts = {
        "TRAIN": len(custom_loader.trainIndexes),
        "VAL": len(custom_loader.valIndexes),
        "TEST": len(custom_loader.testIndexes),
    }

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
        generated_files.append(
            save_checkpoint_metadata(
                checkpoint_dir,
                class_weights,
                loss_type,
                focal_gamma,
                focal_alpha,
            )
        )
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
    val_onset_results = {}
    test_onset_results = {}
    selected_checkpoint = None
    selected_global_threshold = None
    selected_per_class_thresholds = None
    test_global_rows = []
    test_per_class_rows = []
    should_evaluate = bool(args.eval_test or args.calibrate_thresholds_on_val)
    if should_evaluate:
        try:
            test_batches = capture_split_batches(data_access, hparams, "TEST", args.eval_steps)
            val_batches = (
                capture_split_batches(data_access, hparams, "VAL", args.eval_steps)
                if args.calibrate_thresholds_on_val
                else None
            )
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
                    if args.calibrate_thresholds_on_val:
                        _, _, val_onset_metrics, val_onset_error = evaluate_loaded_model(
                            model.model,
                            val_batches,
                            thresholds,
                            onset_thresholds,
                            args.sample_rate,
                            args.onset_tolerance_ms,
                            True,
                        )
                        if val_onset_error is not None:
                            print(f"ERROR: {model_label} VAL onset evaluation failed: {val_onset_error}")
                            raise val_onset_error
                        elif val_onset_metrics is not None:
                            val_onset_results[model_label] = add_selection_columns(
                                val_onset_metrics, "VAL", model_label
                            )

                    metrics, prediction_stats, onset_metrics, onset_error = evaluate_loaded_model(
                        model.model,
                        test_batches,
                        thresholds,
                        onset_thresholds,
                        args.sample_rate,
                        args.onset_tolerance_ms,
                        eval_onsets,
                    )
                    if onset_metrics is not None:
                        onset_metrics = add_selection_columns(
                            onset_metrics, "TEST", model_label
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
                        test_onset_results[model_label] = onset_metrics
                    if onset_error is not None:
                        print(
                            f"ERROR: {model_label} onset evaluation failed; "
                            f"frame-wise results remain valid: {onset_error}"
                        )
                        if args.calibrate_thresholds_on_val:
                            raise onset_error
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
                    if args.calibrate_thresholds_on_val:
                        raise

            if args.calibrate_thresholds_on_val:
                generated_files.extend(save_split_onset_metrics(run_dir, "VAL", val_onset_results))
                generated_files.extend(save_split_onset_metrics(run_dir, "TEST", test_onset_results))
                selected_checkpoint, selected_global_row, selected_val_metrics = select_checkpoint_from_val(
                    val_onset_results
                )
                selected_global_threshold = selected_global_row["threshold"]
                selected_per_class_rows = select_per_class_thresholds(selected_val_metrics)
                selected_per_class_thresholds = {
                    class_name: row["threshold"]
                    for class_name, row in selected_per_class_rows.items()
                }
                selected_test_metrics = test_onset_results[selected_checkpoint]
                test_global_rows = filter_metrics_for_global_threshold(
                    selected_test_metrics, selected_global_threshold
                )
                test_per_class_rows = filter_metrics_for_per_class_thresholds(
                    selected_test_metrics, selected_per_class_thresholds
                )
                global_path = run_dir / "final_test_onset_metrics_using_val_global_threshold.csv"
                per_class_path = run_dir / "final_test_onset_metrics_using_val_per_class_thresholds.csv"
                write_csv(global_path, test_global_rows, onset_metric_fields())
                write_csv(per_class_path, test_per_class_rows, onset_metric_fields())
                generated_files.extend([global_path, per_class_path])
                generated_files.extend(
                    save_calibration_outputs(
                        run_dir,
                        dataset_root,
                        run_dir.name,
                        onset_thresholds,
                        sorted(val_onset_results),
                        selected_checkpoint,
                        selected_global_row,
                        selected_per_class_rows,
                        selected_val_metrics,
                        test_global_rows,
                        test_per_class_rows,
                        args.onset_tolerance_ms,
                    )
                )
        except Exception as error:
            print(f"ERROR: evaluation failed: {error}")
            if args.calibrate_thresholds_on_val:
                raise

    readme_path = write_run_readme(
        run_dir,
        config_data,
        {
            "best": best_checkpoint_path,
            "final": final_checkpoint_path,
        },
        args.calibrate_thresholds_on_val,
        selected_checkpoint=selected_checkpoint,
        selected_global_threshold=selected_global_threshold,
        selected_per_class_thresholds=selected_per_class_thresholds,
        test_global_rows=test_global_rows,
        test_per_class_rows=test_per_class_rows,
        eval_steps=args.eval_steps if should_evaluate else None,
        split_track_counts=split_track_counts,
    )
    generated_files.append(readme_path)

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
