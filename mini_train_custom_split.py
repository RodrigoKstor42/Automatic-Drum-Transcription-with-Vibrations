"""
Minimal custom_split training smoke test for ADTOF.

This script reuses Model.modelFactory(), which builds and compiles the Keras
model in adtof/model/model.py. It keeps training intentionally tiny: the goal is
only to validate DataLoader -> model -> loss -> backpropagation.
"""
import argparse
import collections
import collections.abc
import os
import warnings
from pathlib import Path

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

from adtof import config
from adtof.model.dataLoader import DataLoader
from adtof.model.model import Model


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
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    print(f"dataset_root: {dataset_root}")
    print(f"epochs: {args.epochs}")
    print(f"steps_per_epoch: {args.steps_per_epoch}")
    print(f"validation_steps: {args.validation_steps}")
    print(f"batch_size: {args.batch_size}")

    cache_root = dataset_root / "PREPROCESS_MINI_TRAIN"
    common_kwargs = {
        "sampleRate": args.sample_rate,
        "trainingSequence": args.training_sequence,
        "batchSize": args.batch_size,
        "context": args.context,
        "labels": config.LABELS_5,
        "sampleWeight": config.WEIGHTS_5,
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

    history = model.model.fit(
        data_access.train_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=[],
        verbose=1,
    )

    loss = history.history.get("loss", [None])[-1]
    val_loss = history.history.get("val_loss", [None])[-1]
    print(f"loss: {loss}")
    if val_loss is not None:
        print(f"val_loss: {val_loss}")


if __name__ == "__main__":
    main()
