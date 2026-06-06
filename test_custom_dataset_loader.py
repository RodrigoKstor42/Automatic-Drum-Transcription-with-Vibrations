#!/usr/bin/env python
"""
Smoke test for the custom_split DataLoader scenario.

This script validates loading, preprocessing, cache generation, label creation,
and one batched sample from a small copied subset of the generated dataset.
It does not train the model.
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np


SPLITS = ("TRAIN", "VAL", "TEST")
TARGET_CLASS_NAMES = ("KD", "SD", "TT", "HH", "CY")


def get_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Smoke test custom_split dataset loading.")
    parser.add_argument("--dataset-root", default=str(repo_root / "PHRASE_GENERATOR"), help="Root with TRAIN/VAL/TEST folders.")
    parser.add_argument("--tracks-per-split", type=int, default=1, help="Number of paired WAV/TXT files to copy per split.")
    parser.add_argument("--train-tracks", type=int, default=None, help="Number of paired WAV/TXT files to copy from TRAIN.")
    parser.add_argument("--val-tracks", type=int, default=None, help="Number of paired WAV/TXT files to copy from VAL.")
    parser.add_argument("--test-tracks", type=int, default=None, help="Number of paired WAV/TXT files to copy from TEST.")
    parser.add_argument("--training-sequence", type=int, default=64, help="Target sequence length for one training batch.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used by the DataLoader smoke batch.")
    parser.add_argument("--sample-rate", type=int, default=100, help="Feature frame rate passed to ADTOF preprocessing.")
    parser.add_argument("--context", type=int, default=9, help="Input context used by Track.getSlice.")
    parser.add_argument("--work-dir", default=None, help="Optional directory for the temporary subset.")
    parser.add_argument("--keep-subset", action="store_true", help="Keep the temporary subset after the test finishes.")
    return parser.parse_args()


def get_tracks_by_split(args):
    tracks_by_split = {
        "TRAIN": args.train_tracks if args.train_tracks is not None else args.tracks_per_split,
        "VAL": args.val_tracks if args.val_tracks is not None else args.tracks_per_split,
        "TEST": args.test_tracks if args.test_tracks is not None else args.tracks_per_split,
    }
    for split_name, track_count in tracks_by_split.items():
        if track_count < 1:
            raise SystemExit(f"{split_name} track count must be at least 1, got {track_count}.")
    return tracks_by_split


def find_pairs(dataset_root, split_name):
    audio_dir = dataset_root / split_name / "AUDIO"
    annotation_dir = dataset_root / split_name / "ANNOTATIONS"
    if not audio_dir.is_dir():
        raise AssertionError(f"Missing audio directory: {audio_dir}")
    if not annotation_dir.is_dir():
        raise AssertionError(f"Missing annotation directory: {annotation_dir}")

    audio_by_stem = {path.stem: path for path in sorted(audio_dir.iterdir()) if path.suffix.lower() == ".wav"}
    annotation_by_stem = {path.stem: path for path in sorted(annotation_dir.iterdir()) if path.suffix.lower() == ".txt"}

    missing_annotations = sorted(set(audio_by_stem) - set(annotation_by_stem))
    missing_audio = sorted(set(annotation_by_stem) - set(audio_by_stem))
    if missing_annotations:
        logging.warning("%s WAV files without TXT: %s", split_name, missing_annotations[:10])
    if missing_audio:
        logging.warning("%s TXT files without WAV: %s", split_name, missing_audio[:10])

    paired_stems = sorted(set(audio_by_stem) & set(annotation_by_stem))
    return [(audio_by_stem[stem], annotation_by_stem[stem]) for stem in paired_stems]


def copy_subset(dataset_root, subset_root, tracks_by_split):
    for split_name in SPLITS:
        requested_tracks = tracks_by_split[split_name]
        pairs = find_pairs(dataset_root, split_name)
        if len(pairs) < requested_tracks:
            raise AssertionError(
                f"{split_name} requested {requested_tracks} paired WAV/TXT files, but only {len(pairs)} are available."
            )

        audio_out = subset_root / split_name / "AUDIO"
        annotation_out = subset_root / split_name / "ANNOTATIONS"
        audio_out.mkdir(parents=True, exist_ok=True)
        annotation_out.mkdir(parents=True, exist_ok=True)

        for audio_path, annotation_path in pairs[:requested_tracks]:
            shutil.copy2(audio_path, audio_out / audio_path.name)
            shutil.copy2(annotation_path, annotation_out / annotation_path.name)
        logging.info("%s paired files copied: %s", split_name, requested_tracks)


def assert_shape(name, actual, expected):
    if tuple(actual) != tuple(expected):
        raise AssertionError(f"{name} shape mismatch. Expected {expected}, got {tuple(actual)}")


def count_annotation_events(annotation_path):
    with annotation_path.open("r", encoding="utf-8") as annotation_file:
        return sum(1 for line in annotation_file if line.strip())


def get_target_positive_counts(y_dense):
    positive_targets = np.asarray(y_dense) > 0
    return positive_targets.sum(axis=0).astype(int)


def print_target_summary(positive_counts, annotation_events, y_dense_shape):
    total_positive_frames = int(positive_counts.sum())

    print("target_positive_counts_accumulated:")
    for class_name, count in zip(TARGET_CLASS_NAMES, positive_counts):
        print(f"{class_name} count={int(count)}")
    print(f"total positive frames={total_positive_frames}")
    print(f"annotation events={annotation_events}")
    print(f"yDense shape={y_dense_shape}")

    for class_name, count in zip(TARGET_CLASS_NAMES[3:], positive_counts[3:]):
        if count > 0:
            logging.warning("%s has %d positive target frames; expected zero for this custom dataset.", class_name, int(count))


def validate_all_tracks(dataloader, split_indexes, expected_feature_dim, expected_target_classes, n_channels, track_cls):
    if dataloader.preprocessPaths is None:
        raise AssertionError("Expected preprocess cache paths to be configured.")

    accumulated_counts = np.zeros(expected_target_classes, dtype=int)
    total_annotation_events = 0
    no_positive_tracks = []
    track_errors = []
    checked_tracks = 0
    y_dense_shapes = set()

    for split_name, indexes in split_indexes.items():
        for track_idx in indexes:
            track_idx = int(track_idx)
            audio_path = Path(dataloader.audioPaths[track_idx])
            annotation_path = Path(dataloader.annotationPaths[track_idx])
            cache_path = Path(dataloader.preprocessPaths[track_idx])
            track_label = f"{split_name}/{audio_path.name}"

            try:
                if audio_path.stem != annotation_path.stem:
                    raise AssertionError(f"Unpaired track: {audio_path} / {annotation_path}")
                if cache_path.exists():
                    cache_path.unlink()

                track = dataloader.data[track_idx]
                if not isinstance(track, track_cls):
                    raise AssertionError(f"Expected Track instance, got {type(track)}")
                if not cache_path.exists():
                    raise AssertionError(f"Expected cached .npy file to be generated: {cache_path}")
                if cache_path.suffix != ".npy":
                    raise AssertionError(f"Expected .npy cache extension, got: {cache_path}")

                if track.x.ndim != 3:
                    raise AssertionError(f"Expected track.x to be 3D, got shape {track.x.shape}")
                if track.x.shape[1] != expected_feature_dim:
                    raise AssertionError(f"Unexpected feature dim: {track.x.shape[1]} != {expected_feature_dim}")
                if track.x.shape[2] != n_channels:
                    raise AssertionError(f"Unexpected channel count: {track.x.shape[2]}")
                if not hasattr(track, "yDense"):
                    raise AssertionError("Track did not create dense labels.")
                if not hasattr(track, "sampleWeight"):
                    raise AssertionError("Track did not create sample weights.")
                if track.yDense.shape[1] != expected_target_classes:
                    raise AssertionError(f"Unexpected target class count: {track.yDense.shape[1]}")

                positive_counts = get_target_positive_counts(track.yDense)
                total_positive_frames = int(positive_counts.sum())
                if total_positive_frames == 0:
                    no_positive_tracks.append(track_label)

                accumulated_counts += positive_counts
                total_annotation_events += count_annotation_events(annotation_path)
                y_dense_shapes.add(tuple(track.yDense.shape))
                checked_tracks += 1
                logging.info("Track validated: %s x=%s yDense=%s cache=%s", track_label, track.x.shape, track.yDense.shape, cache_path)
            except Exception as exc:
                track_errors.append(f"{track_label}: {type(exc).__name__}: {exc}")

    print_target_summary(accumulated_counts, total_annotation_events, sorted(y_dense_shapes))
    print(f"tracks checked={checked_tracks}")

    if no_positive_tracks:
        print("tracks without positive targets:")
        for track_label in no_positive_tracks:
            print(f"- {track_label}")

    if track_errors:
        print("tracks with audio or label errors:")
        for error in track_errors:
            print(f"- {error}")
        raise AssertionError(f"{len(track_errors)} tracks failed audio/label validation.")
    if int(accumulated_counts.sum()) == 0:
        raise AssertionError("All target classes are zero after building yDense for the copied subset.")

    return checked_tracks


def run_smoke_test(args):
    try:
        from adtof import config
        from adtof.io import mir
        from adtof.model.dataLoader import DataLoader
        from adtof.model.track import Track
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Activate/install the ADTOF environment before running this smoke test."
        ) from exc

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise AssertionError(f"Dataset root does not exist: {dataset_root}")

    temp_parent = Path(args.work_dir).resolve() if args.work_dir else None
    subset_root = Path(tempfile.mkdtemp(prefix="custom_split_smoke_", dir=temp_parent))
    logging.info("Temporary custom subset: %s", subset_root)

    try:
        tracks_by_split = get_tracks_by_split(args)
        copy_subset(dataset_root, subset_root, tracks_by_split)

        cache_root = subset_root / "PREPROCESS"
        labels = config.LABELS_5
        sample_weight = config.WEIGHTS_5
        common_kwargs = {
            "sampleRate": args.sample_rate,
            "trainingSequence": args.training_sequence,
            "batchSize": args.batch_size,
            "context": args.context,
            "labels": labels,
            "sampleWeight": sample_weight,
            "prefetch": None,
            "n_channels": 1,
        }

        dataloader = DataLoader.getCustomSplit(
            customDatasetRoot=str(subset_root),
            cachePreprocessFolders=str(cache_root),
            **common_kwargs,
        )
        assert len(dataloader.trainIndexes) == tracks_by_split["TRAIN"]
        assert len(dataloader.valIndexes) == tracks_by_split["VAL"]
        assert len(dataloader.testIndexes) == tracks_by_split["TEST"]
        assert len(dataloader.audioPaths) == len(dataloader.annotationPaths)
        logging.info(
            "DataLoader splits: train=%s val=%s test=%s",
            len(dataloader.trainIndexes),
            len(dataloader.valIndexes),
            len(dataloader.testIndexes),
        )

        expected_feature_dim = mir.getDim(**common_kwargs)
        checked_tracks = validate_all_tracks(
            dataloader,
            {
                "TRAIN": dataloader.trainIndexes,
                "VAL": dataloader.valIndexes,
                "TEST": dataloader.testIndexes,
            },
            expected_feature_dim,
            len(labels),
            common_kwargs["n_channels"],
            Track,
        )

        data_access = DataLoader.factoryMixedDatasets(
            folderPath=str(subset_root),
            scenario="custom_split",
            cachePreprocessFolders=str(cache_root),
            **common_kwargs,
        )
        batch = next(iter(data_access.train_dataset))
        if len(batch) != 3:
            raise AssertionError(f"Expected inputs, targets, weights; got {len(batch)} values.")

        inputs, targets, weights = batch
        if "x" not in inputs:
            raise AssertionError("Batch inputs are missing the 'x' key.")

        x_shape = tuple(inputs["x"].shape)
        target_shape = tuple(targets.shape)
        weight_shape = tuple(weights.shape)
        expected_input_frames = args.training_sequence + (args.context - 1)
        assert_shape("inputs['x']", x_shape, (args.batch_size, expected_input_frames, expected_feature_dim, common_kwargs["n_channels"]))
        assert_shape("targets", target_shape, (args.batch_size, args.training_sequence, len(labels)))
        assert_shape("weights", weight_shape, (args.batch_size, args.training_sequence))

        if not np.any(targets.numpy()):
            logging.warning("The smoke batch contains no positive labels. Loading still passed, but choose another file if needed.")

        print("custom_split smoke test passed")
        print(f"subset_root={subset_root}")
        print(f"tracks_checked={checked_tracks}")
        print(f"batch_x_shape={x_shape}")
        print(f"batch_target_shape={target_shape}")
        print(f"batch_weight_shape={weight_shape}")
    finally:
        if args.keep_subset:
            logging.info("Keeping temporary subset: %s", subset_root)
        else:
            shutil.rmtree(subset_root, ignore_errors=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_smoke_test(get_args())


if __name__ == "__main__":
    main()
