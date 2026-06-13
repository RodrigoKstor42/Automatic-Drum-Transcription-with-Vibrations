from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_TARGET_SR = 12_000
DEFAULT_SUBTYPE = "PCM_16"
DEFAULT_PEAK_LIMIT = 0.98
SPLITS = ("TRAIN", "VAL", "TEST")


def find_wav_files(dataset_root: Path) -> list[Path]:
    split_files = [
        path
        for split in SPLITS
        for path in sorted((dataset_root / split / "AUDIO").glob("*.wav"))
    ]
    if split_files:
        return split_files

    raw_audio_dir = dataset_root / "AUDIO"
    if raw_audio_dir.is_dir():
        return sorted(raw_audio_dir.glob("*.wav"))

    return sorted(dataset_root.rglob("*.wav"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check technical consistency of generated dataset WAV files."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR)
    parser.add_argument("--expected-subtype", default=DEFAULT_SUBTYPE)
    parser.add_argument("--peak-limit", type=float, default=DEFAULT_PEAK_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    wav_paths = find_wav_files(dataset_root)

    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    subtypes: Counter[str] = Counter()
    durations: list[float] = []
    max_peak = 0.0
    clipped_files = 0
    over_peak_limit_files = 0
    invalid_files = 0
    unreadable_files: list[tuple[Path, str]] = []

    pcm16_full_scale = 1.0 - (1.0 / 32768.0)
    for wav_path in wav_paths:
        try:
            info = sf.info(wav_path)
            audio, sample_rate = sf.read(wav_path, dtype="float64", always_2d=True)
        except (RuntimeError, OSError) as exc:
            unreadable_files.append((wav_path, str(exc)))
            continue

        sample_rates[sample_rate] += 1
        channels[info.channels] += 1
        subtypes[info.subtype] += 1
        durations.append(info.duration)

        finite_mask = np.isfinite(audio)
        if not np.all(finite_mask):
            invalid_files += 1
        finite_audio = audio[finite_mask]
        peak = float(np.max(np.abs(finite_audio))) if finite_audio.size else 0.0
        max_peak = max(max_peak, peak)
        if peak >= pcm16_full_scale:
            clipped_files += 1
        if peak > args.peak_limit + (1.0 / 32768.0):
            over_peak_limit_files += 1

    checked = len(wav_paths) - len(unreadable_files)
    total_duration = sum(durations)
    min_duration = min(durations, default=0.0)
    max_duration = max(durations, default=0.0)

    print(f"Checked WAV files: {checked}")
    print(f"Sample rates found: {dict(sorted(sample_rates.items()))}")
    print(f"Channels found: {dict(sorted(channels.items()))}")
    print(f"Subtypes found: {dict(sorted(subtypes.items()))}")
    print(
        "Duration seconds: "
        f"min={min_duration:.3f}, max={max_duration:.3f}, total={total_duration:.3f}"
    )
    print(f"Max absolute peak: {max_peak:.6f}")
    print(f"Clipped files: {clipped_files}")
    print(f"Files above peak limit ({args.peak_limit:.3f}): {over_peak_limit_files}")
    print(f"NaN/Inf files: {invalid_files}")
    print(f"Unreadable files: {len(unreadable_files)}")

    consistent = (
        bool(wav_paths)
        and not unreadable_files
        and set(sample_rates) == {args.target_sr}
        and set(channels) == {1}
        and set(subtypes) == {args.expected_subtype}
        and clipped_files == 0
        and over_peak_limit_files == 0
        and invalid_files == 0
    )

    if consistent:
        print("\n[OK] Dataset WAV format is consistent.")
        return

    if not wav_paths:
        print(f"\n[ERROR] No WAV files found under: {dataset_root}")
    else:
        print("\n[ERROR] Dataset WAV format has inconsistencies.")
    for path, error in unreadable_files[:10]:
        print(f"  Unreadable: {path} ({error})")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
