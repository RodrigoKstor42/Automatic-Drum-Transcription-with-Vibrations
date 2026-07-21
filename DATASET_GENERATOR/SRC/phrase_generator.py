from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from generation_profiles import GENERATION_PROFILES, STANDARD_ROCK, GenerationProfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from json_to_adtof_txt import extract_adtof_rows, write_annotation

SAMPLES_DIR = PROJECT_ROOT / "MUESTRAS"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "GENERATED_RAW"
OUTPUT_AUDIO_DIR = DEFAULT_OUTPUT_ROOT / "AUDIO"
OUTPUT_LABELS_DIR = DEFAULT_OUTPUT_ROOT / "LABELS"

TARGET_SR = 12_000
DEFAULT_WAV_SUBTYPE = "PCM_16"
DEFAULT_PEAK_LIMIT = 0.98
PEAK_TARGET = 0.94
DEFAULT_BPM = 120
DEFAULT_PHRASE_COUNT = 100
GRID_STEPS = 16
BEATS_PER_PHRASE = 4
MIN_EVENTS_PER_PHRASE = 4
MAX_EVENTS_PER_PHRASE = 12
FADE_SECONDS = 0.005
TAIL_PADDING_SECONDS = 1.25
MICROTIMING_SECONDS_RANGE = (-0.005, 0.005)
INSTRUMENT_CLASSES = ("KD", "SD", "T12", "T14", "T16")
TOM_CLASSES = ("T12", "T14", "T16")
CLASS_NAMES = INSTRUMENT_CLASSES
CLASS_TO_MIDI = {
    "KD": 35,
    "SD": 38,
    "T12": 47,
    "T14": 45,
    "T16": 43,
}
SPLITS = ("TRAIN", "VAL", "TEST")
DEFAULT_TARGET_TOM_SHARE = 0.36
DEFAULT_TOM_BOOST = 1.5
TOMBOOST_MIN_EVENTS = 10
FILL_PROBABILITY = 0.72
FINAL_TWO_BEAT_FILL_PROBABILITY = 0.28
GHOST_NOTE_GAIN_RANGE = (0.22, 0.42)
ACCENT_GAIN_RANGE = (0.82, 1.0)
TOM_FILL_GAIN_RANGE = (0.58, 0.88)
FILL_TOM_PATH = ("T12", "T14", "T16")
FILL_MOTIFS = (
    (0, 2),
    (0, 1, 3),
    (0, 2, 3),
    (0, 1, 2),
)
FILL_RESOLUTIONS = ("KD", "KD", "SD", "KD+SD")
PRE_MIX_HEADROOM_GAIN = 0.88
INSTRUMENT_PRE_MIX_GAINS = {
    "KD": 0.82,
    "SD": 0.9,
    "T12": 0.98,
    "T14": 1.0,
    "T16": 1.02,
}
SOFT_LIMIT_THRESHOLD = 0.96
SOFT_LIMIT_DRIVE = 1.15
KD_STEPS = (0, 8, 2, 6, 10, 14, 4, 12, 3, 7, 11, 15)
SD_STEPS = (4, 12, 7, 11, 15, 3, 9, 13, 2, 6, 10, 14)
TOM_STEPS = {
    "T12": (9, 10, 11, 8, 12, 13, 14, 15),
    "T14": (11, 12, 13, 10, 14, 15, 9, 8),
    "T16": (13, 14, 15, 12, 11, 10, 9, 8),
}


@dataclass(frozen=True)
class HitEvent:
    instrument: str
    sample_class: str
    beat: float
    gain: float = 1.0
    timing_offset_seconds: float = 0.0


@dataclass(frozen=True)
class GroovePattern:
    name: str
    kick_steps: tuple[int, ...]
    snare_steps: tuple[int, ...]
    optional_kick_steps: tuple[int, ...] = ()
    ghost_snare_steps: tuple[int, ...] = ()


GROOVE_PATTERNS = (
    GroovePattern(
        name="backbeat_basic",
        kick_steps=(0, 8),
        snare_steps=(4, 12),
        optional_kick_steps=(10, 14),
        ghost_snare_steps=(3, 7, 11, 15),
    ),
    GroovePattern(
        name="syncopated_kick",
        kick_steps=(0, 6, 8),
        snare_steps=(4, 12),
        optional_kick_steps=(10, 14),
        ghost_snare_steps=(2, 7, 11, 15),
    ),
    GroovePattern(
        name="four_on_floor_backbeat",
        kick_steps=(0, 4, 8, 12),
        snare_steps=(4, 12),
        optional_kick_steps=(14,),
        ghost_snare_steps=(7, 11, 15),
    ),
    GroovePattern(
        name="sparse_rock",
        kick_steps=(0, 10),
        snare_steps=(4, 12),
        optional_kick_steps=(6, 8, 14),
        ghost_snare_steps=(3, 11, 15),
    ),
)


@dataclass(frozen=True)
class RenderedPhrase:
    audio: np.ndarray
    labels: list[dict]
    sample_rate: int
    bpm: float
    tail_padding_seconds: float


def make_phrase_id(profile_name: str, phrase_index: int, width: int = 6) -> str:
    if phrase_index <= 0:
        raise ValueError("Phrase index must be greater than zero.")
    return f"{profile_name}_phrase_{phrase_index:0{width}d}"


def seconds_per_beat(bpm: float) -> float:
    if bpm <= 0:
        raise ValueError("BPM must be greater than zero.")
    return 60.0 / bpm


@lru_cache(maxsize=None)
def list_wav_samples(
    sample_class: str,
    samples_dir: Path = SAMPLES_DIR,
) -> tuple[Path, ...]:
    class_dir = samples_dir / sample_class
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Sample class directory not found: {class_dir}")

    wavs = tuple(sorted(class_dir.glob("*.wav")))
    if not wavs:
        raise FileNotFoundError(f"No WAV samples found in: {class_dir}")
    return wavs


def choose_sample(
    sample_class: str,
    samples_dir: Path = SAMPLES_DIR,
    rng: random.Random | None = None,
) -> Path:
    samples = list_wav_samples(sample_class, samples_dir)
    if rng is None:
        return samples[0]
    return rng.choice(samples)


def load_hit(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, source_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1, dtype=np.float32)
    if source_sr != target_sr:
        common_divisor = gcd(source_sr, target_sr)
        audio = resample_poly(
            audio,
            up=target_sr // common_divisor,
            down=source_sr // common_divisor,
        ).astype(np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    return audio.astype(np.float32, copy=False)


def apply_fades(audio: np.ndarray, sample_rate: int, fade_seconds: float = FADE_SECONDS) -> np.ndarray:
    if audio.size == 0:
        return audio

    fade_len = min(int(round(fade_seconds * sample_rate)), audio.size // 2)
    if fade_len <= 1:
        return audio

    faded = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    faded[:fade_len] *= fade_in
    faded[-fade_len:] *= fade_out
    return faded


def peak_normalize(audio: np.ndarray, target_peak: float = PEAK_TARGET) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32, copy=False)
    return (audio * (target_peak / peak)).astype(np.float32)


def prepare_wav_audio(audio: np.ndarray, peak_limit: float = DEFAULT_PEAK_LIMIT) -> np.ndarray:
    if not 0.0 < peak_limit <= 1.0:
        raise ValueError("peak_limit must be greater than 0 and less than or equal to 1.")

    prepared = np.asarray(audio)
    if prepared.ndim == 2:
        if prepared.shape[0] <= 8 and prepared.shape[1] > prepared.shape[0]:
            prepared = prepared.T
        prepared = np.mean(prepared, axis=1)
    elif prepared.ndim != 1:
        raise ValueError(f"Audio must be one- or two-dimensional, got shape {prepared.shape}.")

    prepared = prepared.astype(np.float64, copy=False)
    invalid_count = int(np.count_nonzero(~np.isfinite(prepared)))
    if invalid_count:
        warnings.warn(
            f"Replacing {invalid_count} NaN/Inf audio samples with zero.",
            RuntimeWarning,
            stacklevel=2,
        )
        prepared = np.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(prepared))) if prepared.size else 0.0
    if peak > peak_limit:
        prepared = prepared * (peak_limit / peak)

    return np.clip(prepared, -peak_limit, peak_limit).astype(np.float32)


def write_wav_pcm16(
    path: Path,
    audio: np.ndarray,
    sr: int = TARGET_SR,
    peak_limit: float = DEFAULT_PEAK_LIMIT,
    subtype: str = DEFAULT_WAV_SUBTYPE,
) -> np.ndarray:
    if sr <= 0:
        raise ValueError("Sample rate must be greater than zero.")
    if subtype != "PCM_16":
        raise ValueError("The normalized dataset WAV subtype must be PCM_16.")

    prepared = prepare_wav_audio(audio, peak_limit=peak_limit)
    pcm16 = np.round(prepared * np.iinfo(np.int16).max).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, pcm16, sr, format="WAV", subtype=subtype)
    return prepared


def pre_mix_gain(sample_class: str) -> float:
    return PRE_MIX_HEADROOM_GAIN * INSTRUMENT_PRE_MIX_GAINS.get(sample_class, 1.0)


def soft_limit_if_needed(
    audio: np.ndarray,
    threshold: float = SOFT_LIMIT_THRESHOLD,
    drive: float = SOFT_LIMIT_DRIVE,
) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= threshold:
        return audio.astype(np.float32, copy=False)

    limited = audio.copy()
    over_mask = np.abs(limited) > threshold
    over = np.abs(limited[over_mask]) - threshold
    limited[over_mask] = np.sign(limited[over_mask]) * (
        threshold + np.tanh(over * drive) / drive
    )
    return limited.astype(np.float32, copy=False)


def detect_overlaps(labels: list[dict]) -> bool:
    intervals = sorted(
        (
            float(label["timestamp"]),
            float(label["timestamp"]) + float(label["duration"]),
        )
        for label in labels
    )
    previous_end = 0.0
    for start, end in intervals:
        if start < previous_end:
            return True
        previous_end = max(previous_end, end)
    return False


def estimate_density(labels: list[dict], bpm: float, beats_per_phrase: int = BEATS_PER_PHRASE) -> str:
    phrase_beats_per_second = bpm / 60.0
    musical_duration = beats_per_phrase / phrase_beats_per_second
    events_per_second = len(labels) / musical_duration if musical_duration > 0.0 else 0.0

    if events_per_second < 2.0:
        return "low"
    if events_per_second < 3.2:
        return "medium"
    return "high"


def detect_simple_fill(labels: list[dict], bpm: float, beats_per_phrase: int = BEATS_PER_PHRASE) -> bool:
    beat_seconds = seconds_per_beat(bpm)
    final_half_start = (beats_per_phrase / 2.0) * beat_seconds
    final_events = [
        label
        for label in labels
        if float(label["timestamp"]) >= final_half_start
    ]
    tom_events = [
        label
        for label in final_events
        if str(label["instrument"]).startswith("T")
    ]
    return bool(tom_events) or len(final_events) >= 5


def render_phrase(
    events: Iterable[HitEvent],
    bpm: float = DEFAULT_BPM,
    samples_dir: Path = SAMPLES_DIR,
    sample_rate: int = TARGET_SR,
    seed: int = 7,
    tail_padding_seconds: float = TAIL_PADDING_SECONDS,
) -> RenderedPhrase:
    ordered_events = sorted(events, key=lambda event: event.beat)
    if not ordered_events:
        raise ValueError("At least one hit event is required.")

    beat_seconds = seconds_per_beat(bpm)
    rng = random.Random(seed)

    prepared_hits: list[tuple[HitEvent, Path, int, np.ndarray]] = []
    for event in ordered_events:
        sample_path = choose_sample(event.sample_class, samples_dir, rng)
        onset_seconds = max(0.0, event.beat * beat_seconds + event.timing_offset_seconds)
        onset_sample = int(round(onset_seconds * sample_rate))
        hit_audio = load_hit(sample_path, sample_rate)
        prepared_hits.append((event, sample_path, onset_sample, hit_audio))

    rendered_hits = [
        (event, sample_path, onset_sample, apply_fades(hit_audio, sample_rate))
        for event, sample_path, onset_sample, hit_audio in prepared_hits
    ]

    tail_padding_samples = max(0, int(round(tail_padding_seconds * sample_rate)))
    end_sample = max(onset + hit.size for _, _, onset, hit in rendered_hits) + tail_padding_samples
    timeline = np.zeros(end_sample, dtype=np.float32)

    labels: list[dict] = []
    for event, sample_path, onset_sample, hit_audio in rendered_hits:
        end = onset_sample + hit_audio.size
        render_gain = event.gain * pre_mix_gain(event.sample_class)
        timeline[onset_sample:end] += hit_audio * render_gain

        onset_seconds = onset_sample / sample_rate
        labels.append(
            {
                "instrument": event.instrument,
                "class": event.sample_class,
                "timestamp": round(onset_seconds, 6),
                "onset_sample": onset_sample,
                "duration": round(hit_audio.size / sample_rate, 6),
                "gain": round(event.gain, 3),
                "render_gain": round(render_gain, 3),
                "timing_offset_seconds": round(event.timing_offset_seconds, 6),
                "source_wav": str(sample_path.relative_to(PROJECT_ROOT)),
            }
        )

    return RenderedPhrase(
        audio=peak_normalize(soft_limit_if_needed(timeline)),
        labels=labels,
        sample_rate=sample_rate,
        bpm=bpm,
        tail_padding_seconds=tail_padding_seconds,
    )


def export_phrase(
    rendered: RenderedPhrase,
    audio_path: Path,
    labels_path: Path,
    phrase_id: str | None = None,
    profile_name: str | None = None,
    generation_seed: int | None = None,
    wav_subtype: str = DEFAULT_WAV_SUBTYPE,
    peak_limit: float = DEFAULT_PEAK_LIMIT,
) -> dict:
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    written_audio = write_wav_pcm16(
        audio_path,
        rendered.audio,
        sr=rendered.sample_rate,
        peak_limit=peak_limit,
        subtype=wav_subtype,
    )

    phrase_duration_seconds = round(written_audio.size / rendered.sample_rate, 6)
    payload = {
        "phrase_id": phrase_id or labels_path.stem,
        "profile_name": profile_name or "unknown",
        "sample_rate": rendered.sample_rate,
        "bpm": rendered.bpm,
        "total_events": len(rendered.labels),
        "phrase_duration_seconds": phrase_duration_seconds,
        "contains_fill": detect_simple_fill(rendered.labels, rendered.bpm),
        "contains_overlap": detect_overlaps(rendered.labels),
        "generation_seed": generation_seed,
        "density_level": estimate_density(rendered.labels, rendered.bpm),
        "format": "mono_pcm16_wav",
        "wav_subtype": wav_subtype,
        "peak_limit": peak_limit,
        "peak": round(float(np.max(np.abs(written_audio))) if written_audio.size else 0.0, 6),
        "duration": phrase_duration_seconds,
        "tail_padding_seconds": round(rendered.tail_padding_seconds, 6),
        "events": rendered.labels,
    }
    labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def step_to_beat(step: int, grid_steps: int = GRID_STEPS, beats_per_phrase: int = BEATS_PER_PHRASE) -> float:
    return round(step * (beats_per_phrase / grid_steps), 6)


def choose_timing_offset(rng: random.Random, step: int) -> float:
    if step in (0, 4, 8, 12, 16):
        return rng.uniform(-0.0025, 0.0025)
    return rng.uniform(*MICROTIMING_SECONDS_RANGE)


def scaled_probability(probability: float, density: float) -> float:
    return min(1.0, max(0.0, probability * density))


def choose_profile_bpm(profile: GenerationProfile, rng: random.Random) -> float:
    low, high = profile.bpm_range
    if low <= 0 or high <= 0 or low > high:
        raise ValueError(f"Invalid BPM range for profile {profile.name!r}: {profile.bpm_range}")
    return rng.uniform(low, high)


def make_event(
    sample_class: str,
    step: int,
    gain: float = 1.0,
    timing_offset_seconds: float = 0.0,
) -> HitEvent:
    return HitEvent(
        instrument=sample_class,
        sample_class=sample_class,
        beat=step_to_beat(step),
        gain=round(gain, 3),
        timing_offset_seconds=round(timing_offset_seconds, 6),
    )


def add_event(
    events_by_step: dict[int, list[HitEvent]],
    step: int,
    sample_class: str,
    gain: float,
    timing_offset_seconds: float = 0.0,
) -> None:
    step_events = events_by_step.setdefault(step, [])
    if any(event.sample_class == sample_class for event in step_events):
        return
    step_events.append(make_event(sample_class, step, gain, timing_offset_seconds))


def choose_accent_gain(rng: random.Random) -> float:
    return rng.uniform(*ACCENT_GAIN_RANGE)


def choose_ghost_gain(rng: random.Random) -> float:
    return rng.uniform(*GHOST_NOTE_GAIN_RANGE)


def choose_tom_gain(rng: random.Random, fill_progress: float) -> float:
    low, high = TOM_FILL_GAIN_RANGE
    return min(1.0, rng.uniform(low, high) + fill_progress * 0.12)


def choose_fill_tom(
    rng: random.Random,
    progress: float,
    previous_tom_index: int,
) -> int:
    target_index = min(int(progress * len(FILL_TOM_PATH)), len(FILL_TOM_PATH) - 1)

    if target_index < previous_tom_index:
        return previous_tom_index

    if target_index > previous_tom_index:
        return min(previous_tom_index + 1, target_index)

    if previous_tom_index < len(FILL_TOM_PATH) - 1 and rng.random() < 0.18 + progress * 0.22:
        return previous_tom_index + 1

    return previous_tom_index


def flatten_events(events_by_step: dict[int, list[HitEvent]]) -> list[HitEvent]:
    return [
        event
        for step in sorted(events_by_step)
        for event in sorted(events_by_step[step], key=lambda item: INSTRUMENT_CLASSES.index(item.sample_class))
    ]


def humanize_events(
    events: list[HitEvent],
    rng: random.Random,
    timing_humanization: float = 1.0,
    dynamic_variation: float = 1.0,
) -> list[HitEvent]:
    humanized: list[HitEvent] = []
    timing_amount = min(1.0, max(0.0, timing_humanization))
    dynamic_amount = max(0.0, dynamic_variation)
    for event in events:
        gain_variation = rng.uniform(-0.035, 0.035) * dynamic_amount
        gain = min(1.0, max(0.16, event.gain + gain_variation))
        step = int(round(event.beat / (BEATS_PER_PHRASE / GRID_STEPS)))
        timing_offset_seconds = choose_timing_offset(rng, step) * timing_amount
        humanized.append(
            HitEvent(
                instrument=event.instrument,
                sample_class=event.sample_class,
                beat=event.beat,
                gain=round(gain, 3),
                timing_offset_seconds=round(timing_offset_seconds, 6),
            )
        )
    return humanized


def generate_groove(
    rng: random.Random,
    pattern: GroovePattern | None = None,
    rhythmic_density: float = 1.0,
) -> dict[int, list[HitEvent]]:
    pattern = pattern or rng.choice(GROOVE_PATTERNS)
    events_by_step: dict[int, list[HitEvent]] = {}

    for step in pattern.kick_steps:
        add_event(events_by_step, step, "KD", choose_accent_gain(rng))

    for step in pattern.snare_steps:
        add_event(events_by_step, step, "SD", choose_accent_gain(rng))

    for step in pattern.optional_kick_steps:
        if rng.random() < scaled_probability(0.42, rhythmic_density):
            add_event(events_by_step, step, "KD", rng.uniform(0.62, 0.9))

    for step in pattern.ghost_snare_steps:
        if rng.random() < scaled_probability(0.32, rhythmic_density):
            add_event(events_by_step, step, "SD", choose_ghost_gain(rng))

    # A drummer often leaves space; remove one non-essential weak event sometimes.
    removable_steps = [
        step
        for step, events in events_by_step.items()
        if step not in pattern.kick_steps and step not in pattern.snare_steps and len(events) == 1
    ]
    sparsity = max(0.0, 1.0 - rhythmic_density)
    if removable_steps and rng.random() < 0.12 + sparsity * 0.28:
        events_by_step.pop(rng.choice(removable_steps))

    return events_by_step


def apply_groove_variations(
    events_by_step: dict[int, list[HitEvent]],
    rng: random.Random,
    rhythmic_density: float = 1.0,
) -> None:
    if rng.random() < scaled_probability(0.3, rhythmic_density):
        add_event(events_by_step, rng.choice((2, 6, 10, 14)), "KD", rng.uniform(0.55, 0.78))

    if rng.random() < scaled_probability(0.22, rhythmic_density):
        add_event(events_by_step, rng.choice((7, 11, 15)), "SD", choose_ghost_gain(rng))

    if rng.random() < scaled_probability(0.18, rhythmic_density):
        add_event(events_by_step, 12, "KD", rng.uniform(0.45, 0.7))


def fill_start_step(rng: random.Random) -> int:
    if rng.random() < FINAL_TWO_BEAT_FILL_PROBABILITY:
        return rng.choice((8, 10))
    return rng.choice((12, 13))


def generate_fill(
    rng: random.Random,
    start_step: int | None = None,
    tom_probability: float = 1.0,
    rhythmic_density: float = 1.0,
) -> dict[int, list[HitEvent]]:
    start = start_step if start_step is not None else fill_start_step(rng)
    fill_steps = list(range(start, GRID_STEPS))
    events_by_step: dict[int, list[HitEvent]] = {}
    motif = rng.choice(FILL_MOTIFS)
    response_motif = tuple(offset for offset in motif if offset != 3 or rng.random() < 0.45)
    motif_length = 4
    previous_tom_index = 0
    last_fill_step = GRID_STEPS - 1

    for step in fill_steps:
        progress = (step - start) / max(GRID_STEPS - start - 1, 1)
        motif_position = (step - start) % motif_length
        cycle_index = (step - start) // motif_length
        active_motif = motif if cycle_index == 0 else response_motif
        repeated_motif_hit = motif_position in active_motif
        density_gate = 0.08 + progress * 0.28
        leave_space_before_resolution = step == last_fill_step and rng.random() < 0.68

        has_tom_hit = repeated_motif_hit or rng.random() < scaled_probability(density_gate, rhythmic_density)
        if not leave_space_before_resolution and has_tom_hit and rng.random() < tom_probability:
            tom_index = choose_fill_tom(rng, progress, previous_tom_index)
            previous_tom_index = tom_index
            sample_class = FILL_TOM_PATH[tom_index]
            add_event(events_by_step, step, sample_class, choose_tom_gain(rng, progress))

        if progress > 0.58 and rng.random() < scaled_probability(0.22, rhythmic_density):
            snare_gain = rng.uniform(0.42, 0.72) if step != last_fill_step else rng.uniform(0.58, 0.82)
            add_event(events_by_step, step, "SD", snare_gain)

    resolution = rng.choice(FILL_RESOLUTIONS)
    if resolution == "KD+SD":
        add_event(events_by_step, GRID_STEPS, "KD", choose_accent_gain(rng))
        add_event(events_by_step, GRID_STEPS, "SD", rng.uniform(0.68, 0.94))
    else:
        add_event(events_by_step, GRID_STEPS, resolution, choose_accent_gain(rng))

    if resolution == "KD" and rng.random() < 0.28:
        add_event(events_by_step, GRID_STEPS, "SD", rng.uniform(0.65, 0.92))

    return events_by_step


def apply_natural_overlaps(
    events_by_step: dict[int, list[HitEvent]],
    rng: random.Random,
    overlap_probability: float,
    rhythmic_density: float,
) -> None:
    if overlap_probability <= 0.0:
        return

    candidate_steps = sorted(step for step in events_by_step if 0 <= step <= GRID_STEPS)
    for step in candidate_steps:
        if rng.random() >= overlap_probability:
            continue

        sample_classes = {event.sample_class for event in events_by_step.get(step, [])}
        if "KD" in sample_classes and "SD" not in sample_classes:
            add_event(events_by_step, step, "SD", rng.uniform(0.42, 0.76))
        elif "SD" in sample_classes and "KD" not in sample_classes:
            add_event(events_by_step, step, "KD", rng.uniform(0.46, 0.82))
        elif any(sample_class.startswith("T") for sample_class in sample_classes) and rng.random() < rhythmic_density:
            add_event(events_by_step, step, rng.choice(("KD", "SD")), rng.uniform(0.38, 0.72))


def ensure_minimum_events(
    events_by_step: dict[int, list[HitEvent]],
    rng: random.Random,
    min_events: int,
    rhythmic_density: float,
    tom_probability: float,
) -> None:
    if sum(len(events) for events in events_by_step.values()) >= min_events:
        return

    candidate_steps = (2, 3, 6, 7, 10, 11, 14, 15)
    while sum(len(events) for events in events_by_step.values()) < min_events:
        open_steps = [
            step
            for step in candidate_steps
            if len(events_by_step.get(step, [])) < 2
        ]
        if not open_steps:
            break

        step = rng.choice(open_steps)
        if rng.random() < tom_probability * rhythmic_density and step >= 8:
            sample_class = rng.choice(FILL_TOM_PATH)
            gain = rng.uniform(0.48, 0.78)
        elif rng.random() < 0.55:
            sample_class = "KD"
            gain = rng.uniform(0.48, 0.78)
        else:
            sample_class = "SD"
            gain = choose_ghost_gain(rng)
        add_event(events_by_step, step, sample_class, gain)


def assemble_phrase(
    rng: random.Random | None = None,
    profile: GenerationProfile = STANDARD_ROCK,
) -> list[HitEvent]:
    if rng is None:
        rng = random.Random()

    events_by_step = generate_groove(rng, rhythmic_density=profile.rhythmic_density)
    apply_groove_variations(events_by_step, rng, rhythmic_density=profile.rhythmic_density)

    if rng.random() < profile.fill_probability:
        fill_events = generate_fill(
            rng,
            tom_probability=profile.tom_probability,
            rhythmic_density=profile.rhythmic_density,
        )
        fill_start = min(fill_events)

        # Let the fill take over the ending instead of stacking unrelated groove notes.
        for step in [step for step in events_by_step if step >= fill_start]:
            events_by_step.pop(step)

        for step, events in fill_events.items():
            for event in events:
                add_event(events_by_step, step, event.sample_class, event.gain)

    apply_natural_overlaps(
        events_by_step,
        rng,
        overlap_probability=profile.overlap_probability,
        rhythmic_density=profile.rhythmic_density,
    )
    ensure_minimum_events(
        events_by_step,
        rng,
        min_events=profile.min_events,
        rhythmic_density=profile.rhythmic_density,
        tom_probability=profile.tom_probability,
    )

    events = limit_phrase_density(
        flatten_events(events_by_step),
        min_events=profile.min_events,
        max_events=profile.max_events,
    )
    return humanize_events(
        events,
        rng,
        timing_humanization=profile.timing_humanization,
        dynamic_variation=profile.dynamic_variation,
    )


def limit_phrase_density(
    events: list[HitEvent],
    min_events: int = MIN_EVENTS_PER_PHRASE,
    max_events: int = MAX_EVENTS_PER_PHRASE,
) -> list[HitEvent]:
    if len(events) <= max_events:
        return events

    required = [
        event
        for event in events
        if (event.sample_class == "KD" and event.beat in (0.0, 2.0, 4.0))
        or (event.sample_class == "SD" and event.beat in (1.0, 3.0, 4.0))
    ]
    optional = [event for event in events if event not in required]

    def musical_priority(event: HitEvent) -> tuple[int, float, float]:
        is_fill_tom = event.sample_class.startswith("T") and event.beat >= 2.0
        is_resolution = event.beat >= 4.0 and event.sample_class in ("KD", "SD")
        is_ghost = event.sample_class == "SD" and event.gain <= 0.45
        priority = 0
        if is_resolution:
            priority += 5
        if is_fill_tom:
            priority += 4
        if event.sample_class in ("KD", "SD") and event.gain > 0.55:
            priority += 2
        if is_ghost:
            priority -= 2
        return (priority, event.beat, event.gain)

    optional = sorted(optional, key=musical_priority, reverse=True)
    kept = required + optional[: max(max_events - len(required), min_events - len(required))]
    return sorted(kept[:max_events], key=lambda event: (event.beat, INSTRUMENT_CLASSES.index(event.sample_class)))


def generate_random_phrase(
    profile: GenerationProfile = STANDARD_ROCK,
    rng: random.Random | None = None,
    grid_steps: int = GRID_STEPS,
    beats_per_phrase: int = BEATS_PER_PHRASE,
) -> list[HitEvent]:
    if grid_steps != GRID_STEPS or beats_per_phrase != BEATS_PER_PHRASE:
        raise ValueError("Pattern-based generation currently expects a 16-step, 4-beat phrase.")
    events = assemble_phrase(rng, profile=profile)
    return limit_phrase_density(events, profile.min_events, profile.max_events)


def generate_training_dataset(
    phrase_count: int = DEFAULT_PHRASE_COUNT,
    bpm: float | None = None,
    profile: GenerationProfile = STANDARD_ROCK,
    samples_dir: Path = SAMPLES_DIR,
    output_audio_dir: Path = OUTPUT_AUDIO_DIR,
    output_labels_dir: Path = OUTPUT_LABELS_DIR,
    sample_rate: int = TARGET_SR,
    wav_subtype: str = DEFAULT_WAV_SUBTYPE,
    peak_limit: float = DEFAULT_PEAK_LIMIT,
    seed: int = 7,
) -> None:
    rng = random.Random(seed)

    for index in range(1, phrase_count + 1):
        phrase_id = make_phrase_id(profile.name, index)
        phrase_seed = seed + index
        phrase_bpm = bpm if bpm is not None else choose_profile_bpm(profile, rng)
        events = generate_random_phrase(profile, rng)
        rendered = render_phrase(
            events,
            bpm=phrase_bpm,
            samples_dir=samples_dir,
            sample_rate=sample_rate,
            seed=phrase_seed,
        )

        stem = f"{phrase_id}_{int(round(phrase_bpm))}bpm"
        export_phrase(
            rendered,
            output_audio_dir / f"{stem}.wav",
            output_labels_dir / f"{stem}.json",
            phrase_id=phrase_id,
            profile_name=profile.name,
            generation_seed=phrase_seed,
            wav_subtype=wav_subtype,
            peak_limit=peak_limit,
        )


def build_target_shares(
    target_tom_share: float,
    balance_individual_toms: bool,
) -> dict[str, float]:
    if not 0.0 < target_tom_share < 1.0:
        raise ValueError("target_tom_share must be between 0 and 1.")

    groove_share = (1.0 - target_tom_share) / 2.0
    tom_weights = (1 / 3, 1 / 3, 1 / 3) if balance_individual_toms else (0.5, 0.3, 0.2)
    shares = {"KD": groove_share, "SD": groove_share}
    shares.update(
        {
            class_name: target_tom_share * weight
            for class_name, weight in zip(TOM_CLASSES, tom_weights)
        }
    )
    return shares


def allocate_class_quotas(
    event_count: int,
    cumulative_counts: Counter[str],
    target_shares: dict[str, float],
) -> Counter[str]:
    if event_count < len(CLASS_NAMES):
        raise ValueError(f"event_count must be at least {len(CLASS_NAMES)} to include all classes.")

    quotas = Counter({class_name: 1 for class_name in CLASS_NAMES})
    future_total = sum(cumulative_counts.values()) + event_count
    for _ in range(event_count - len(CLASS_NAMES)):
        class_name = max(
            CLASS_NAMES,
            key=lambda name: (
                target_shares[name] * future_total - cumulative_counts[name] - quotas[name],
                -CLASS_NAMES.index(name),
            ),
        )
        quotas[class_name] += 1
    return quotas


def event_step(event: HitEvent) -> int:
    return int(round(event.beat / (BEATS_PER_PHRASE / GRID_STEPS)))


def select_existing_events(
    base_events: Iterable[HitEvent],
    class_name: str,
    count: int,
) -> list[HitEvent]:
    candidates = [event for event in base_events if event.sample_class == class_name]
    if class_name in TOM_CLASSES:
        candidates.sort(key=lambda event: (event.beat < 2.0, -event.beat, -event.gain))
    else:
        candidates.sort(
            key=lambda event: (
                event_step(event) not in (0, 4, 8, 12),
                -event.gain,
                event.beat,
            )
        )
    return [make_event(class_name, event_step(event), gain=event.gain) for event in candidates[:count]]


def choose_missing_step(
    class_name: str,
    selected: list[HitEvent],
    rng: random.Random,
    tom_boost: float,
) -> int:
    used_for_class = {
        event_step(event)
        for event in selected
        if event.sample_class == class_name
    }

    if class_name == "KD":
        candidates = list(KD_STEPS)
    elif class_name == "SD":
        candidates = list(SD_STEPS)
    else:
        candidates = list(TOM_STEPS[class_name])
        overlap_steps = [
            event_step(event)
            for event in selected
            if event.sample_class in ("KD", "SD")
            and event_step(event) >= 8
            and event_step(event) not in used_for_class
        ]
        overlap_probability = min(0.55, 0.18 * max(1.0, tom_boost))
        if overlap_steps and rng.random() < overlap_probability:
            return rng.choice(overlap_steps)

    available = [step for step in candidates if step not in used_for_class]
    if available:
        return rng.choice(available[: min(4, len(available))])

    fallback = [step for step in range(GRID_STEPS) if step not in used_for_class]
    if not fallback:
        raise RuntimeError(f"No free rhythmic step remains for {class_name}.")
    return rng.choice(fallback)


def event_gain(class_name: str, rng: random.Random) -> float:
    if class_name == "KD":
        return rng.uniform(0.68, 0.98)
    if class_name == "SD":
        return rng.uniform(0.55, 0.98)
    return rng.uniform(0.62, 0.94)


def build_balanced_phrase(
    base_events: list[HitEvent],
    quotas: Counter[str],
    rng: random.Random,
    profile: GenerationProfile,
    tom_boost: float,
) -> list[HitEvent]:
    selected: list[HitEvent] = []
    for class_name in CLASS_NAMES:
        selected.extend(select_existing_events(base_events, class_name, quotas[class_name]))

    for class_name in CLASS_NAMES:
        missing = quotas[class_name] - sum(event.sample_class == class_name for event in selected)
        for _ in range(missing):
            step = choose_missing_step(class_name, selected, rng, tom_boost)
            selected.append(make_event(class_name, step, event_gain(class_name, rng)))

    selected.sort(key=lambda event: (event.beat, CLASS_NAMES.index(event.sample_class)))
    return humanize_events(
        selected,
        rng,
        timing_humanization=profile.timing_humanization,
        dynamic_variation=profile.dynamic_variation,
    )


def boosted_profile(profile: GenerationProfile, tom_boost: float) -> GenerationProfile:
    if tom_boost <= 0:
        raise ValueError("tom_boost must be greater than zero.")
    return replace(
        profile,
        fill_probability=min(1.0, profile.fill_probability * tom_boost),
        tom_probability=min(1.0, profile.tom_probability * tom_boost),
        overlap_probability=min(0.6, profile.overlap_probability * tom_boost),
        min_events=max(TOMBOOST_MIN_EVENTS, profile.min_events),
        max_events=max(12, profile.max_events),
    )


def normalize_profile_weights(profile_names: Sequence[str], weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / len(profile_names)] * len(profile_names)
    if len(weights) != len(profile_names):
        raise ValueError("--profile-weights must have the same length as --profiles.")
    if any(weight < 0 for weight in weights):
        raise ValueError("--profile-weights cannot contain negative values.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("--profile-weights must sum to a positive value.")
    return [weight / total for weight in weights]


def choose_weighted_profile(
    rng: random.Random,
    profiles: Sequence[GenerationProfile],
    weights: Sequence[float],
) -> GenerationProfile:
    return rng.choices(list(profiles), weights=list(weights), k=1)[0]


def write_payload_annotation(annotation_path: Path, payload: dict) -> int:
    rows = extract_adtof_rows(payload, annotation_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    write_annotation(annotation_path, rows)
    return len(rows)


def ensure_empty_output(output_root: Path) -> None:
    existing_files = [
        path
        for split_name in SPLITS
        for folder_name in ("AUDIO", "LABELS", "ANNOTATIONS")
        for path in (output_root / split_name / folder_name).glob("*")
        if path.is_file()
    ]
    if existing_files:
        raise FileExistsError(
            f"Output dataset already contains files: {output_root}. Choose a new --output-root."
        )


def write_event_counts(
    output_root: Path,
    counts_by_split: dict[str, Counter[str]],
    target_shares: dict[str, float],
) -> None:
    rows = []
    for split_name in SPLITS:
        split_counts = counts_by_split[split_name]
        split_total = sum(split_counts.values())
        for class_name in CLASS_NAMES:
            count = split_counts[class_name]
            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "midi": CLASS_TO_MIDI[class_name],
                    "event_count": count,
                    "percentage": round(100 * count / split_total, 4) if split_total else 0.0,
                }
            )

    csv_path = output_root / "dataset_event_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "class_schema": "vibro_5_toms",
        "class_names": list(CLASS_NAMES),
        "labels": [CLASS_TO_MIDI[name] for name in CLASS_NAMES],
        "target_shares": target_shares,
        "counts": rows,
    }
    (output_root / "dataset_event_counts.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def generate_dataset_split(
    split_name: str,
    phrase_count: int,
    output_root: Path,
    samples_dir: Path,
    profiles: Sequence[GenerationProfile],
    profile_weights: Sequence[float],
    bpm: float | None,
    target_shares: dict[str, float],
    tom_boost: float,
    sample_rate: int,
    wav_subtype: str,
    peak_limit: float,
    write_annotations: bool,
    seed: int,
) -> Counter[str]:
    split_root = output_root / split_name
    audio_dir = split_root / "AUDIO"
    labels_dir = split_root / "LABELS"
    annotations_dir = split_root / "ANNOTATIONS"
    for directory in (audio_dir, labels_dir, annotations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    cumulative_counts: Counter[str] = Counter()
    progress_interval = max(1, phrase_count // 10)
    for index in range(1, phrase_count + 1):
        profile = choose_weighted_profile(rng, profiles, profile_weights)
        event_count = rng.randint(profile.min_events, profile.max_events)
        quotas = allocate_class_quotas(event_count, cumulative_counts, target_shares)
        base_events = generate_random_phrase(profile, rng)
        events = build_balanced_phrase(base_events, quotas, rng, profile, tom_boost)
        phrase_bpm = bpm if bpm is not None else choose_profile_bpm(profile, rng)
        phrase_seed = seed + index
        rendered = render_phrase(
            events,
            bpm=phrase_bpm,
            samples_dir=samples_dir,
            sample_rate=sample_rate,
            seed=phrase_seed,
        )

        phrase_id = f"{split_name.lower()}_{profile.name}_{index:06d}"
        stem = f"{phrase_id}_{int(round(phrase_bpm))}bpm"
        labels_path = labels_dir / f"{stem}.json"
        payload = export_phrase(
            rendered,
            audio_dir / f"{stem}.wav",
            labels_path,
            phrase_id=phrase_id,
            profile_name=profile.name,
            generation_seed=phrase_seed,
            wav_subtype=wav_subtype,
            peak_limit=peak_limit,
        )
        if write_annotations:
            write_payload_annotation(annotations_dir / f"{stem}.txt", payload)
        cumulative_counts.update(event.sample_class for event in events)

        if index % progress_interval == 0 or index == phrase_count:
            print(f"{split_name}: {index}/{phrase_count} phrases")

    return cumulative_counts


def generate_split_dataset(args: argparse.Namespace) -> dict[str, Counter[str]]:
    if args.target_sr != TARGET_SR:
        raise ValueError(f"This dataset must be generated at {TARGET_SR} Hz, got {args.target_sr}.")
    if args.wav_subtype != DEFAULT_WAV_SUBTYPE:
        raise ValueError(f"This dataset must use WAV subtype {DEFAULT_WAV_SUBTYPE}.")
    if any(count < 0 for count in (args.num_train, args.num_val, args.num_test)):
        raise ValueError("Split sizes cannot be negative.")
    if not any((args.num_train, args.num_val, args.num_test)):
        raise ValueError("At least one split must contain phrases.")

    output_root = args.output_root.resolve()
    ensure_empty_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profile_names = list(args.profiles or [args.profile])
    unknown_profiles = [name for name in profile_names if name not in GENERATION_PROFILES]
    if unknown_profiles:
        raise ValueError(f"Unknown profiles: {', '.join(unknown_profiles)}")
    profile_weights = normalize_profile_weights(profile_names, args.profile_weights)
    profiles = [
        boosted_profile(GENERATION_PROFILES[name], args.tom_boost)
        for name in profile_names
    ]
    target_shares = build_target_shares(args.target_tom_share, args.balance_individual_toms)
    split_sizes = {
        "TRAIN": args.num_train,
        "VAL": args.num_val,
        "TEST": args.num_test,
    }

    metadata = {
        "class_schema": "vibro_5_toms",
        "class_names": list(CLASS_NAMES),
        "labels": [CLASS_TO_MIDI[name] for name in CLASS_NAMES],
        "split_sizes": split_sizes,
        "profiles": profile_names,
        "profile_weights": profile_weights,
        "tom_boost": args.tom_boost,
        "balance_individual_toms": args.balance_individual_toms,
        "target_tom_share": args.target_tom_share,
        "target_shares": target_shares,
        "sample_rate": TARGET_SR,
        "channels": 1,
        "wav_subtype": DEFAULT_WAV_SUBTYPE,
        "peak_limit": args.peak_limit,
        "write_annotations": args.write_annotations,
        "seed": args.seed,
    }
    (output_root / "dataset_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    counts_by_split: dict[str, Counter[str]] = {}
    for split_index, split_name in enumerate(SPLITS):
        counts_by_split[split_name] = generate_dataset_split(
            split_name=split_name,
            phrase_count=split_sizes[split_name],
            output_root=output_root,
            samples_dir=args.samples_dir.resolve(),
            profiles=profiles,
            profile_weights=profile_weights,
            bpm=args.bpm,
            target_shares=target_shares,
            tom_boost=args.tom_boost,
            sample_rate=args.target_sr,
            wav_subtype=args.wav_subtype,
            peak_limit=args.peak_limit,
            write_annotations=args.write_annotations,
            seed=args.seed + split_index * 1_000_000,
        )

    write_event_counts(output_root, counts_by_split, target_shares)
    print(f"Dataset written to: {output_root}")
    return counts_by_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate procedural ADTOF vibration drum datasets.")
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--count", type=int, default=None, help="Legacy alias for --num-train.")
    parser.add_argument("--profile", choices=sorted(GENERATION_PROFILES), default=STANDARD_ROCK.name)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output dataset root.",
    )
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=0)
    parser.add_argument("--num-test", type=int, default=0)
    parser.add_argument("--profiles", nargs="+", choices=sorted(GENERATION_PROFILES), default=None)
    parser.add_argument("--profile-weights", nargs="+", type=float, default=None)
    parser.add_argument(
        "--tom-boost",
        type=float,
        nargs="?",
        const=DEFAULT_TOM_BOOST,
        default=DEFAULT_TOM_BOOST,
        help="Multiplier applied to fill, tom and natural-overlap probabilities.",
    )
    parser.add_argument("--target-tom-share", type=float, default=DEFAULT_TARGET_TOM_SHARE)
    parser.add_argument(
        "--balance-individual-toms",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    parser.add_argument("--wav-subtype", default=DEFAULT_WAV_SUBTYPE)
    parser.add_argument("--peak-limit", type=float, default=DEFAULT_PEAK_LIMIT)
    parser.add_argument("--write-annotations", action="store_true")
    args = parser.parse_args()
    if args.num_train is None:
        args.num_train = args.count if args.count is not None else DEFAULT_PHRASE_COUNT
    return args


def main() -> None:
    args = parse_args()
    generate_split_dataset(args)


if __name__ == "__main__":
    main()
