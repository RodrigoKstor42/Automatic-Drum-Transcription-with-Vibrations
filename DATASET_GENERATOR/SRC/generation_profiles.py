from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    bpm_range: tuple[float, float]
    min_events: int
    max_events: int
    fill_probability: float
    tom_probability: float
    rhythmic_density: float
    overlap_probability: float
    timing_humanization: float
    dynamic_variation: float


BASIC_GROOVE = GenerationProfile(
    name="basic_groove",
    bpm_range=(90.0, 115.0),
    min_events=4,
    max_events=8,
    fill_probability=0.15,
    tom_probability=0.18,
    rhythmic_density=0.35,
    overlap_probability=0.08,
    timing_humanization=0.35,
    dynamic_variation=0.35,
)

STANDARD_ROCK = GenerationProfile(
    name="standard_rock",
    bpm_range=(100.0, 145.0),
    min_events=6,
    max_events=12,
    fill_probability=0.55,
    tom_probability=0.55,
    rhythmic_density=0.58,
    overlap_probability=0.18,
    timing_humanization=0.55,
    dynamic_variation=0.55,
)

FILL_HEAVY = GenerationProfile(
    name="fill_heavy",
    bpm_range=(95.0, 135.0),
    min_events=8,
    max_events=15,
    fill_probability=0.88,
    tom_probability=0.88,
    rhythmic_density=0.72,
    overlap_probability=0.25,
    timing_humanization=0.6,
    dynamic_variation=0.68,
)

FAST_DENSE = GenerationProfile(
    name="fast_dense",
    bpm_range=(145.0, 190.0),
    min_events=9,
    max_events=16,
    fill_probability=0.62,
    tom_probability=0.62,
    rhythmic_density=0.88,
    overlap_probability=0.28,
    timing_humanization=0.42,
    dynamic_variation=0.5,
)

HUMANIZED_MIXED = GenerationProfile(
    name="humanized_mixed",
    bpm_range=(80.0, 155.0),
    min_events=5,
    max_events=14,
    fill_probability=0.68,
    tom_probability=0.7,
    rhythmic_density=0.64,
    overlap_probability=0.22,
    timing_humanization=0.9,
    dynamic_variation=0.85,
)


GENERATION_PROFILES = {
    profile.name: profile
    for profile in (
        BASIC_GROOVE,
        STANDARD_ROCK,
        FILL_HEAVY,
        FAST_DENSE,
        HUMANIZED_MIXED,
    )
}

