# Current Official Baseline

Updated: 2026-06-29

## Official Run

- Current official baseline: `RUNS/serious_5class_low_60ep_weighted_t14t16`
- Previous official baseline: `RUNS/serious_5class_low_60ep_calibrated`
- Source inference config: `RUNS/serious_5class_low_60ep_weighted_t14t16/fulltrack_calibration_f1_nms/fulltrack_selected_inference_config.json`
- Checkpoint: `best`
- Checkpoint path: `RUNS/serious_5class_low_60ep_weighted_t14t16/checkpoints/best_weights.weights.h5`
- Threshold mode: `global`
- Threshold global: `0.030`
- NMS: `110 ms`
- Onset tolerance: `50 ms`
- Selection split: `VAL`
- Final report split: `TEST`
- TEST not used for selection: `true`

## Class Schema

Use the current vibro 5-class tom schema:

| Class | MIDI |
| --- | ---: |
| KD | 35 |
| SD | 38 |
| T12 | 47 |
| T14 | 45 |
| T16 | 43 |

Do not use the legacy schema `KD, SD, TT, HH, CY`.

## TEST Full-Track Metrics

| Metric | Previous baseline | Current baseline | Delta |
| --- | ---: | ---: | ---: |
| micro-F1 | 0.5716 | 0.5863 | +0.0147 |
| micro-recall | 0.5857 | 0.6196 | +0.0339 |
| macro-F1 | 0.5365 | 0.5855 | +0.0490 |
| T14 F1 | 0.4960 | 0.5425 | +0.0465 |
| T16 F1 | 0.4317 | 0.6349 | +0.2032 |
| SD F1 | 0.5917 | 0.5733 | -0.0184 |

The weighted run is now the official baseline because it improves the primary TEST full-track onset micro-F1 and substantially improves macro-F1. The largest intended gains are on the weak tom classes T14 and T16, especially T16.

## Current Baseline TEST Metrics

| Metric | Value |
| --- | ---: |
| TP | 8657 |
| FP | 6901 |
| FN | 5315 |
| micro-precision | 0.5564 |
| micro-recall | 0.6196 |
| micro-F1 | 0.5863 |
| macro-F1 | 0.5855 |

## Current Baseline By Class

| Class | MIDI | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KD | 35 | 3640 | 3698 | 1251 | 0.4960 | 0.7442 | 0.5953 |
| SD | 38 | 2533 | 1413 | 2357 | 0.6419 | 0.5180 | 0.5733 |
| T12 | 47 | 820 | 604 | 577 | 0.5758 | 0.5870 | 0.5814 |
| T14 | 45 | 750 | 618 | 647 | 0.5482 | 0.5369 | 0.5425 |
| T16 | 43 | 914 | 568 | 483 | 0.6167 | 0.6543 | 0.6349 |

## Interpretation

The new baseline trades a small decrease in precision and SD F1 for a clear recall gain and a much better class balance. This is consistent with the goal of the class-weighted run: recover more T14/T16 events without changing the dataset, generator, DataLoader, or training protocol.

For WAV inference, use the selected `best` checkpoint with global threshold `0.030` and NMS `110 ms` under the `KD, SD, T12, T14, T16` schema.
