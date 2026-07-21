# Error analysis: serious_5class_low_60ep_focal_soft_toms_v3

- split: `TEST`
- checkpoint: `final`
- threshold source: `fulltrack_inference_config`
- inference_config: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\RUNS\serious_5class_low_60ep_focal_soft_toms_v3\fulltrack_calibration\fulltrack_selected_inference_config.json`
- threshold modes: `global`
- global threshold: `0.04`
- per-class thresholds: `{'KD': 0.04, 'SD': 0.04, 'T12': 0.04, 'T14': 0.04, 'T16': 0.04}`
- nms_ms: `150.0`

## Reconstructed Metrics
- `global`: micro-P=0.5366, micro-R=0.5574, micro-F1=0.5468, macro-F1=0.5408, TP=7788, FP=6726, FN=6184

## Ground-truth count check
- status: `match`
- counts_from_annotations: `{'KD': 4891, 'SD': 4890, 'T12': 1397, 'T14': 1397, 'T16': 1397}`
- counts_used_by_analyzer_by_mode: `{'global': {'KD': 4891, 'SD': 4890, 'T12': 1397, 'T14': 1397, 'T16': 1397}}`
- counts_from_dataset_event_counts_csv: `{'KD': 4891, 'SD': 4890, 'T12': 1397, 'T14': 1397, 'T16': 1397}`
- delta_analyzer_vs_annotations_by_mode: `{'global': {'KD': 0, 'SD': 0, 'T12': 0, 'T14': 0, 'T16': 0}}`

## Track coverage check
- status: `match`
- n_annotation_tracks: `1200`
- n_unique_tracks_used_by_analyzer: `1200`
- n_total_track_iterations_by_analyzer: `1200`
- n_total_track_iterations_by_analyzer_by_mode: `{'global': 1200}`
- duplicated_track_ids: `[]`
- missing_track_ids_count: `0`
- extra_track_ids: `[]`
- tracks_with_gt_mismatch: `[]`

## By Class
| threshold_mode | class_name | threshold | TP | FP | FN | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | KD | 0.0400 | 3210 | 3146 | 1681 | 0.5050 | 0.6563 | 0.5708 |
| global | SD | 0.0400 | 2399 | 1779 | 2491 | 0.5742 | 0.4906 | 0.5291 |
| global | T12 | 0.0400 | 797 | 876 | 600 | 0.4764 | 0.5705 | 0.5192 |
| global | T14 | 0.0400 | 648 | 541 | 749 | 0.5450 | 0.4639 | 0.5012 |
| global | T16 | 0.0400 | 734 | 384 | 663 | 0.6565 | 0.5254 | 0.5837 |

## Worst Tracks By F1
| threshold_mode | track_id | profile | TP | FP | FN | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | test_standard_rock_000680_129bpm | standard_rock | 1 | 9 | 9 | 0.1000 | 0.1000 | 0.1000 |
| global | test_standard_rock_000126_122bpm | standard_rock | 1 | 6 | 10 | 0.1429 | 0.0909 | 0.1111 |
| global | test_basic_groove_001134_104bpm | basic_groove | 2 | 8 | 10 | 0.2000 | 0.1667 | 0.1818 |
| global | test_standard_rock_001090_109bpm | standard_rock | 2 | 6 | 10 | 0.2500 | 0.1667 | 0.2000 |
| global | test_basic_groove_001014_104bpm | basic_groove | 2 | 8 | 8 | 0.2000 | 0.2000 | 0.2000 |
| global | test_standard_rock_000368_123bpm | standard_rock | 2 | 7 | 8 | 0.2222 | 0.2000 | 0.2105 |
| global | test_standard_rock_000601_139bpm | standard_rock | 2 | 7 | 8 | 0.2222 | 0.2000 | 0.2105 |
| global | test_standard_rock_000514_105bpm | standard_rock | 2 | 5 | 9 | 0.2857 | 0.1818 | 0.2222 |
| global | test_fast_dense_000643_186bpm | fast_dense | 3 | 10 | 11 | 0.2308 | 0.2143 | 0.2222 |
| global | test_standard_rock_001050_130bpm | standard_rock | 3 | 12 | 9 | 0.2000 | 0.2500 | 0.2222 |

## Worst Tracks By False Positives
| threshold_mode | track_id | profile | TP | FP | FN | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | test_fill_heavy_000787_104bpm | fill_heavy | 6 | 16 | 9 | 0.2727 | 0.4000 | 0.3243 |
| global | test_fill_heavy_000426_109bpm | fill_heavy | 4 | 14 | 8 | 0.2222 | 0.3333 | 0.2667 |
| global | test_basic_groove_001153_104bpm | basic_groove | 4 | 14 | 7 | 0.2222 | 0.3636 | 0.2759 |
| global | test_fill_heavy_000995_131bpm | fill_heavy | 9 | 14 | 6 | 0.3913 | 0.6000 | 0.4737 |
| global | test_humanized_mixed_000008_148bpm | humanized_mixed | 5 | 13 | 9 | 0.2778 | 0.3571 | 0.3125 |
| global | test_fill_heavy_000233_100bpm | fill_heavy | 6 | 13 | 9 | 0.3158 | 0.4000 | 0.3529 |
| global | test_basic_groove_000658_100bpm | basic_groove | 6 | 13 | 6 | 0.3158 | 0.5000 | 0.3871 |
| global | test_standard_rock_001050_130bpm | standard_rock | 3 | 12 | 9 | 0.2000 | 0.2500 | 0.2222 |
| global | test_standard_rock_001126_105bpm | standard_rock | 4 | 12 | 8 | 0.2500 | 0.3333 | 0.2857 |
| global | test_fill_heavy_000622_105bpm | fill_heavy | 6 | 12 | 9 | 0.3333 | 0.4000 | 0.3636 |

## Worst Tracks By False Negatives
| threshold_mode | track_id | profile | TP | FP | FN | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | test_fast_dense_000643_186bpm | fast_dense | 3 | 10 | 11 | 0.2308 | 0.2143 | 0.2222 |
| global | test_fast_dense_000894_182bpm | fast_dense | 3 | 8 | 11 | 0.2727 | 0.2143 | 0.2400 |
| global | test_fast_dense_001034_155bpm | fast_dense | 4 | 9 | 11 | 0.3077 | 0.2667 | 0.2857 |
| global | test_fast_dense_000238_186bpm | fast_dense | 5 | 9 | 11 | 0.3571 | 0.3125 | 0.3333 |
| global | test_standard_rock_000126_122bpm | standard_rock | 1 | 6 | 10 | 0.1429 | 0.0909 | 0.1111 |
| global | test_basic_groove_001134_104bpm | basic_groove | 2 | 8 | 10 | 0.2000 | 0.1667 | 0.1818 |
| global | test_standard_rock_001090_109bpm | standard_rock | 2 | 6 | 10 | 0.2500 | 0.1667 | 0.2000 |
| global | test_fast_dense_000563_154bpm | fast_dense | 4 | 9 | 10 | 0.3077 | 0.2857 | 0.2963 |
| global | test_fast_dense_000813_161bpm | fast_dense | 4 | 9 | 10 | 0.3077 | 0.2857 | 0.2963 |
| global | test_fill_heavy_000901_98bpm | fill_heavy | 4 | 9 | 10 | 0.3077 | 0.2857 | 0.2963 |

## Problematic Profiles
| threshold_mode | profile | class_name | TP | FP | FN | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | basic_groove | T14 | 147 | 129 | 217 | 0.5326 | 0.4038 | 0.4594 |
| global | fill_heavy | T12 | 122 | 158 | 108 | 0.4357 | 0.5304 | 0.4784 |
| global | fast_dense | T14 | 68 | 63 | 74 | 0.5191 | 0.4789 | 0.4982 |
| global | standard_rock | T14 | 199 | 167 | 225 | 0.5437 | 0.4693 | 0.5038 |
| global | fast_dense | T12 | 83 | 95 | 67 | 0.4663 | 0.5533 | 0.5061 |
| global | basic_groove | SD | 609 | 461 | 690 | 0.5692 | 0.4688 | 0.5141 |
| global | fill_heavy | T14 | 114 | 91 | 121 | 0.5561 | 0.4851 | 0.5182 |
| global | humanized_mixed | T12 | 128 | 140 | 93 | 0.4776 | 0.5792 | 0.5235 |
| global | standard_rock | SD | 717 | 538 | 763 | 0.5713 | 0.4845 | 0.5243 |
| global | standard_rock | T12 | 253 | 275 | 183 | 0.4792 | 0.5803 | 0.5249 |
| global | fill_heavy | SD | 425 | 326 | 415 | 0.5659 | 0.5060 | 0.5343 |
| global | humanized_mixed | SD | 390 | 299 | 378 | 0.5660 | 0.5078 | 0.5353 |
| global | fill_heavy | KD | 509 | 548 | 318 | 0.4816 | 0.6155 | 0.5403 |
| global | basic_groove | T12 | 211 | 208 | 149 | 0.5036 | 0.5861 | 0.5417 |
| global | humanized_mixed | T14 | 120 | 91 | 112 | 0.5687 | 0.5172 | 0.5418 |

## Consistency Check
- `global`: official_file_missing
