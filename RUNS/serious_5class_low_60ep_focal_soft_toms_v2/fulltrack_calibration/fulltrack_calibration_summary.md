# Full-track calibration: serious_5class_low_60ep_focal_soft_toms_v2

- dataset: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_original_from_run_config: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_resolved: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_source: `cli_override`
- selection_split: `VAL`
- final_report_split: `TEST`
- test_not_used_for_selection: `true`
- threshold_mode_requested: `global`
- selection_metric: `f1`
- beta: `1.5`
- selection_rule: `maximize selected metric -> maximize precision -> minimize FP -> choose highest threshold`
- selected_inference_config_alias: `global`
- previous_eval_artifacts: `not_available`

## Best Global From VAL
- checkpoint: `best`
- threshold: `0.05`
- onset_nms_ms: `110.0`
- VAL: `{'TP': 9005, 'FP': 6065, 'FN': 4911, 'micro_precision': 0.5975447909754479, 'micro_recall': 0.6470968669157804, 'micro_f1': 0.6213344373145656, 'micro_fbeta': 0.6309965287509972, 'macro_f1': 0.5956706154190894, 'macro_fbeta': 0.5858672344847682}`
- TEST frozen from VAL: `{'TP': 8985, 'FP': 6134, 'FN': 4987, 'micro_precision': 0.5942853363317679, 'micro_recall': 0.6430718580017177, 'micro_f1': 0.6177168196349387, 'micro_fbeta': 0.6272284990119426, 'macro_f1': 0.589873833503554, 'macro_fbeta': 0.5797787366088839}`

## Best Per-Class From VAL
- checkpoint: `best`
- thresholds_per_class: `{'KD': 0.075, 'SD': 0.05, 'T12': 0.04, 'T14': 0.04, 'T16': 0.03}`
- onset_nms_ms: `110.0`
- VAL: `{'TP': 9156, 'FP': 5282, 'FN': 4760, 'micro_precision': 0.6341598559357252, 'micro_recall': 0.6579476861167002, 'micro_f1': 0.6458348028496862, 'micro_fbeta': 0.6504404467857221, 'macro_f1': 0.630569575658867, 'macro_fbeta': 0.6301734742637559}`
- TEST frozen from VAL: `{'TP': 9149, 'FP': 5277, 'FN': 4823, 'micro_precision': 0.6342021350339665, 'micro_recall': 0.6548096192384769, 'micro_f1': 0.6443411507852664, 'micro_fbeta': 0.6483276279353728, 'macro_f1': 0.6270768985362782, 'macro_fbeta': 0.6254879260637394}`

## TEST By Class: Global
| class | midi_pitch | threshold | onset_nms_ms | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 35 | 0.0500 | 110.0000 | 3819 | 3192 | 1072 | 0.5447 | 0.7808 | 0.6417 | 0.6889 |
| SD | 38 | 0.0500 | 110.0000 | 3166 | 1957 | 1724 | 0.6180 | 0.6474 | 0.6324 | 0.6381 |
| T12 | 47 | 0.0500 | 110.0000 | 640 | 354 | 757 | 0.6439 | 0.4581 | 0.5353 | 0.5027 |
| T14 | 45 | 0.0500 | 110.0000 | 667 | 431 | 730 | 0.6075 | 0.4775 | 0.5347 | 0.5111 |
| T16 | 43 | 0.0500 | 110.0000 | 693 | 200 | 704 | 0.7760 | 0.4961 | 0.6052 | 0.5580 |

## TEST By Class: Per-Class
| class | midi_pitch | threshold | onset_nms_ms | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 35 | 0.0750 | 110.0000 | 3543 | 1869 | 1348 | 0.6547 | 0.7244 | 0.6878 | 0.7014 |
| SD | 38 | 0.0500 | 110.0000 | 3166 | 1957 | 1724 | 0.6180 | 0.6474 | 0.6324 | 0.6381 |
| T12 | 47 | 0.0400 | 110.0000 | 763 | 521 | 634 | 0.5942 | 0.5462 | 0.5692 | 0.5601 |
| T14 | 45 | 0.0400 | 110.0000 | 773 | 612 | 624 | 0.5581 | 0.5533 | 0.5557 | 0.5548 |
| T16 | 43 | 0.0300 | 110.0000 | 904 | 318 | 493 | 0.7398 | 0.6471 | 0.6903 | 0.6730 |

## TEST Delta Per-Class Minus Global
| class | midi_pitch | global_threshold | per_class_threshold | global_f1 | per_class_f1 | delta_f1 | global_FP | per_class_FP | global_FN | per_class_FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 35 | 0.0500 | 0.0750 | 0.6417 | 0.6878 | 0.0460 | 3192 | 1869 | 1072 | 1348 |
| SD | 38 | 0.0500 | 0.0500 | 0.6324 | 0.6324 | 0.0000 | 1957 | 1957 | 1724 | 1724 |
| T12 | 47 | 0.0500 | 0.0400 | 0.5353 | 0.5692 | 0.0338 | 354 | 521 | 757 | 634 |
| T14 | 45 | 0.0500 | 0.0400 | 0.5347 | 0.5557 | 0.0210 | 431 | 612 | 730 | 624 |
| T16 | 43 | 0.0500 | 0.0300 | 0.6052 | 0.6903 | 0.0851 | 200 | 318 | 704 | 493 |

## Toms
- toms_with_higher_TEST_f1_under_per_class: `['T12', 'T14', 'T16']`
- TEST is shown only as a frozen report after VAL selection; it did not choose the mode, checkpoint, thresholds, or NMS.