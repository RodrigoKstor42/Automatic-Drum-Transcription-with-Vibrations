# Full-track calibration: serious_5class_low_60ep_weighted_t14t16

- dataset: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_original_from_run_config: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_resolved: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_source: `cli_override`
- selection_split: `VAL`
- final_report_split: `TEST`
- selection_metric: `f1`
- beta: `1.5`
- selection_rule: `maximize selected metric -> maximize precision -> minimize FP -> choose highest threshold`
- WARNING: TEST was not used for checkpoint or threshold selection.

## Selected From Full VAL
- checkpoint: `best`
- global_threshold: `0.04`
- per_class_thresholds: `{'KD': 0.075, 'SD': 0.03, 'T12': 0.03, 'T14': 0.05, 'T16': 0.05}`

## Full VAL Metrics At Selected Thresholds
- global: `{'TP': 9371, 'FP': 16853, 'FN': 4545, 'micro_precision': 0.35734441732763883, 'micro_recall': 0.6733975280252946, 'micro_f1': 0.4669157947184853, 'micro_fbeta': 0.5293430086034587, 'macro_f1': 0.4611082524204817, 'macro_fbeta': 0.5215802014357133}`
- per-class: `{'TP': 9541, 'FP': 16256, 'FN': 4375, 'micro_precision': 0.3698492072721634, 'micro_recall': 0.6856136820925554, 'micro_f1': 0.4804975700652179, 'micro_fbeta': 0.5429755901099672, 'macro_f1': 0.4713588474575543, 'macro_fbeta': 0.531462066312794}`

## Full TEST Metrics With Frozen VAL Selection
- global: `{'TP': 9437, 'FP': 16970, 'FN': 4535, 'micro_precision': 0.3573673647139016, 'micro_recall': 0.6754222731176639, 'micro_f1': 0.46742118427895685, 'micro_fbeta': 0.5302235322591798, 'macro_f1': 0.4623525399802567, 'macro_fbeta': 0.5229811846579384}`
- per-class: `{'TP': 9577, 'FP': 16324, 'FN': 4395, 'micro_precision': 0.3697540635496699, 'micro_recall': 0.6854423131978242, 'micro_f1': 0.4803751912321621, 'micro_fbeta': 0.5428380829467369, 'macro_f1': 0.4712720821901413, 'macro_fbeta': 0.5315041093983754}`

## TEST By Class: VAL Global Threshold
| class | threshold | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.0400 | 4202 | 8172 | 689 | 0.3396 | 0.8591 | 0.4868 | 0.5841 |
| SD | 0.0400 | 2444 | 3675 | 2446 | 0.3994 | 0.4998 | 0.4440 | 0.4639 |
| T12 | 0.0400 | 960 | 1757 | 437 | 0.3533 | 0.6872 | 0.4667 | 0.5324 |
| T14 | 0.0400 | 814 | 1664 | 583 | 0.3285 | 0.5827 | 0.4201 | 0.4706 |
| T16 | 0.0400 | 1017 | 1702 | 380 | 0.3740 | 0.7280 | 0.4942 | 0.5638 |

## TEST By Class: VAL Per-Class Thresholds
| class | threshold | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.0750 | 3876 | 6512 | 1015 | 0.3731 | 0.7925 | 0.5074 | 0.5888 |
| SD | 0.0300 | 2962 | 4898 | 1928 | 0.3768 | 0.6057 | 0.4646 | 0.5104 |
| T12 | 0.0300 | 1020 | 1997 | 377 | 0.3381 | 0.7301 | 0.4622 | 0.5381 |
| T14 | 0.0500 | 750 | 1420 | 647 | 0.3456 | 0.5369 | 0.4205 | 0.4588 |
| T16 | 0.0500 | 969 | 1497 | 428 | 0.3929 | 0.6936 | 0.5017 | 0.5614 |

## Comparison Against Previous Run Thresholds
- previous_eval_artifacts: `checkpoint_selection_from_val, selected_thresholds_from_val, test_onset_metrics_by_threshold, val_onset_metrics_by_threshold`
- comparison: `{'previous_eval_artifacts': 'checkpoint_selection_from_val, selected_thresholds_from_val, test_onset_metrics_by_threshold, val_onset_metrics_by_threshold', 'previous_checkpoint': 'best', 'previous_global_threshold': 0.125, 'previous_per_class_thresholds': {'KD': 0.125, 'SD': 0.125, 'T12': 0.125, 'T14': 0.125, 'T16': 0.125}, 'fulltrack_test_with_previous_global': {'TP': 4630, 'FP': 6287, 'FN': 9342, 'micro_precision': 0.424109187505725, 'micro_recall': 0.33137703979387345, 'micro_f1': 0.3720519104825425, 'micro_fbeta': 0.35527931246163286, 'macro_f1': 0.34542683653812806, 'macro_fbeta': 0.339445314031217}, 'fulltrack_test_with_previous_per_class': {'TP': 4630, 'FP': 6287, 'FN': 9342, 'micro_precision': 0.424109187505725, 'micro_recall': 0.33137703979387345, 'micro_f1': 0.3720519104825425, 'micro_fbeta': 0.35527931246163286, 'macro_f1': 0.34542683653812806, 'macro_fbeta': 0.339445314031217}, 'fulltrack_test_with_new_global': {'TP': 9437, 'FP': 16970, 'FN': 4535, 'micro_precision': 0.3573673647139016, 'micro_recall': 0.6754222731176639, 'micro_f1': 0.46742118427895685, 'micro_fbeta': 0.5302235322591798, 'macro_f1': 0.4623525399802567, 'macro_fbeta': 0.5229811846579384}, 'fulltrack_test_with_new_per_class': {'TP': 9577, 'FP': 16324, 'FN': 4395, 'micro_precision': 0.3697540635496699, 'micro_recall': 0.6854423131978242, 'micro_f1': 0.4803751912321621, 'micro_fbeta': 0.5428380829467369, 'macro_f1': 0.4712720821901413, 'macro_fbeta': 0.5315041093983754}}`