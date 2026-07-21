# Full-track calibration: serious_5class_low_60ep_focal_soft_toms_v2

- dataset: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- selection_split: `VAL`
- final_report_split: `TEST`
- selection_metric: `f1`
- beta: `1.5`
- selection_rule: `maximize selected metric -> maximize precision -> minimize FP -> choose highest threshold`
- WARNING: TEST was not used for checkpoint or threshold selection.

## Selected From Full VAL
- checkpoint: `best`
- global_threshold: `0.05`
- per_class_thresholds: `{'KD': 0.125, 'SD': 0.075, 'T12': 0.05, 'T14': 0.04, 'T16': 0.03}`

## Full VAL Metrics At Selected Thresholds
- global: `{'TP': 10548, 'FP': 20181, 'FN': 3368, 'micro_precision': 0.3432588108952455, 'micro_recall': 0.7579764300086231, 'micro_f1': 0.4725277186695039, 'micro_fbeta': 0.5525628626692456, 'macro_f1': 0.46414765581956374, 'macro_fbeta': 0.5238259209060441}`
- per-class: `{'TP': 8246, 'FP': 11635, 'FN': 5670, 'micro_precision': 0.41476786881947586, 'micro_recall': 0.5925553319919518, 'micro_f1': 0.4879723052341923, 'micro_fbeta': 0.5235095327394906, 'macro_f1': 0.48244906307404045, 'macro_fbeta': 0.5268188305548203}`

## Full TEST Metrics With Frozen VAL Selection
- global: `{'TP': 10535, 'FP': 20438, 'FN': 3437, 'micro_precision': 0.3401349562522197, 'micro_recall': 0.7540080160320641, 'micro_f1': 0.46879519412615417, 'micro_fbeta': 0.5486099983976928, 'macro_f1': 0.45757977978326136, 'macro_fbeta': 0.5161261043766925}`
- per-class: `{'TP': 8179, 'FP': 11682, 'FN': 5793, 'micro_precision': 0.41181209405367303, 'micro_recall': 0.5853850558259376, 'micro_f1': 0.48349244820146015, 'micro_fbeta': 0.5181829700963, 'macro_f1': 0.47780248653095186, 'macro_fbeta': 0.520926563089199}`

## TEST By Class: VAL Global Threshold
| class | threshold | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.0500 | 4499 | 9270 | 392 | 0.3267 | 0.9199 | 0.4822 | 0.5902 |
| SD | 0.0500 | 3736 | 7301 | 1154 | 0.3385 | 0.7640 | 0.4691 | 0.5509 |
| T12 | 0.0500 | 753 | 1309 | 644 | 0.3652 | 0.5390 | 0.4354 | 0.4702 |
| T14 | 0.0500 | 788 | 1535 | 609 | 0.3392 | 0.5641 | 0.4237 | 0.4685 |
| T16 | 0.0500 | 759 | 1023 | 638 | 0.4259 | 0.5433 | 0.4775 | 0.5008 |

## TEST By Class: VAL Per-Class Thresholds
| class | threshold | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.1250 | 2656 | 2559 | 2235 | 0.5093 | 0.5430 | 0.5256 | 0.5322 |
| SD | 0.0750 | 2754 | 4213 | 2136 | 0.3953 | 0.5632 | 0.4645 | 0.4981 |
| T12 | 0.0500 | 753 | 1309 | 644 | 0.3652 | 0.5390 | 0.4354 | 0.4702 |
| T14 | 0.0400 | 946 | 2055 | 451 | 0.3152 | 0.6772 | 0.4302 | 0.5004 |
| T16 | 0.0300 | 1070 | 1546 | 327 | 0.4090 | 0.7659 | 0.5333 | 0.6038 |

## Comparison Against Previous Run Thresholds
- previous_eval_artifacts: `not_available`
- comparison: `{'previous_eval_artifacts': 'not_available', 'message': 'No previous eval_steps calibration artifacts were found. This is expected for runs trained with the new protocol.', 'fulltrack_test_with_new_global': {'TP': 10535, 'FP': 20438, 'FN': 3437, 'micro_precision': 0.3401349562522197, 'micro_recall': 0.7540080160320641, 'micro_f1': 0.46879519412615417, 'micro_fbeta': 0.5486099983976928, 'macro_f1': 0.45757977978326136, 'macro_fbeta': 0.5161261043766925}, 'fulltrack_test_with_new_per_class': {'TP': 8179, 'FP': 11682, 'FN': 5793, 'micro_precision': 0.41181209405367303, 'micro_recall': 0.5853850558259376, 'micro_f1': 0.48349244820146015, 'micro_fbeta': 0.5181829700963, 'macro_f1': 0.47780248653095186, 'macro_fbeta': 0.520926563089199}}`

No previous eval_steps calibration artifacts were found. This is expected for runs trained with the new protocol.