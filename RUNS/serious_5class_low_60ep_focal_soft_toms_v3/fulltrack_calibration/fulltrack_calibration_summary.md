# Full-track calibration: serious_5class_low_60ep_focal_soft_toms_v3

- dataset: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_original_from_run_config: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_resolved: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_source: `config`
- selection_split: `VAL`
- final_report_split: `TEST`
- selection_metric: `f1`
- beta: `1.5`
- selection_rule: `maximize selected metric -> maximize precision -> minimize FP -> choose highest threshold`
- WARNING: TEST was not used for checkpoint or threshold selection.

## Selected From Full VAL
- checkpoint: `final`
- global_threshold: `0.05`
- per_class_thresholds: `{'KD': 0.1, 'SD': 0.04, 'T12': 0.05, 'T14': 0.04, 'T16': 0.04}`

## Full VAL Metrics At Selected Thresholds
- global: `{'TP': 8816, 'FP': 16505, 'FN': 5100, 'micro_precision': 0.3481695035741085, 'micro_recall': 0.6335153779821788, 'micro_f1': 0.4493717664449372, 'micro_fbeta': 0.5059330413900269, 'macro_f1': 0.4316564869998416, 'macro_fbeta': 0.4854174391455768}`
- per-class: `{'TP': 9127, 'FP': 16945, 'FN': 4789, 'micro_precision': 0.3500690395826941, 'micro_recall': 0.6558637539522851, 'micro_f1': 0.45648694608382523, 'micro_fbeta': 0.5169257445584929, 'macro_f1': 0.4407130026852434, 'macro_fbeta': 0.5023469179431215}`

## Full TEST Metrics With Frozen VAL Selection
- global: `{'TP': 8820, 'FP': 16466, 'FN': 5152, 'micro_precision': 0.34880961797041843, 'micro_recall': 0.6312625250501002, 'micro_f1': 0.4493351673544245, 'micro_fbeta': 0.5053505632635792, 'macro_f1': 0.43202036388368575, 'macro_fbeta': 0.486180018221184}`
- per-class: `{'TP': 9150, 'FP': 16870, 'FN': 4822, 'micro_precision': 0.351652574942352, 'micro_recall': 0.6548811909533352, 'micro_f1': 0.4575915183036607, 'micro_fbeta': 0.5175609586299319, 'macro_f1': 0.4413066501605113, 'macro_fbeta': 0.5024928910158917}`

## TEST By Class: VAL Global Threshold
| class | threshold | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.0500 | 3885 | 6662 | 1006 | 0.3684 | 0.7943 | 0.5033 | 0.5859 |
| SD | 0.0500 | 2443 | 4447 | 2447 | 0.3546 | 0.4996 | 0.4148 | 0.4437 |
| T12 | 0.0500 | 992 | 2358 | 405 | 0.2961 | 0.7101 | 0.4179 | 0.4965 |
| T14 | 0.0500 | 739 | 1635 | 658 | 0.3113 | 0.5290 | 0.3919 | 0.4353 |
| T16 | 0.0500 | 761 | 1364 | 636 | 0.3581 | 0.5447 | 0.4321 | 0.4695 |

## TEST By Class: VAL Per-Class Thresholds
| class | threshold | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.1000 | 3369 | 4326 | 1522 | 0.4378 | 0.6888 | 0.5354 | 0.5855 |
| SD | 0.0400 | 3087 | 6406 | 1803 | 0.3252 | 0.6313 | 0.4293 | 0.4895 |
| T12 | 0.0500 | 992 | 2358 | 405 | 0.2961 | 0.7101 | 0.4179 | 0.4965 |
| T14 | 0.0400 | 839 | 2043 | 558 | 0.2911 | 0.6006 | 0.3921 | 0.4526 |
| T16 | 0.0400 | 863 | 1737 | 534 | 0.3319 | 0.6178 | 0.4318 | 0.4884 |

## Comparison Against Previous Run Thresholds
- previous_eval_artifacts: `not_available`
- comparison: `{'previous_eval_artifacts': 'not_available', 'message': 'No previous eval_steps calibration artifacts were found. This is expected for runs trained with the new protocol.', 'fulltrack_test_with_new_global': {'TP': 8820, 'FP': 16466, 'FN': 5152, 'micro_precision': 0.34880961797041843, 'micro_recall': 0.6312625250501002, 'micro_f1': 0.4493351673544245, 'micro_fbeta': 0.5053505632635792, 'macro_f1': 0.43202036388368575, 'macro_fbeta': 0.486180018221184}, 'fulltrack_test_with_new_per_class': {'TP': 9150, 'FP': 16870, 'FN': 4822, 'micro_precision': 0.351652574942352, 'micro_recall': 0.6548811909533352, 'micro_f1': 0.4575915183036607, 'micro_fbeta': 0.5175609586299319, 'macro_f1': 0.4413066501605113, 'macro_fbeta': 0.5024928910158917}}`

No previous eval_steps calibration artifacts were found. This is expected for runs trained with the new protocol.