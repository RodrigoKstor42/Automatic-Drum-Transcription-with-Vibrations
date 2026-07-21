# Full-track NMS calibration: serious_5class_low_60ep_focal_soft_toms_v3

- dataset: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_original_from_run_config: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_resolved: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_source: `config`
- selection_split: `VAL`
- final_report_split: `TEST`
- selection_metric: `f1`
- beta: `1.5`
- selection_rule: `maximize selected metric -> maximize precision -> minimize FP -> choose highest threshold`
- WARNING: TEST was not used for checkpoint, threshold, or NMS selection.
- TEST no participa en seleccion.
- Esta configuracion es la recomendada para inferencia.
- previous_eval_artifacts: `not_available`

## Selected From Full VAL
- checkpoint: `final`
- threshold: `0.04`
- onset_nms_ms: `150.0`

## Full VAL Metrics At Selected Config
- selected: `{'TP': 7835, 'FP': 6773, 'FN': 6081, 'micro_precision': 0.5363499452354874, 'micro_recall': 0.5630209830411038, 'micro_f1': 0.5493619408217641, 'micro_fbeta': 0.5545362486116858, 'macro_f1': 0.5416692098686877, 'macro_fbeta': 0.5423618764404451}`

## Full TEST Metrics With Frozen VAL Config
- without NMS at same checkpoint/threshold: `{'TP': 9841, 'FP': 20706, 'FN': 4131, 'micro_precision': 0.3221592955118342, 'micro_recall': 0.7043372459204122, 'micro_f1': 0.4421033715941508, 'micro_fbeta': 0.5159920302013423, 'macro_f1': 0.4279815688526824, 'macro_fbeta': 0.4996838136174005}`
- with selected NMS: `{'TP': 7788, 'FP': 6726, 'FN': 6184, 'micro_precision': 0.5365853658536586, 'micro_recall': 0.5574005153163469, 'micro_f1': 0.5467949168012357, 'micro_fbeta': 0.5508258797414638, 'macro_f1': 0.540801679537714, 'macro_fbeta': 0.5396551917257177}`
- change: `{'TP': {'without_nms': 9841, 'with_selected_nms': 7788, 'delta': -2053}, 'FP': {'without_nms': 20706, 'with_selected_nms': 6726, 'delta': -13980}, 'FN': {'without_nms': 4131, 'with_selected_nms': 6184, 'delta': 2053}, 'micro_precision': {'without_nms': 0.3221592955118342, 'with_selected_nms': 0.5365853658536586, 'delta': 0.21442607034182437}, 'micro_recall': {'without_nms': 0.7043372459204122, 'with_selected_nms': 0.5574005153163469, 'delta': -0.1469367306040653}, 'micro_f1': {'without_nms': 0.4421033715941508, 'with_selected_nms': 0.5467949168012357, 'delta': 0.1046915452070849}, 'micro_fbeta': {'without_nms': 0.5159920302013423, 'with_selected_nms': 0.5508258797414638, 'delta': 0.034833849540121475}, 'macro_f1': {'without_nms': 0.4279815688526824, 'with_selected_nms': 0.540801679537714, 'delta': 0.11282011068503156}, 'macro_fbeta': {'without_nms': 0.4996838136174005, 'with_selected_nms': 0.5396551917257177, 'delta': 0.039971378108317246}}`

## TEST By Class: Selected VAL Config
| class | threshold | onset_nms_ms | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.0400 | 150.0000 | 3210 | 3146 | 1681 | 0.5050 | 0.6563 | 0.5708 | 0.6009 |
| SD | 0.0400 | 150.0000 | 2399 | 1779 | 2491 | 0.5742 | 0.4906 | 0.5291 | 0.5136 |
| T12 | 0.0400 | 150.0000 | 797 | 876 | 600 | 0.4764 | 0.5705 | 0.5192 | 0.5378 |
| T14 | 0.0400 | 150.0000 | 648 | 541 | 749 | 0.5450 | 0.4639 | 0.5012 | 0.4861 |
| T16 | 0.0400 | 150.0000 | 734 | 384 | 663 | 0.6565 | 0.5254 | 0.5837 | 0.5598 |

## Interpretation
- NMS is applied per track and per class before onset matching.
- The selected NMS value comes only from VAL complete tracks.
- TEST is reported only after freezing checkpoint, threshold, and NMS from VAL.
- This frozen VAL-selected configuration is recommended for inference.

No previous eval_steps calibration artifacts were found. This is expected for runs trained with the new protocol.