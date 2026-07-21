# Full-track NMS calibration: serious_5class_low_60ep_focal_soft_toms_v2

- dataset: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_original_from_run_config: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_resolved: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\DATASET_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- dataset_root_source: `cli_override`
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
- checkpoint: `best`
- threshold: `0.05`
- onset_nms_ms: `110.0`

## Full VAL Metrics At Selected Config
- selected: `{'TP': 9005, 'FP': 6065, 'FN': 4911, 'micro_precision': 0.5975447909754479, 'micro_recall': 0.6470968669157804, 'micro_f1': 0.6213344373145656, 'micro_fbeta': 0.6309965287509972, 'macro_f1': 0.5956706154190894, 'macro_fbeta': 0.5858672344847682}`

## Full TEST Metrics With Frozen VAL Config
- without NMS at same checkpoint/threshold: `{'TP': 10535, 'FP': 20438, 'FN': 3437, 'micro_precision': 0.3401349562522197, 'micro_recall': 0.7540080160320641, 'micro_f1': 0.46879519412615417, 'micro_fbeta': 0.5486099983976928, 'macro_f1': 0.45757977978326136, 'macro_fbeta': 0.5161261043766925}`
- with selected NMS: `{'TP': 8985, 'FP': 6134, 'FN': 4987, 'micro_precision': 0.5942853363317679, 'micro_recall': 0.6430718580017177, 'micro_f1': 0.6177168196349387, 'micro_fbeta': 0.6272284990119426, 'macro_f1': 0.589873833503554, 'macro_fbeta': 0.5797787366088839}`
- change: `{'TP': {'without_nms': 10535, 'with_selected_nms': 8985, 'delta': -1550}, 'FP': {'without_nms': 20438, 'with_selected_nms': 6134, 'delta': -14304}, 'FN': {'without_nms': 3437, 'with_selected_nms': 4987, 'delta': 1550}, 'micro_precision': {'without_nms': 0.3401349562522197, 'with_selected_nms': 0.5942853363317679, 'delta': 0.25415038007954827}, 'micro_recall': {'without_nms': 0.7540080160320641, 'with_selected_nms': 0.6430718580017177, 'delta': -0.1109361580303464}, 'micro_f1': {'without_nms': 0.46879519412615417, 'with_selected_nms': 0.6177168196349387, 'delta': 0.14892162550878457}, 'micro_fbeta': {'without_nms': 0.5486099983976928, 'with_selected_nms': 0.6272284990119426, 'delta': 0.0786185006142498}, 'macro_f1': {'without_nms': 0.45757977978326136, 'with_selected_nms': 0.589873833503554, 'delta': 0.13229405372029268}, 'macro_fbeta': {'without_nms': 0.5161261043766925, 'with_selected_nms': 0.5797787366088839, 'delta': 0.06365263223219142}}`

## TEST By Class: Selected VAL Config
| class | threshold | onset_nms_ms | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.0500 | 110.0000 | 3819 | 3192 | 1072 | 0.5447 | 0.7808 | 0.6417 | 0.6889 |
| SD | 0.0500 | 110.0000 | 3166 | 1957 | 1724 | 0.6180 | 0.6474 | 0.6324 | 0.6381 |
| T12 | 0.0500 | 110.0000 | 640 | 354 | 757 | 0.6439 | 0.4581 | 0.5353 | 0.5027 |
| T14 | 0.0500 | 110.0000 | 667 | 431 | 730 | 0.6075 | 0.4775 | 0.5347 | 0.5111 |
| T16 | 0.0500 | 110.0000 | 693 | 200 | 704 | 0.7760 | 0.4961 | 0.6052 | 0.5580 |

## Interpretation
- NMS is applied per track and per class before onset matching.
- The selected NMS value comes only from VAL complete tracks.
- TEST is reported only after freezing checkpoint, threshold, and NMS from VAL.
- This frozen VAL-selected configuration is recommended for inference.

No previous eval_steps calibration artifacts were found. This is expected for runs trained with the new protocol.