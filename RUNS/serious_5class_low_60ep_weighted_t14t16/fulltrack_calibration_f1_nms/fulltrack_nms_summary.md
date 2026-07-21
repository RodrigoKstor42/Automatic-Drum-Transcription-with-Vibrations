# Full-track NMS calibration: serious_5class_low_60ep_weighted_t14t16

- dataset: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- selection_split: `VAL`
- final_report_split: `TEST`
- selection_metric: `f1`
- beta: `1.5`
- selection_rule: `maximize selected metric -> maximize precision -> minimize FP -> choose highest threshold`
- WARNING: TEST was not used for checkpoint, threshold, or NMS selection.
- TEST no participa en seleccion.
- Esta configuracion es la recomendada para inferencia.

## Selected From Full VAL
- checkpoint: `best`
- threshold: `0.03`
- onset_nms_ms: `110.0`

## Full VAL Metrics At Selected Config
- selected: `{'TP': 8622, 'FP': 6952, 'FN': 5294, 'micro_precision': 0.5536149993579041, 'micro_recall': 0.6195745903995401, 'micro_f1': 0.5847405900305188, 'micro_fbeta': 0.5976644982403754, 'macro_f1': 0.5862380797526118, 'macro_fbeta': 0.5941474215541145}`

## Full TEST Metrics With Frozen VAL Config
- without NMS at same checkpoint/threshold: `{'TP': 10237, 'FP': 19582, 'FN': 3735, 'micro_precision': 0.3433046044468292, 'micro_recall': 0.7326796450042943, 'micro_f1': 0.4675389920303259, 'micro_fbeta': 0.5431345500848895, 'macro_f1': 0.46269461342888674, 'macro_fbeta': 0.5368255199357562}`
- with selected NMS: `{'TP': 8657, 'FP': 6901, 'FN': 5315, 'micro_precision': 0.5564339889445944, 'micro_recall': 0.6195963355281993, 'micro_f1': 0.5863189976295292, 'micro_fbeta': 0.598686030428769, 'macro_f1': 0.5854869775873086, 'macro_fbeta': 0.5923404859857292}`
- change: `{'TP': {'without_nms': 10237, 'with_selected_nms': 8657, 'delta': -1580}, 'FP': {'without_nms': 19582, 'with_selected_nms': 6901, 'delta': -12681}, 'FN': {'without_nms': 3735, 'with_selected_nms': 5315, 'delta': 1580}, 'micro_precision': {'without_nms': 0.3433046044468292, 'with_selected_nms': 0.5564339889445944, 'delta': 0.2131293844977652}, 'micro_recall': {'without_nms': 0.7326796450042943, 'with_selected_nms': 0.6195963355281993, 'delta': -0.11308330947609502}, 'micro_f1': {'without_nms': 0.4675389920303259, 'with_selected_nms': 0.5863189976295292, 'delta': 0.11878000559920332}, 'micro_fbeta': {'without_nms': 0.5431345500848895, 'with_selected_nms': 0.598686030428769, 'delta': 0.05555148034387947}, 'macro_f1': {'without_nms': 0.46269461342888674, 'with_selected_nms': 0.5854869775873086, 'delta': 0.12279236415842187}, 'macro_fbeta': {'without_nms': 0.5368255199357562, 'with_selected_nms': 0.5923404859857292, 'delta': 0.05551496604997297}}`

## TEST By Class: Selected VAL Config
| class | threshold | onset_nms_ms | TP | FP | FN | precision | recall | f1 | fbeta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KD | 0.0300 | 110.0000 | 3640 | 3698 | 1251 | 0.4960 | 0.7442 | 0.5953 | 0.6449 |
| SD | 0.0300 | 110.0000 | 2533 | 1413 | 2357 | 0.6419 | 0.5180 | 0.5733 | 0.5507 |
| T12 | 0.0300 | 110.0000 | 820 | 604 | 577 | 0.5758 | 0.5870 | 0.5814 | 0.5835 |
| T14 | 0.0300 | 110.0000 | 750 | 618 | 647 | 0.5482 | 0.5369 | 0.5425 | 0.5403 |
| T16 | 0.0300 | 110.0000 | 914 | 568 | 483 | 0.6167 | 0.6543 | 0.6349 | 0.6422 |

## Interpretation
- NMS is applied per track and per class before onset matching.
- The selected NMS value comes only from VAL complete tracks.
- TEST is reported only after freezing checkpoint, threshold, and NMS from VAL.
- This frozen VAL-selected configuration is recommended for inference.