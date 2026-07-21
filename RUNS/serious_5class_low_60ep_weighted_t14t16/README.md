# Run summary

- run_dir: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\RUNS\serious_5class_low_60ep_weighted_t14t16`
- dataset_root: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\PHRASE_GENERATOR\CUSTOM_5CLASS_MULTIPROFILE_LOW_FULL`
- class_schema: `vibro_5_toms`
- class_names: `KD, SD, T12, T14, T16`
- labels: `35, 38, 47, 45, 43`

## Training configuration

- model_name: `Frame_RNN`
- epochs: `60`
- steps_per_epoch: `600`
- validation_steps: `150`
- batch_size: `1`
- training_sequence: `64`
- context: `9`
- sample_rate: `100`
- class_weights_order: `KD, SD, T12, T14, T16`
- class_weights: `1.0, 1.0, 1.2, 1.5, 1.7`
- onset_tolerance_ms: `50.0`
- threshold_grid: `0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125`

## Checkpoints

- best: available (C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\RUNS\serious_5class_low_60ep_weighted_t14t16\checkpoints\best_weights.weights.h5)
- final: available (C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\RUNS\serious_5class_low_60ep_weighted_t14t16\checkpoints\final_weights.weights.h5)

## VAL threshold calibration

- calibration_enabled: `True`
- selection_split: `VAL`
- selected_checkpoint: `best`
- threshold_selection_rule: `maximize F1 -> maximize precision -> minimize FP -> choose highest threshold`
- threshold_selection_eps: `1e-09`
- selected_global_threshold_from_VAL: `0.125`
- selected_per_class_thresholds_from_VAL: `KD=0.125, SD=0.125, T12=0.125, T14=0.125, T16=0.125`
- TEST was used only after threshold/checkpoint selection.

## Final TEST metrics using VAL thresholds

Global VAL threshold applied to TEST:

- micro_precision: `0.6081`
- micro_recall: `0.6119`
- micro_f1: `0.6100`
- macro_f1: `0.6005`
- TP/FP/FN: `3135/2020/1988`

Per-class VAL thresholds applied to TEST:

- micro_precision: `0.6081`
- micro_recall: `0.6119`
- micro_f1: `0.6100`
- macro_f1: `0.6005`
- TP/FP/FN: `3135/2020/1988`


## Evaluation coverage

- Note: evaluation is limited by eval_steps batches; confirm coverage manually if full-split reporting is required.
