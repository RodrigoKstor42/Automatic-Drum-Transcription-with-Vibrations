# Run summary

- run_dir: `C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\RUNS\serious_5class_low_60ep_focal_soft_toms_v2`
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
- class_weights: `1.0, 1.0, 1.1, 1.3, 1.5`
- loss_type: `focal`
- focal_gamma: `1.0`
- focal_alpha: `0.25`
- onset_tolerance_ms: `50.0`
- threshold_grid: `0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5`

## Checkpoints

- best: available (C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\RUNS\serious_5class_low_60ep_focal_soft_toms_v2\checkpoints\best_weights.weights.h5)
- final: available (C:\Users\rsanc\Documents\MAGISTER EN ACUSTICA Y VIBRACIONES\SEMESTRE IV\Automatic-Drum-Transcription-with-Vibrations\RUNS\serious_5class_low_60ep_focal_soft_toms_v2\checkpoints\final_weights.weights.h5)

## VAL threshold calibration

- calibration_enabled: `False`
- No VAL calibration was requested for this run.

## Final TEST metrics using VAL thresholds

No calibrated TEST metrics were generated.
