# Object Category Reliable128 Best Keep

This directory records the current best object_category adaptation result from the step0-reliable128 diverse_text run. The source run directory is already marked `_keep`:

`/root/autodl-tmp/TTRV/outputs/physx_reliable128_soft_keep/runs/20260528_object_original_reliable128_diverse_text_geo_harmony_soft_r8_lr5e10_gamma2_32step_test4_b4_keep`

## Best Result

| step | acc@1 | delta vs step0 | invalid rate | hitmax rate | note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.395107186 | 0.0000 pt | 0.016039 | 0.033139 | cached static baseline |
| 4 | 0.403906762 | +0.8800 pt | 0.015540 | 0.038288 | current best |
| 8 | 0.244515867 | -15.0591 pt | 0.018691 | 0.054202 | collapse after more updates |

## Interpretation

Step4 improves object_category acc@1 from `0.395107186` to `0.403906762`, a gain of `+0.8800` points on the full 32,047-example object validation set. Step8 collapses to `0.244515867`, so this run should be cited as **best-step selection by validation acc@1**, not as final-step performance.

Important caveat: `TRAIN_BATCH_SIZE=4`, so step4 has consumed the first 16 adaptation prompts from the reliable128 ordering. This supports the idea that increasing the candidate pool helps, but also shows later samples can inject bad pseudo-labels.

## Saved Artifacts

Lightweight metric/report files are copied here. Large prediction/log files are symlinked back to the `_keep` source run:

- `val_metrics_step4.json`: best validation metrics.
- `validation_predictions_step4.jsonl`: symlink to full step4 predictions.
- `training_predictions_step1-4.jsonl`: symlinks for training rollout inspection up to the best step.
- `run_config.txt` and `command.sh`: exact rerun configuration.

## Checkpoint Note

No step4 model checkpoint is available from this run because it used `SAVE_FREQ=-1`. The best result is preserved as predictions/metrics/logs. To preserve weights for this configuration, rerun the same command with `SAVE_FREQ=1`, `DEBUG_STEPS=4`, and validation at every step.

## Training Samples Preserved

The exact adaptation sample ordering is preserved under `/root/autodl-tmp/TTRV/results/20260528_object_reliable128_step4_best_keep/training_samples`. This matters because the best validation point is step4 with `TRAIN_BATCH_SIZE=4`, so only the first 16 selected prompts had been consumed before the best validation.

- Full ordered reliable128 sample list: `training_samples/selected_training_samples_128.jsonl` and `.csv`.
- Exact first128 training rows: `training_samples/selected_training_rows_first128.parquet`.
- Exact best-step consumed first16 rows: `training_samples/best_step_consumed_rows_first16.parquet`.
- Human-readable selection table: `training_samples/step0_reliable128_diverse_text_object_category.md`.
- Source dataset symlink: `training_samples/source_dataset_dir`.

First128 answer counts: `{'A': 32, 'B': 32, 'C': 32, 'D': 32}`. First16 answer counts: `{'A': 4, 'B': 4, 'C': 4, 'D': 4}`.

