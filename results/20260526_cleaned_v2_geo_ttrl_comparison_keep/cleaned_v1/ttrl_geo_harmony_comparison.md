# TTRL / Geo-Harmony Cleaned Combined Comparison

Dataset: `physx_cleaned_options_v1` combined validation, 10,572 examples.
Step0 uses the fixed prediction cache. Geo-Harmony and TTRL both used 4 GPUs, rollout/votes 8, lr 5e-10, KL 0.3, validation every step, early-stop plus recovery.

| method | step | acc | delta | joint acc | movable acc | invalid | hit_max | early stop | recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_step0_cache | 0 | 39.77% | +0.00 pp | 41.91% | 26.34% | 1.09% | 10.00% | no | n/a |
| geo_harmony | 1 | 38.98% | -0.79 pp | 41.02% | 26.20% | 1.01% | 10.44% | accuracy_drop | restored step 0 |
| ttrl_majority | 1 | 38.66% | -1.11 pp | 40.63% | 26.27% | 1.00% | 10.22% | accuracy_drop | restored step 0 |

## Train Step1 Reward Snapshot

| method | groups | responses | response acc | reward mean | majority acc | pass@votes | geo hm | view support | tie rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| geo_harmony | 4 | 32 | 37.50% | 0.3125 | 50.00% | 100.00% | 0.5188 | 1.0000 | 0.00% |
| ttrl_majority | 4 | 32 | 40.62% | 0.3750 | 25.00% | 100.00% | 0.0000 | 0.0000 | 50.00% |

## Conclusion

- Cleaned static step0 remains the best checkpoint by acc@1.
- Geo-Harmony step1 dropped by 0.79 percentage points overall; TTRL majority step1 dropped by 1.11 percentage points overall.
- The update still behaves like sharpening the current sampled distribution: stable pseudo-labels/rewards do not imply better ground-truth accuracy.
- Next useful direction is bad-case/data-level analysis on the fixed step0 cache and step1 changed predictions, instead of further scalar hyperparameter search.

## Artifacts

- Step0 cache: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/fixed_step0_caches/combined_validation_predictions_step0.jsonl`
- Geo-Harmony run: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/runs/20260526_cleaned_geo_harmony_combined_sensor_r8_lr5e10_4step_cache`
- TTRL majority run: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/runs/20260526_cleaned_ttrl_majority_combined_r8_lr5e10_4step_cache`
- JSON summary: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/ttrl_geo_harmony_comparison.json`
