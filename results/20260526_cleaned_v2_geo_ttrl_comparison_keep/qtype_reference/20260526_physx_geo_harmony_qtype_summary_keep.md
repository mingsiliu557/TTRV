# PhysX-MCQ Geo-Harmony QType Summary

Date: 2026-05-26 UTC

## Best Full Results

| question_type | final strategy | fixed step0 | best acc | delta | invalid / hitmax |
| --- | --- | ---: | ---: | ---: | --- |
| object_category | Geo-Harmony weight update, step 4 | 0.395107 | 0.403002 | +0.007895 | 0.016819 / 0.039130 |
| joint_type | test-time vote8 majority | 0.351393 | 0.365760 | +0.014367 | majority invalid 0.007787, vote hitmax 0.060457 |
| movable_part | test-time vote8 majority | 0.267610 | 0.273028 | +0.005418 | majority invalid 0.038832, vote hitmax 0.110476 |

Metric note: for `joint_type` and `movable_part`, `best acc` is `majority_accuracy` over 8 sampled responses per question, i.e. the reported acc@8 majority-vote result. The JSON top-level `accuracy` in those vote8 runs is response-average accuracy over all sampled rows and is not the headline metric. `pass_at_votes` / `best_accuracy_at_votes` are oracle diagnostics and are not used as the reported method output.

## Step History

| question_type | run type | step | response acc | majority acc@8 | note |
| --- | --- | ---: | ---: | ---: | --- |
| object_category | weight update | 0 | 0.395107 | 0.395107 | fixed baseline |
| object_category | weight update | 4 | 0.403002 | 0.403002 | selected best |
| joint_type | fixed baseline | 0 | 0.351393 | 0.351393 | single response fixed step0 |
| joint_type | vote8 baseline | 0 | 0.331515 | 0.365760 | selected majority-vote output |
| movable_part | fixed baseline | 0 | 0.267610 | 0.267610 | single response fixed step0 |
| movable_part | vote8 baseline | 0 | 0.261025 | 0.273028 | selected majority-vote output |

The final object run only saved validation at step 0 and step 4 because it used `TEST_FREQ=4`. The final joint/movable selected runs are no-update vote8 baselines, so they only have step 0. Earlier failed tuning logs were cleaned from `outputs`; only fixed step0 caches and key archived metrics remain in `results/physx_geo_harmony_qtype_repro_20260526/`.

## TTRL Reference

Reference run ID: `20260524_220158_physx_ttrl_valall_grid_detached/gate075_lr5e9_kl0p2_max24_valall_s3_bs64_stop2pt` (large original log directory cleaned; metrics are archived in this summary).

TTRL adapted step 1 by type:

| question_type | TTRL step1 acc | Geo-Harmony-QType best |
| --- | ---: | ---: |
| object_category | 0.382938 | 0.403002 |
| joint_type | 0.340535 | 0.365760 |
| movable_part | 0.261288 | 0.273028 |

TTRL full run dropped overall from 0.377189 at step0 to 0.365163 at step1. The comparison above is against adapted TTRL, not against every noisy raw step0 sample.

## Collapse Diagnosis

The failure mode matches the mechanism in arXiv 2603.08660: self-reward sharpens whatever predictions are already stable. If stable predictions are wrong, the update amplifies wrong answers. In joint_type runs, training reward stayed positive while validation dropped; one inspected batch had 7/8 agreement on an incorrect joint option.

For this dataset, object_category tolerates small LR Geo-Harmony updates. joint_type and movable_part are safer with no weight update and test-time vote selection.

## Implemented Controls

- Added fixed step0 cache support: `trainer.validation_step0_cache_jsonl`.
- Added per-step validation flow using `TEST_FREQ=1`.
- Added qtype dataset filtering.
- Added `TTRL_MIN_MAJORITY_RATIO` and `TTRL_MAX_MAJORITY_RATIO` gates for pseudo-label reward.
- Added point-cloud reframe policies and Geo-Harmony parsed-response reward logging.

## Recommended Early Stops

1. Save one fixed step0 JSONL per question_type and reuse it for all tuning runs.
2. For weight-update runs, validate every step and stop if current acc drops more than 0.003 from fixed step0, best acc drops more than 0.002, invalid rises by 0.008, or hitmax rises by 0.010.
3. Treat high `train/physx/ttrv_reward_mean` with low `train/physx/majority_acc@votes` as collapse even before validation finishes.
4. For vote8 full runs, monitor streaming prefixes after at least 1000 prompts and abort if majority acc is below fixed step0 by more than 0.005.
5. Keep `unknown_reward=0.0` for fragile types; negative unknown reward made early joint/movable updates less stable.

## Artifacts

- Object: `outputs/20260525_geo_harmony_full_object_category_step4_seed13_lr5e10_hm0p60_r8_unk0_sensor_keep`
- Joint: `outputs/20260526_geo_harmony_full_joint_type_vote8_sensor_baseline_keep`
- Movable: `outputs/20260526_geo_harmony_full_movable_part_vote8_sensor_baseline_keep`
- JSON summary: `outputs/20260526_physx_geo_harmony_qtype_summary_keep.json`
