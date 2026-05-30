# PhysX-MCQ Geo-Harmony QType Repro Report

Date: 2026-05-26 UTC

This directory keeps the reproducible summary for the final PointLLM + PhysX-MCQ Geo-Harmony/TTRV full-data experiments split by `question_type`.

## Final Results

| question_type | final strategy | fixed step0 | best acc | delta |
| --- | --- | ---: | ---: | ---: |
| `object_category` | Geo-Harmony weight update, step 4 | 0.395107 | 0.403002 | +0.007895 |
| `joint_type` | test-time vote8 majority | 0.351393 | 0.365760 | +0.014367 |
| `movable_part` | test-time vote8 majority | 0.267610 | 0.273028 | +0.005418 |

Metric note:

- For `object_category`, `best acc` is normal parsed-response accuracy after the step-4 weight update. Validation used one response per question, so `accuracy`, `response_accuracy`, and `majority_accuracy` are identical.
- For `joint_type` and `movable_part`, `best acc` is `majority_accuracy` over 8 sampled responses per question, i.e. the reported `acc@8` majority-vote result.
- The JSON top-level `accuracy` in vote8 runs is the mean accuracy over all sampled responses, not the final method output. It is lower than majority accuracy and should not be used as the headline metric.
- `pass_at_votes` / `best_accuracy_at_votes` are oracle-style diagnostics: whether any of the 8 sampled responses was correct. They are not the deployed prediction accuracy.

## Step History

Current archived step metrics:

| question_type | run type | step | response acc | majority acc@8 | note |
| --- | --- | ---: | ---: | ---: | --- |
| `object_category` | weight update | 0 | 0.395107 | 0.395107 | fixed baseline |
| `object_category` | weight update | 4 | 0.403002 | 0.403002 | selected best |
| `joint_type` | fixed baseline | 0 | 0.351393 | 0.351393 | single response fixed step0 |
| `joint_type` | vote8 baseline | 0 | 0.331515 | 0.365760 | selected majority-vote output |
| `movable_part` | fixed baseline | 0 | 0.267610 | 0.267610 | single response fixed step0 |
| `movable_part` | vote8 baseline | 0 | 0.261025 | 0.273028 | selected majority-vote output |

The final `object_category` run only saved validation at step 0 and step 4 because it was launched with `TEST_FREQ=4`. `joint_type` and `movable_part` final selected runs are `RUN_MODE=baseline`, so they have no training-step history; the step is 0 with vote8 sampling. Earlier failed tuning logs were cleaned from `outputs`; their fixed step0 caches and key metrics are archived here.

The comparison target is the adapted TTRL step1 run. The original large TTRL log directory was cleaned from the outputs directory; the comparison metrics are archived in summary.json.

| question_type | TTRL step1 acc | Geo-Harmony-QType best |
| --- | ---: | ---: |
| `object_category` | 0.382938 | 0.403002 |
| `joint_type` | 0.340535 | 0.365760 |
| `movable_part` | 0.261288 | 0.273028 |

## Kept Artifacts

- Full summary: `summary.md`, `summary.json`
- Reproduction commands: `commands/*.sh`
- Metrics and run configs: `metrics/<question_type>/`
- Fixed baseline caches for reuse: `fixed_step0_caches/*.jsonl`

The full prediction JSONL artifacts for the best runs remain in:

- `/root/autodl-tmp/TTRV/outputs/20260525_geo_harmony_full_object_category_step4_seed13_lr5e10_hm0p60_r8_unk0_sensor_keep`
- `/root/autodl-tmp/TTRV/outputs/20260526_geo_harmony_full_joint_type_vote8_sensor_baseline_keep`
- `/root/autodl-tmp/TTRV/outputs/20260526_geo_harmony_full_movable_part_vote8_sensor_baseline_keep`

## Reproduction Commands

Run from `/root/autodl-tmp/TTRV`.

### Object Category

```bash
bash results/physx_geo_harmony_qtype_repro_20260526/commands/object_category_best.sh
```

This is the only final strategy that updates weights. It uses 4 GPUs, full `object_category`, `sensor_noise` point reframe, Geo-Harmony reward, rollout/votes 8, LR `5e-10`, KL `0.3`, and selects step 4.

### Joint Type

```bash
bash results/physx_geo_harmony_qtype_repro_20260526/commands/joint_type_vote8_majority.sh
```

This uses no weight update. The reported result is the majority-selected prediction across 8 sampled validation responses, not the per-vote mean accuracy.

### Movable Part

```bash
bash results/physx_geo_harmony_qtype_repro_20260526/commands/movable_part_vote8_majority.sh
```

This also uses no weight update and reports vote8 majority accuracy.

## Fixed Step0 Caches

For future tuning runs, reuse fixed step0 responses to reduce baseline noise:

```bash
VALIDATION_STEP0_CACHE_JSONL=/root/autodl-tmp/TTRV/results/physx_geo_harmony_qtype_repro_20260526/fixed_step0_caches/object_category_validation_predictions_step0.jsonl
```

```bash
VALIDATION_STEP0_CACHE_JSONL=/root/autodl-tmp/TTRV/results/physx_geo_harmony_qtype_repro_20260526/fixed_step0_caches/joint_type_validation_predictions_step0.jsonl
```

```bash
VALIDATION_STEP0_CACHE_JSONL=/root/autodl-tmp/TTRV/results/physx_geo_harmony_qtype_repro_20260526/fixed_step0_caches/movable_part_validation_predictions_step0.jsonl
```

## Collapse Notes

The observed failure mode matches arXiv 2603.08660: intrinsic self-reward sharpens stable predictions. If stable predictions are wrong, the update reinforces wrong pseudo-labels. In `joint_type` and `movable_part`, weight-update runs raised pseudo reward but validation dropped, so the final policy uses test-time vote selection instead of adaptation.

Recommended tuning gates:

1. Validate every step with `TEST_FREQ=1` during tuning.
2. Reuse fixed step0 JSONL via `trainer.validation_step0_cache_jsonl`.
3. Stop if accuracy drops by more than 0.003 from fixed step0, best accuracy drops by more than 0.002, invalid rate rises by 0.008, or hit-max-response-length rises by 0.010.
4. Treat high train reward with low majority accuracy as collapse.
