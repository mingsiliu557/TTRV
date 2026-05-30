# Reliable20 Soft-label Geo-Harmony Results

Date: 2026-05-28

## Setup

- Method: soft-label Geo-Harmony reward, `q_geo(a) = normalize(geo_score(a)^gamma)`, gamma=2.0.
- Adaptation samples: 20 reliable step0 samples per subtask, selected from cached step0 predictions. This is an oracle diagnostic selector because correctness is used for selection.
- Validation: deterministic freeform MCQ, acc@1 primary, `VAL_ROLLOUT_N=1`, temperature=0.0, top_p=1.0.
- GPUs: 4. Point cloud: full object OBJ surface points, pointnum=8192.
- Data: object_category uses the original uncleaned PhysX MCQ object type; joint_type and movable_part use clean v2 optimized data.

## Accuracy Summary

| Subtask | Data | n | step0 acc@1 | best step | best acc@1 | delta | final/eval stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| object_category original | original uncleaned PhysX object_category; no clean filtering | 32047 | 39.51% | 0 | 39.51% | +0.00 pp | step1 acc 39.07% |
| joint_type clean v2 | optimized clean v2 joint_type subset | 7669 | 52.24% | 3 | 53.20% | +0.96 pp | step5 acc 52.28% |
| movable_part clean v2 | optimized clean v2 movable_part subset | 696 | 26.01% | 0 | 26.01% | +0.00 pp | step3 acc 24.71% |

## Validation Curves

### object_category original

| step | acc@1 | invalid | hit_max_len | correct / n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 39.51% | 1.60% | 3.31% | 12662 / 32047 |
| 1 | 39.07% | 1.34% | 3.12% | 12520 / 32047 |

### joint_type clean v2

| step | acc@1 | invalid | hit_max_len | correct / n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 52.24% | 1.63% | 5.85% | 4006 / 7669 |
| 1 | 52.67% | 1.33% | 5.89% | 4039 / 7669 |
| 2 | 52.86% | 1.16% | 6.14% | 4054 / 7669 |
| 3 | 53.20% | 1.29% | 5.44% | 4080 / 7669 |
| 4 | 52.91% | 1.24% | 5.18% | 4058 / 7669 |
| 5 | 52.28% | 1.15% | 5.14% | 4009 / 7669 |

### movable_part clean v2

| step | acc@1 | invalid | hit_max_len | correct / n |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 26.01% | 1.01% | 4.74% | 181 / 696 |
| 1 | 25.72% | 1.29% | 4.45% | 179 / 696 |
| 2 | 25.86% | 1.15% | 4.02% | 180 / 696 |
| 3 | 24.71% | 1.44% | 4.45% | 172 / 696 |

## Adaptation Batch Diagnostics

These metrics are computed on the 4 prompts x 8 rollouts actually used by each update step, not on the full validation set.

### object_category original

| step | response acc | Pass@8 | Majority@8 | geo soft selected | GT mass | HM mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 56.25% | 100.00% | 75.00% | 50.00% | 71.30% | 0.583 |

### joint_type clean v2

| step | response acc | Pass@8 | Majority@8 | geo soft selected | GT mass | HM mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65.62% | 100.00% | 100.00% | 50.00% | 82.50% | 0.637 |
| 2 | 56.25% | 100.00% | 50.00% | 25.00% | 70.19% | 0.485 |
| 3 | 81.25% | 100.00% | 75.00% | 75.00% | 87.50% | 0.813 |
| 4 | 50.00% | 100.00% | 75.00% | 25.00% | 53.50% | 0.548 |
| 5 | 65.62% | 100.00% | 100.00% | 75.00% | 91.30% | 0.639 |

### movable_part clean v2

| step | response acc | Pass@8 | Majority@8 | geo soft selected | GT mass | HM mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 59.38% | 100.00% | 100.00% | 50.00% | 79.81% | 0.588 |
| 2 | 59.38% | 100.00% | 75.00% | 25.00% | 74.81% | 0.527 |
| 3 | 75.00% | 100.00% | 100.00% | 75.00% | 100.00% | 0.710 |

## Interpretation

- `joint_type clean v2` is the only subtask with a real positive result: step0 52.24% to best step3 53.20%, +0.96 pp. Later steps regress, so recovery selects step3.
- `object_category original` does not improve: step1 drops from 39.51% to 39.07%; early stop keeps step0.
- `movable_part clean v2` does not improve: best remains step0 26.01%; step3 drops to 24.71%, and recovery restores step0.
- In all three subtasks, adaptation batches often have Pass@8=100%. That means PointLLM can sample a correct answer on the selected reliable prompts. The failure mode is not simply no correct samples; it is that the update from only 20 reliable prompts does not generalize evenly, especially for object_category and movable_part.
- Majority@8 is high on many update batches, but acc@1 can still be low because Majority@8 is measured on the selected adaptation prompts under stochastic rollouts, while acc@1 is deterministic validation over the full subtask distribution. High majority on easy/reliable prompts can sharpen local behavior without improving hard validation cases.

## Artifacts

- object_category original run: `/root/autodl-tmp/TTRV/outputs/physx_reliable20_soft_keep/runs/20260528_object_original_reliable20_geo_harmony_soft_r8_lr5e10_gamma2_5step_b4_keep`
- object_category original reliable20 data: `/root/autodl-tmp/TTRV/outputs/physx_reliable20_soft_keep/object_category_original_step0_reliable20_keep`
- joint_type clean v2 run: `/root/autodl-tmp/TTRV/outputs/physx_reliable20_soft_keep/runs/20260528_joint_v2_reliable20_geo_harmony_soft_r8_lr5e10_gamma2_5step_b4_keep`
- joint_type clean v2 reliable20 data: `/root/autodl-tmp/TTRV/outputs/physx_reliable20_soft_keep/joint_type_v2_step0_reliable20_keep`
- movable_part clean v2 run: `/root/autodl-tmp/TTRV/outputs/physx_reliable20_soft_keep/runs/20260528_movable_v2_reliable20_geo_harmony_soft_r8_lr5e10_gamma2_5step_b4_keep`
- movable_part clean v2 reliable20 data: `/root/autodl-tmp/TTRV/outputs/physx_reliable20_soft_keep/movable_part_v2_step0_reliable20_keep`
- Machine-readable summary: `/root/autodl-tmp/TTRV/results/20260528_reliable20_soft_label_qtypes_keep/metrics.json`
