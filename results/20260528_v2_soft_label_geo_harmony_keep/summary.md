# Soft-label Geo-Harmony / RESTRAIN-lite Result

## What Changed
- Implemented the requested soft reward path through existing `REWARD_VARIANT=geo_harmony_soft`.
- `geo_score(a)` is still harmonic mean over per-view probabilities, but the probability denominator now uses all `M` samples in that view, so `unknown` reduces confidence instead of disappearing from the denominator.
- `q_geo(a) = normalize(geo_score(a)^gamma)` with `GEO_SOFT_GAMMA=2.0`. Rollout reward is `q_geo(prediction_i)`; unknown gets `UNKNOWN_REWARD=0.0`.
- `GEO_MIN_HM=0.60` now gates the raw max `geo_score`, matching the proposed “skip if max geo_score below threshold” behavior.

## Paper Alignment
- This matches the RESTRAIN direction: replace brittle majority pseudo-labeling with pseudo-label weighting over the whole sampled answer distribution.
- It is not full RESTRAIN. The paper also has negative/self-penalization for low-consistency examples and prompt-level weighting; this run only tests the pseudo-label weighting part, plus our Geo-Harmony cross-view score.
- Our shaping function is `score^gamma`; the paper describes a generic shaped frequency weight, with Gaussian-style shaping in its formulation. This is a simpler MCQ-compatible ablation.

## Run Setup
- Run dir: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates_keep/runs/20260528_v2_oracle_semantic_correct20_first_geo_harmony_soft_sensor_r8_lr5e10_gamma2_5step_b4_keep`
- Dataset: cleaned v2 oracle semantic-correct first20, `joint_type + movable_part`, n=8365.
- Parameters held aligned with the hard-label run: 4 GPU, `ROLLOUT_N=8`, `GEO_NUM_VIEWS=2`, `GEO_SAMPLES_PER_VIEW=4`, `GEO_MIN_VIEW_SUPPORT=2`, `GEO_MIN_HM=0.60`, `ACTOR_LR=5e-10`, `KL_LOSS_COEF=0.3`, full deterministic validation after each step.

## Validation
| step | acc@1 | delta vs step0 | invalid | hitmax | joint acc | movable acc | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.500538 | +0.000000 | 0.015780 | 0.057621 | 0.522363 | 0.260057 |  |
| 1 | 0.503527 | +0.002989 | 0.014704 | 0.058338 | 0.525883 | 0.257184 |  |
| 2 | 0.502929 | +0.002391 | 0.012074 | 0.057621 | 0.524710 | 0.262931 |  |
| 3 | 0.510102 | +0.009564 | 0.012313 | 0.049731 | 0.532664 | 0.261494 | best; restored target |
| 4 | 0.505559 | +0.005021 | 0.011596 | 0.048057 | 0.528231 | 0.255747 | early stop; recovered to step3 |

Best soft-label step is `3` with acc@1 `0.510102`: `+0.009564` over static step0 and `+0.002989` over hard-label best step3.
At best step, joint_type improves `+0.010301` over step0; movable_part improves `+0.001437` over step0, though movable remains weak overall.

## Adapt Batch Diagnostics
| step | response acc | Pass@8 | Majority@8 | soft selected rate | soft GT mass | reward mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.500 | 1.000 | 0.250 | 0.250 | 0.715 | 0.219 |
| 2 | 0.562 | 1.000 | 0.750 | 0.250 | 0.646 | 0.250 |
| 3 | 0.531 | 1.000 | 0.750 | 0.750 | 0.636 | 0.504 |
| 4 | 0.531 | 1.000 | 0.750 | 0.750 | 0.635 | 0.550 |

Key point: step1 had `Pass@8=1.0` but `Majority@8=0.25`; hard majority would be fragile there. Soft reward still produced a positive validation delta at step1. The best jump happened at step3, where `geo_soft_selected_rate=0.75` and validation acc reached `0.510102`.

## Comparison
| method | best step | best acc@1 | delta vs static |
| --- | ---: | ---: | ---: |
| static step0 | 0 | 0.500538 | +0.000000 |
| hard-label Geo-Harmony oracle-first20 | 3 | 0.507113 | +0.006575 |
| soft-label Geo-Harmony oracle-first20 | 3 | 0.510102 | +0.009564 |

## Conclusion
This supports your idea: weighting pseudo labels instead of collapsing onto majority improved the current best result. It is still an oracle-first20 diagnostic, so the next fair method test should remove oracle selection and use step0-only confidence to pick or weight adaptation prompts.
