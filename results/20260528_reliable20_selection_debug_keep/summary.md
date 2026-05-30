# Reliable20 Selection Debug Summary

Date: 2026-05-28

## Goal

Check whether the failing `object_category` and `movable_part` soft-label Geo-Harmony runs can be improved by replacing only the 20 adaptation samples. The new selector keeps step-0-correct reliable samples, but adds `diverse_text` preference so the selected answer texts and objects are less repetitive.

Common setup:

- `REWARD_VARIANT=geo_harmony_soft`
- `GEO_SOFT_GAMMA=2.0`
- `GEO_MIN_HM=0.60`
- `ROLLOUT_N=8`, `N_VOTES_PER_PROMPT=8`
- `ACTOR_LR=5e-10`, `USE_KL_LOSS=True`, `KL_LOSS_COEF=0.3`
- full validation, deterministic `acc@1`, 4 GPUs

## Main Results

| task | selection | step0 acc@1 | best acc@1 | best step | delta |
| --- | --- | ---: | ---: | ---: | ---: |
| object_category original | old reliable20 | 0.395107 | 0.395107 | 0 | +0.000000 |
| object_category original | diverse_text, early stop | 0.395107 | 0.395107 | 0 | +0.000000 |
| object_category original | diverse_text, no early stop | 0.395107 | 0.395107 | 0 | +0.000000 |
| movable_part v2 | old reliable20 | 0.260057 | 0.260057 | 0 | +0.000000 |
| movable_part v2 | diverse_text | 0.260057 | 0.265805 | 3 | +0.005747 |

## Object Category No-Early-Stop Check

The extra object run disabled early stop/recovery and ran 3 training steps:

| step | acc@1 | invalid rate | hit max rate |
| ---: | ---: | ---: | ---: |
| 0 | 0.395107 | 0.016039 | 0.033139 |
| 1 | 0.392018 | 0.013168 | 0.030642 |
| 2 | 0.389553 | 0.008893 | 0.030018 |
| 3 | 0.394233 | 0.012201 | 0.033420 |

Step 3 partially recovered, but still did not exceed step 0. This suggests the object failure is not just an over-aggressive early stop; under the current reliable20 + soft Geo-Harmony setup, later steps do not produce a real acc@1 gain.

Training diagnostics for object no-early-stop:

| step | response acc | Pass@8 | Majority@8 | soft selected | GT mass | HM mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.625000 | 1.000000 | 1.000000 | 0.500000 | 0.816038 | 0.583037 |
| 2 | 0.500000 | 1.000000 | 0.500000 | 0.250000 | 0.500000 | 0.485120 |
| 3 | 0.468750 | 1.000000 | 0.500000 | 0.500000 | 0.500000 | 0.458037 |

Interpretation: even when Pass@8 is high on the 20 adaptation prompts, the signal does not transfer to the full object validation set. The model is sharpening local category priors from a very small adaptation set, especially toward frequent or visually similar categories, instead of learning a robust object classifier.

## Movable Part Result

The same `diverse_text` replacement helped `movable_part_v2` slightly:

| step | acc@1 | invalid rate | hit max rate |
| ---: | ---: | ---: | ---: |
| 0 | 0.260057 | 0.010057 | 0.047414 |
| 1 | 0.261494 | 0.014368 | 0.041667 |
| 2 | 0.261494 | 0.012931 | 0.047414 |
| 3 | 0.265805 | 0.010057 | 0.038793 |
| 4 | 0.262931 | 0.008621 | 0.035920 |
| 5 | 0.261494 | 0.007184 | 0.038793 |

Best step is step 3: `+0.005747` absolute acc@1, about `+0.57 pp`. This is small but positive, and better than the old movable reliable20 run, which dropped below baseline.

Training diagnostics for movable diverse_text:

| step | response acc | Pass@8 | Majority@8 | soft selected | GT mass | HM mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.718750 | 1.000000 | 1.000000 | 0.750000 | 0.900000 | 0.714287 |
| 2 | 0.468750 | 0.750000 | 0.750000 | 0.500000 | 0.463018 | 0.510120 |
| 3 | 0.500000 | 1.000000 | 0.500000 | 0.750000 | 0.500000 | 0.587501 |
| 4 | 0.406250 | 0.750000 | 0.500000 | 0.500000 | 0.486923 | 0.572620 |
| 5 | 0.531250 | 1.000000 | 0.750000 | 0.500000 | 0.717961 | 0.550001 |

## Conclusion

Replacing the 20-sample selection can help when the selected samples cover more useful movable-part labels. It fixed part of the old movable collapse and produced a small positive acc@1 gain.

For `object_category`, selection alone is not enough. High Pass@8 on the adaptation prompts only means the model can sample a correct answer locally; it does not guarantee that the pseudo-label distribution is useful for global adaptation. The no-early-stop test confirms that later steps do not recover above step 0.

Practical next direction:

1. Keep `diverse_text` for movable-part experiments because it is currently the best movable selection.
2. For object-category, avoid reporting a gain under this setup; use it as a negative/diagnostic case.
3. If continuing object-category, sample selection should be made more distribution-aware, not just answer-letter balanced. Candidate constraints should penalize overrepresented categories such as table/lamp/headphone-style labels and explicitly cover category clusters.

## Artifacts

- Metrics JSON: `/root/autodl-tmp/TTRV/results/20260528_reliable20_selection_debug_keep/metrics.json`
- Object no-early-stop run: `/root/autodl-tmp/TTRV/outputs/physx_reliable20_soft_keep/runs/20260528_object_original_reliable20_diverse_text_geo_harmony_soft_noearly_3step_b4_keep`
- Movable diverse_text run: `/root/autodl-tmp/TTRV/outputs/physx_reliable20_soft_keep/runs/20260528_movable_v2_reliable20_diverse_text_geo_harmony_soft_r8_lr5e10_gamma2_5step_b4_keep`
