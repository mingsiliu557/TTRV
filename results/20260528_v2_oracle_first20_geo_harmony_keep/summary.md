# V2 Oracle First-20 Geo-Harmony Diagnostic

## Material Passport
- Skill: academic-research-suite / experiment-agent
- Date: 2026-05-28 UTC
- Run dir: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates_keep/runs/20260528_v2_oracle_semantic_correct20_first_geo_harmony_sensor_r8_lr5e10_5step_b4_keep`
- Dataset dir: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates_keep/pragmatic_plain_balanced/parquet_combined_step0_oracle_semantic_correct20_balanced_first_keep`
- Model/input: local PointLLM 7B v1.2 cache; OBJ full-object point cloud; POINTNUM=8192; unit-sphere normalized XYZ + zero RGB.
- Validation set: cleaned v2 `joint_type + movable_part`, n=8365; validation is deterministic acc@1 (`VAL_ROLLOUT_N=1`, `VAL_DO_SAMPLE=False`, `temperature=0`).

## Hypothesis Tested
TTRL/Geo-Harmony behaves like sharpening of the model sampling distribution. If the first 20 adaptation prompts are already correct and confident at static step0, the pseudo-label signal should be less noisy and adaptation should have a better chance to improve acc@1.

This run is an oracle diagnostic: the first 20 were selected using step0 ground-truth correctness, non-invalid/non-hitmax status, short/clean responses, qtype balance, answer balance, and a simple semantic contradiction filter. It is useful as an upper bound for curriculum quality, but it is not deployable without replacing GT correctness with a non-oracle confidence proxy.

## Parameters Held Fixed
- `NO_GPU=4`, `RUN_MODE=adapt`, `TRAIN_BATCH_SIZE=4`, `DEBUG_STEPS=5`, `DATA_SHUFFLE=False`.
- `ROLLOUT_N=8`, `N_SAMPLES_PER_PROMPT=8`, `N_VOTES_PER_PROMPT=8`.
- `REWARD_VARIANT=geo_harmony`, `GEO_NUM_VIEWS=2`, `GEO_SAMPLES_PER_VIEW=4`, `GEO_MIN_VIEW_SUPPORT=2`, `GEO_MIN_HM=0.60`.
- `POINT_REFRAME_POLICY=sensor_noise`; `ACTOR_LR=5e-10`; KL enabled with coef `0.3`.
- Step0 validation cache reused from the cleaned v2 static run.

## Full Validation Results
| step | acc@1 | delta vs step0 | invalid | hitmax | joint acc | movable acc | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.500538 | +0.000000 | 0.015780 | 0.057621 | 0.522363 | 0.260057 |  |
| 1 | 0.503766 | +0.003228 | 0.013270 | 0.059414 | 0.525753 | 0.261494 |  |
| 2 | 0.501494 | +0.000956 | 0.013030 | 0.057621 | 0.523667 | 0.257184 |  |
| 3 | 0.507113 | +0.006575 | 0.012672 | 0.050807 | 0.529795 | 0.257184 | best |
| 4 | 0.505200 | +0.004662 | 0.012313 | 0.050926 | 0.527579 | 0.258621 |  |
| 5 | 0.500179 | -0.000359 | 0.009922 | 0.052959 | 0.523015 | 0.248563 | early stop; restored step3 |

Best step is `3` with acc@1 `0.507113`, a gain of `+0.006575` absolute over static step0. The gain is mainly from `joint_type` (`0.529795` vs `0.522363`, delta `+0.007433`). `movable_part` did not improve at best step (`0.257184` vs `0.260057`, delta `-0.002874`).

## Adapt-Batch Pass@8 And Majority@8
| step | groups | response acc | Pass@8 | Majority@8 | selected groups | selected precision | joint Pass/Maj | movable Pass/Maj |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.688 | 1.000 | 0.750 | 3 | 1.000 | 1.000/0.500 | 1.000/1.000 |
| 2 | 4 | 0.406 | 1.000 | 0.250 | 3 | 0.333 | 1.000/0.000 | 1.000/0.500 |
| 3 | 4 | 0.625 | 1.000 | 1.000 | 2 | 1.000 | 1.000/1.000 | 1.000/1.000 |
| 4 | 4 | 0.469 | 1.000 | 1.000 | 0 | NA | 1.000/1.000 | 1.000/1.000 |
| 5 | 4 | 0.531 | 1.000 | 0.500 | 3 | 0.667 | 1.000/0.000 | 1.000/1.000 |

The important pattern is that Pass@8 stays at 1.0 on all five adapt batches, while Majority@8 swings from 0.75 to 0.25 to 1.0 to 1.0 to 0.5. This means PointLLM often samples at least one correct answer, but the hard pseudo-label/majority choice is not reliably selecting it. Step2 is the clearest bad case: Pass@8=1.0 but Majority@8=0.25, and the next validation result barely improves over step0.

## Comparison To Previous Cleaned V2 Runs
| run | best step | best acc@1 | delta vs static step0 | comment |
| --- | ---: | ---: | ---: | --- |
| static cleaned v2 | 0 | 0.500538 | +0.000000 | baseline |
| loose Geo-Harmony HM0.60 | 1 | 0.501614 | +0.001076 | small unstable drift |
| strict Geo-Harmony view0.75 | 1 | 0.503288 | +0.002750 | collapsed by step2 |
| oracle semantic-correct first20 Geo-Harmony | 3 | 0.507113 | +0.006575 | best current v2 result; oracle diagnostic |

## Interpretation
This supports the hypothesis in a limited, diagnostic sense: making the first 20 adaptation prompts cleaner raises the best observed cleaned-v2 Geo-Harmony result from the previous 0.503288 to 0.507113. The result is still small and not uniformly beneficial: `joint_type` improves, while `movable_part` is flat or worse.

Why can Majority@8 be high while acc@1 remains low? Majority@8 here is measured only on the current 4-prompt adaptation batch with stochastic sampling; acc@1 is deterministic full validation over 8365 examples after a weight update. A correct majority on a tiny, oracle-clean batch does not guarantee that the update moves the global model in a direction that helps all validation prompts. Also, when Pass@8 is high but Majority@8 or selected precision drops, hard-label sharpening can amplify a wrong local mode.

## Method Implication
The current hard-label reward is the likely weak link. A better next method should avoid putting all reward mass on one majority answer. Use a soft pseudo-label distribution from vote counts, optionally filtered by non-oracle confidence: low entropy, no unknown/hitmax, high cross-view agreement, semantic consistency between raw text and option text, and qtype-specific thresholds. This matches the earlier idea of replacing hard label reward with vote-count-weighted soft label reward.

Pseudo-code for the deployable version:

```text
for prompt x in adaptation_stream:
    samples = generate K answers under point-view/noise variants
    parsed = parse each raw answer into A/B/C/D/unknown
    if unknown_rate high or hitmax_rate high: skip update for x
    vote_dist = normalized counts over A/B/C/D
    if entropy(vote_dist) too high or cross_view_agreement too low: skip or downweight x
    optional: reject samples whose verbalized answer contradicts selected option text
    reward(answer=a) = soft_weight(vote_dist[a]) - entropy_penalty
    update with small LR/KL, select best checkpoint only by validation acc@1
```

## Next Ablation
Run a non-oracle first20 selector using only step0 observables: high vote agreement, low entropy, no unknown/hitmax, short parseable responses, and semantic consistency. If this keeps most of the +0.0066 best-step gain, the curriculum idea is practically useful. If not, the gain is mostly oracle leakage and the priority should shift to soft-label reward.
