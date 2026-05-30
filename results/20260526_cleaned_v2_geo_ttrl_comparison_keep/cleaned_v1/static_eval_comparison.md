# Cleaned PhysX-MCQ Static Eval Comparison

- Predictions cache: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/fixed_step0_caches/combined_validation_predictions_step0.jsonl`
- Metrics: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/fixed_step0_caches/combined_metrics_step0.json`

## Overall

| split | n | acc | invalid | hitmax |
| --- | ---: | ---: | ---: | ---: |
| original retained step0 | 10572 | 0.335887 | 0.013999 | 0.039444 |
| cleaned step0 | 10572 | 0.397654 | 0.010878 | 0.099981 |
| delta |  | 0.061767 | -0.003121 | 0.060537 |

## By Question Type

| qtype | n | original acc | cleaned acc | delta | original invalid | cleaned invalid | cleaned hitmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 0.351393 | 0.419061 | 0.067668 | 0.009980 | 0.007787 | 0.106054 |
| movable_part | 1454 | 0.238652 | 0.263411 | 0.024759 | 0.039202 | 0.030261 | 0.061898 |

## Cleaning Reason

| reason | n | acc | invalid | hitmax |
| --- | ---: | ---: | ---: | ---: |
| joint_type_canonical_mutually_exclusive | 9118 | 0.419061 | 0.007787 | 0.106054 |
| movable_part_same_object_distractors | 1454 | 0.263411 | 0.030261 | 0.061898 |

## Joint Original Conflict Reason

| conflict | n | acc | invalid | hitmax |
| --- | ---: | ---: | ---: | ---: |
| compound_prismatic_revolute_vs_single | 5497 | 0.431326 | 0.007822 | 0.104057 |
| hinge_vs_revolute | 5293 | 0.439259 | 0.009069 | 0.106745 |
| rigid_vs_no_constraint | 3150 | 0.370159 | 0.006984 | 0.105397 |

## Movable Same-Object Distractor Count

| bucket | n | acc | invalid | hitmax |
| --- | ---: | ---: | ---: | ---: |
| 3 | 561 | 0.278075 | 0.023173 | 0.055258 |
| 4-7 | 775 | 0.267097 | 0.034839 | 0.059355 |
| 8+ | 118 | 0.169492 | 0.033898 | 0.110169 |
