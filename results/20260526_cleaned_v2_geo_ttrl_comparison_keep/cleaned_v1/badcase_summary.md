# PhysX Cleaned Options V1 Audit

## Dataset

- Original rows: 44487
- Cleaned rows: 10572
- Output dir: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1`

## Question Types

| question_type | original | cleaned | skipped |
| --- | ---: | ---: | ---: |
| joint_type | 9118 | 9118 | 0 |
| movable_part | 3322 | 1454 | 1868 |
| object_category | 32047 | 0 | 32047 |

## Skipped Reasons

- movable_part_not_enough_same_object_distractors: 1868
- object_category_not_in_cleaned_v1: 32047

## Original Fixed Step0 On Retained IDs

| question_type | n | acc | invalid | hitmax |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 0.351393 | 0.009980 | 0.033012 |
| movable_part | 1454 | 0.238652 | 0.039202 | 0.079780 |

## Original Vote8 On Retained IDs

| question_type | groups | majority_acc | pass@8 | high_conf_wrong | high_conf_all |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 0.365760 | 0.713863 | 2348 | 3921 |
| movable_part | 1454 | 0.249656 | 0.676066 | 385 | 528 |

## Cleaning Reasons

- joint_type_canonical_mutually_exclusive: 9118
- movable_part_same_object_distractors: 1454
