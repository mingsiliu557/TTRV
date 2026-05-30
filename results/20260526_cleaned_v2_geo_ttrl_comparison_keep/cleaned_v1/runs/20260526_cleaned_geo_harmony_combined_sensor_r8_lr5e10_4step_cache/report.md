# PointLLM PhysX-MCQ adapt train=-1 val=-1

- Vote records: 10572
- Correct vote records: 4121
- Vote accuracy: 0.389803
- Invalid vote outputs: 107
- Hit max response length outputs: 1104
- Mean response token length: 17.646614
- Prompt groups: 10572
- Mean votes per group: 1.000000

## TTRV-Style Vote Metrics

| metric | value |
| --- | ---: |
| response_accuracy | 0.389803 |
| acc_at_1 | 0.389803 |
| majority_accuracy | 0.389803 |
| pass_at_votes | 0.389803 |
| best_accuracy_at_votes | 0.389803 |
| worst_accuracy_at_votes | 0.389803 |
| vote_invalid_rate | 0.010121 |
| group_invalid_rate | 0.010121 |
| frequency_mean | 1.000000 |
| normalized_entropy_mean | 0.000000 |
| raw_entropy_mean | -0.000000 |
| ttrv_reward_mean | 0.000000 |
| reward_mean | 0.389803 |
| unique_pred_ratio_mean | 1.000000 |
| group_unknown_rate_mean | 0.000000 |
| ttrl_majority_count_mean | 0.000000 |
| ttrl_majority_ratio_mean | 0.000000 |
| ttrl_majority_tie_mean | 0.000000 |
| ttrl_max_majority_ratio_mean | 1.000000 |
| geo_harmony_hm_mean | 0.000000 |
| geo_original_majority_ratio_mean | 0.000000 |
| geo_original_majority_tie_mean | 0.000000 |
| geo_view_support_mean | 0.000000 |
| geo_skipped_rate | 0.000000 |

## Response Stats

| metric | value |
| --- | ---: |
| hit_max_response_length_rate | 0.104427 |
| response_token_len_mean | 17.646614 |
| response_token_len_max | 24.000000 |

## Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 1888 |
| B | 1983 |
| C | 2308 |
| D | 4286 |
| unknown | 107 |

## TTRL Pseudo Label Distribution

| pseudo_label | count |
| --- | ---: |
| none | 10572 |

## Accuracy By Question Type

| question_type | vote_records | correct | vote_accuracy | invalid_outputs |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 3740 | 0.410178 | 69 |
| movable_part | 1454 | 381 | 0.262036 | 38 |

## TTRV Metrics By Question Type

| question_type | groups | votes/group | response_acc | acc@1 | majority_acc | pass@votes | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 1.000000 | 0.410178 | 0.410178 | 0.410178 | 0.410178 | 0.007567 |
| movable_part | 1454 | 1.000000 | 0.262036 | 0.262036 | 0.262036 | 0.262036 | 0.026135 |
