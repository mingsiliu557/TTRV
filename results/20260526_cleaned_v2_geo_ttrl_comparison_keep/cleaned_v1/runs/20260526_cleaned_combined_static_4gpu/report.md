# PointLLM PhysX-MCQ baseline train=-1 val=-1

- Vote records: 10572
- Correct vote records: 4204
- Vote accuracy: 0.397654
- Invalid vote outputs: 115
- Hit max response length outputs: 1057
- Mean response token length: 17.663640
- Prompt groups: 10572
- Mean votes per group: 1.000000

## TTRV-Style Vote Metrics

| metric | value |
| --- | ---: |
| response_accuracy | 0.397654 |
| acc_at_1 | 0.397654 |
| majority_accuracy | 0.397654 |
| pass_at_votes | 0.397654 |
| best_accuracy_at_votes | 0.397654 |
| worst_accuracy_at_votes | 0.397654 |
| vote_invalid_rate | 0.010878 |
| group_invalid_rate | 0.010878 |
| frequency_mean | 1.000000 |
| normalized_entropy_mean | 0.000000 |
| raw_entropy_mean | -0.000000 |
| ttrv_reward_mean | 0.989122 |
| reward_mean | 0.397654 |
| unique_pred_ratio_mean | 1.000000 |
| group_unknown_rate_mean | 0.000000 |
| ttrl_majority_count_mean | 1.000000 |
| ttrl_majority_ratio_mean | 1.000000 |
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
| hit_max_response_length_rate | 0.099981 |
| response_token_len_mean | 17.663640 |
| response_token_len_max | 24.000000 |

## Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 1925 |
| B | 2058 |
| C | 2362 |
| D | 4112 |
| unknown | 115 |

## TTRL Pseudo Label Distribution

| pseudo_label | count |
| --- | ---: |
| A | 1925 |
| B | 2058 |
| C | 2362 |
| D | 4112 |
| none | 115 |

## Accuracy By Question Type

| question_type | vote_records | correct | vote_accuracy | invalid_outputs |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 3821 | 0.419061 | 71 |
| movable_part | 1454 | 383 | 0.263411 | 44 |

## TTRV Metrics By Question Type

| question_type | groups | votes/group | response_acc | acc@1 | majority_acc | pass@votes | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 1.000000 | 0.419061 | 0.419061 | 0.419061 | 0.419061 | 0.007787 |
| movable_part | 1454 | 1.000000 | 0.263411 | 0.263411 | 0.263411 | 0.263411 | 0.030261 |
