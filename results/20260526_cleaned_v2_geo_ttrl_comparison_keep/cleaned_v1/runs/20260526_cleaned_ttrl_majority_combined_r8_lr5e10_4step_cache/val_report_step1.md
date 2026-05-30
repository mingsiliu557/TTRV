# PointLLM PhysX-MCQ adapt val step1 train=-1 val=-1

- Vote records: 10572
- Correct vote records: 4087
- Vote accuracy: 0.386587
- Invalid vote outputs: 106
- Hit max response length outputs: 1080
- Mean response token length: 17.653708
- Prompt groups: 10572
- Mean votes per group: 1.000000

## TTRV-Style Vote Metrics

| metric | value |
| --- | ---: |
| response_accuracy | 0.386587 |
| acc_at_1 | 0.386587 |
| majority_accuracy | 0.386587 |
| pass_at_votes | 0.386587 |
| best_accuracy_at_votes | 0.386587 |
| worst_accuracy_at_votes | 0.386587 |
| vote_invalid_rate | 0.010026 |
| group_invalid_rate | 0.010026 |
| frequency_mean | 1.000000 |
| normalized_entropy_mean | 0.000000 |
| raw_entropy_mean | -0.000000 |
| ttrv_reward_mean | 0.989974 |
| reward_mean | 0.386587 |
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
| hit_max_response_length_rate | 0.102157 |
| response_token_len_mean | 17.653708 |
| response_token_len_max | 24.000000 |

## Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 1875 |
| B | 1987 |
| C | 2302 |
| D | 4302 |
| unknown | 106 |

## TTRL Pseudo Label Distribution

| pseudo_label | count |
| --- | ---: |
| A | 1875 |
| B | 1987 |
| C | 2302 |
| D | 4302 |
| none | 106 |

## Accuracy By Question Type

| question_type | vote_records | correct | vote_accuracy | invalid_outputs |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 3705 | 0.406339 | 68 |
| movable_part | 1454 | 382 | 0.262724 | 38 |

## TTRV Metrics By Question Type

| question_type | groups | votes/group | response_acc | acc@1 | majority_acc | pass@votes | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_type | 9118 | 1.000000 | 0.406339 | 0.406339 | 0.406339 | 0.406339 | 0.007458 |
| movable_part | 1454 | 1.000000 | 0.262724 | 0.262724 | 0.262724 | 0.262724 | 0.026135 |
