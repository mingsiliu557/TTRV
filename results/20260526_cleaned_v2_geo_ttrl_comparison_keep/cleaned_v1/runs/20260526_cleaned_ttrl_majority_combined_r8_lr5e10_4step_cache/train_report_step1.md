# PointLLM PhysX-MCQ adapt train step1 train=-1 val=-1

- Vote records: 32
- Correct vote records: 13
- Vote accuracy: 0.406250
- Invalid vote outputs: 0
- Hit max response length outputs: 5
- Mean response token length: 19.500000
- Prompt groups: 4
- Mean votes per group: 8.000000

## TTRV-Style Vote Metrics

| metric | value |
| --- | ---: |
| response_accuracy | 0.406250 |
| acc_at_1 | n/a |
| majority_accuracy | 0.250000 |
| pass_at_votes | 1.000000 |
| best_accuracy_at_votes | 1.000000 |
| worst_accuracy_at_votes | 0.000000 |
| vote_invalid_rate | 0.000000 |
| group_invalid_rate | 0.000000 |
| frequency_mean | 0.476562 |
| normalized_entropy_mean | 0.813427 |
| raw_entropy_mean | 0.903676 |
| ttrv_reward_mean | 0.375000 |
| reward_mean | 0.375000 |
| unique_pred_ratio_mean | 0.375000 |
| group_unknown_rate_mean | 0.000000 |
| ttrl_majority_count_mean | 4.500000 |
| ttrl_majority_ratio_mean | 0.562500 |
| ttrl_majority_tie_mean | 0.500000 |
| ttrl_max_majority_ratio_mean | 1.000000 |
| geo_harmony_hm_mean | 0.000000 |
| geo_original_majority_ratio_mean | 0.000000 |
| geo_original_majority_tie_mean | 0.000000 |
| geo_view_support_mean | 0.000000 |
| geo_skipped_rate | 0.000000 |

## Response Stats

| metric | value |
| --- | ---: |
| hit_max_response_length_rate | 0.156250 |
| response_token_len_mean | 19.500000 |
| response_token_len_max | 24.000000 |

## Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 5 |
| B | 10 |
| C | 8 |
| D | 9 |

## TTRL Pseudo Label Distribution

| pseudo_label | count |
| --- | ---: |
| B | 8 |
| C | 8 |
| none | 16 |

## Accuracy By Question Type

| question_type | vote_records | correct | vote_accuracy | invalid_outputs |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 32 | 13 | 0.406250 | 0 |

## TTRV Metrics By Question Type

| question_type | groups | votes/group | response_acc | acc@1 | majority_acc | pass@votes | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_type | 4 | 8.000000 | 0.406250 | n/a | 0.250000 | 1.000000 | 0.000000 |
