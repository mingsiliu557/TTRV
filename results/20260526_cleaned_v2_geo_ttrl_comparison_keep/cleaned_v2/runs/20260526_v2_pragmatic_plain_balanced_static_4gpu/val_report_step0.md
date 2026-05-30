# PointLLM PhysX-MCQ baseline val step0 train=-1 val=-1

- Vote records: 8365
- Correct vote records: 4022
- Vote accuracy: 0.480813
- Invalid vote outputs: 675
- Hit max response length outputs: 482
- Mean response token length: 15.841841
- Prompt groups: 8365
- Mean votes per group: 1.000000

## TTRV-Style Vote Metrics

| metric | value |
| --- | ---: |
| response_accuracy | 0.480813 |
| acc_at_1 | 0.480813 |
| majority_accuracy | 0.480813 |
| pass_at_votes | 0.480813 |
| best_accuracy_at_votes | 0.480813 |
| worst_accuracy_at_votes | 0.480813 |
| vote_invalid_rate | 0.080693 |
| group_invalid_rate | 0.080693 |
| frequency_mean | 1.000000 |
| normalized_entropy_mean | 0.000000 |
| raw_entropy_mean | -0.000000 |
| ttrv_reward_mean | 0.838613 |
| reward_mean | 0.480813 |
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
| hit_max_response_length_rate | 0.057621 |
| response_token_len_mean | 15.841841 |
| response_token_len_max | 24.000000 |

## Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 3324 |
| B | 1901 |
| C | 1932 |
| D | 533 |
| unknown | 675 |

## TTRL Pseudo Label Distribution

| pseudo_label | count |
| --- | ---: |
| A | 3324 |
| B | 1901 |
| C | 1932 |
| D | 533 |
| none | 675 |

## Accuracy By Question Type

| question_type | vote_records | correct | vote_accuracy | invalid_outputs |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 7669 | 3841 | 0.500848 | 668 |
| movable_part | 696 | 181 | 0.260057 | 7 |

## TTRV Metrics By Question Type

| question_type | groups | votes/group | response_acc | acc@1 | majority_acc | pass@votes | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_type | 7669 | 1.000000 | 0.500848 | 0.500848 | 0.500848 | 0.500848 | 0.087104 |
| movable_part | 696 | 1.000000 | 0.260057 | 0.260057 | 0.260057 | 0.260057 | 0.010057 |
