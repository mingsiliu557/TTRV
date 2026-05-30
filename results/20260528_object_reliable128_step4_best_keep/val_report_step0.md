# PointLLM PhysX-MCQ adapt val step0 train=20 val=20

- Vote records: 32047
- Correct vote records: 12662
- Vote accuracy: 0.395107
- Invalid vote outputs: 514
- Hit max response length outputs: 1062
- Mean response token length: 12.335195
- Prompt groups: 32047
- Mean votes per group: 1.000000

## TTRV-Style Vote Metrics

| metric | value |
| --- | ---: |
| response_accuracy | 0.395107 |
| acc_at_1 | 0.395107 |
| majority_accuracy | 0.395107 |
| pass_at_votes | 0.395107 |
| best_accuracy_at_votes | 0.395107 |
| worst_accuracy_at_votes | 0.395107 |
| vote_invalid_rate | 0.016039 |
| group_invalid_rate | 0.016039 |
| frequency_mean | 1.000000 |
| normalized_entropy_mean | 0.000000 |
| raw_entropy_mean | -0.000000 |
| ttrv_reward_mean | 0.000000 |
| reward_mean | 0.395107 |
| unique_pred_ratio_mean | 1.000000 |
| group_unknown_rate_mean | 0.000000 |
| ttrl_majority_count_mean | 0.000000 |
| ttrl_majority_ratio_mean | 0.000000 |
| ttrl_majority_tie_mean | 0.000000 |
| ttrl_max_majority_ratio_mean | 0.000000 |
| geo_harmony_hm_mean | 0.000000 |
| geo_original_majority_ratio_mean | 0.000000 |
| geo_original_majority_tie_mean | 0.000000 |
| geo_view_support_mean | 0.000000 |
| geo_skipped_rate | 0.000000 |
| geo_soft_selected_rate | 0.000000 |
| geo_soft_top_prob_mean | 0.000000 |
| geo_soft_known_count_mean | 0.000000 |
| geo_soft_gt_mass_mean | 0.000000 |
| geo_soft_view_support_mean | 0.000000 |

## Response Stats

| metric | value |
| --- | ---: |
| hit_max_response_length_rate | 0.033139 |
| response_token_len_mean | 12.335195 |
| response_token_len_max | 24.000000 |

## Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 18735 |
| B | 3450 |
| C | 3377 |
| D | 5971 |
| unknown | 514 |

## TTRL Pseudo Label Distribution

| pseudo_label | count |
| --- | ---: |
| none | 32047 |

## Accuracy By Question Type

| question_type | vote_records | correct | vote_accuracy | invalid_outputs |
| --- | ---: | ---: | ---: | ---: |
| object_category | 32047 | 12662 | 0.395107 | 514 |

## TTRV Metrics By Question Type

| question_type | groups | votes/group | response_acc | acc@1 | majority_acc | pass@votes | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| object_category | 32047 | 1.000000 | 0.395107 | 0.395107 | 0.395107 | 0.395107 | 0.016039 |
