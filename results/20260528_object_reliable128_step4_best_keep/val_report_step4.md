# PointLLM PhysX-MCQ adapt val step4 train=20 val=20

- Vote records: 32047
- Correct vote records: 12944
- Vote accuracy: 0.403907
- Invalid vote outputs: 498
- Hit max response length outputs: 1227
- Mean response token length: 12.864980
- Prompt groups: 32047
- Mean votes per group: 1.000000

## TTRV-Style Vote Metrics

| metric | value |
| --- | ---: |
| response_accuracy | 0.403907 |
| acc_at_1 | 0.403907 |
| majority_accuracy | 0.403907 |
| pass_at_votes | 0.403907 |
| best_accuracy_at_votes | 0.403907 |
| worst_accuracy_at_votes | 0.403907 |
| vote_invalid_rate | 0.015540 |
| group_invalid_rate | 0.015540 |
| frequency_mean | 1.000000 |
| normalized_entropy_mean | 0.000000 |
| raw_entropy_mean | -0.000000 |
| ttrv_reward_mean | 0.000000 |
| reward_mean | 0.403907 |
| unique_pred_ratio_mean | 1.000000 |
| group_unknown_rate_mean | 0.000000 |
| ttrl_majority_count_mean | 0.000000 |
| ttrl_majority_ratio_mean | 0.000000 |
| ttrl_majority_tie_mean | 0.000000 |
| ttrl_max_majority_ratio_mean | 1.000000 |
| geo_harmony_hm_mean | 0.000000 |
| geo_original_majority_ratio_mean | 1.000000 |
| geo_original_majority_tie_mean | 0.000000 |
| geo_view_support_mean | 0.000000 |
| geo_skipped_rate | 0.000000 |
| geo_soft_selected_rate | 0.000000 |
| geo_soft_top_prob_mean | 0.000000 |
| geo_soft_known_count_mean | 0.984460 |
| geo_soft_gt_mass_mean | 0.000000 |
| geo_soft_view_support_mean | 0.000000 |

## Response Stats

| metric | value |
| --- | ---: |
| hit_max_response_length_rate | 0.038288 |
| response_token_len_mean | 12.864980 |
| response_token_len_max | 24.000000 |

## Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 17909 |
| B | 5137 |
| C | 3542 |
| D | 4961 |
| unknown | 498 |

## TTRL Pseudo Label Distribution

| pseudo_label | count |
| --- | ---: |
| none | 32047 |

## Accuracy By Question Type

| question_type | vote_records | correct | vote_accuracy | invalid_outputs |
| --- | ---: | ---: | ---: | ---: |
| object_category | 32047 | 12944 | 0.403907 | 498 |

## TTRV Metrics By Question Type

| question_type | groups | votes/group | response_acc | acc@1 | majority_acc | pass@votes | invalid_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| object_category | 32047 | 1.000000 | 0.403907 | 0.403907 | 0.403907 | 0.403907 | 0.015540 |
