# PhysX-MCQ TTRL Majority Analysis

- Run status: `0`
- Early stop: `not_configured`
- Weight recovery: `not_configured`
- Full-run gate: `passed`
- Best validation step: `0`

## Validation Steps

| step | acc | invalid | hit_max | resp_len_mean | joint_type | movable_part | object_category |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.480813 | 675 | 0.057621 | 15.841841 | 0.500848 | 0.260057 | n/a |

## Training Pseudo Labels

| step | response_acc | majority_acc | pass@votes | invalid_rate | ttrl_reward_mean | majority_ratio_mean | tie_rate | geo_hm | geo_support | geo_skip |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Latest Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 3324 |
| B | 1901 |
| C | 1932 |
| D | 533 |
| unknown | 675 |

## Early Stop Details

```json
{
  "triggered": false,
  "reason": "not_configured"
}
```
