# PhysX-MCQ TTRL Majority Analysis

- Run status: `0`
- Early stop: `not_configured`
- Weight recovery: `not_configured`
- Full-run gate: `passed`
- Best validation step: `0`

## Validation Steps

| step | acc | invalid | hit_max | resp_len_mean | joint_type | movable_part | object_category |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.397654 | 115 | 0.099981 | 17.663640 | 0.419061 | 0.263411 | n/a |

## Training Pseudo Labels

| step | response_acc | majority_acc | pass@votes | invalid_rate | ttrl_reward_mean | majority_ratio_mean | tie_rate | geo_hm | geo_support | geo_skip |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Latest Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 1925 |
| B | 2058 |
| C | 2362 |
| D | 4112 |
| unknown | 115 |

## Early Stop Details

```json
{
  "triggered": false,
  "reason": "not_configured"
}
```
