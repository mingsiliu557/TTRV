# PhysX-MCQ TTRL Majority Analysis

- Run status: `0`
- Early stop: `accuracy_drop`
- Weight recovery: `restored`
- Full-run gate: `blocked`
- Best validation step: `0`

## Validation Steps

| step | acc | invalid | hit_max | resp_len_mean | joint_type | movable_part | object_category |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.397654 | 115 | 0.099981 | 17.663640 | 0.419061 | 0.263411 | n/a |
| 1 | 0.386587 | 106 | 0.102157 | 17.653708 | 0.406339 | 0.262724 | n/a |

## Training Pseudo Labels

| step | response_acc | majority_acc | pass@votes | invalid_rate | ttrl_reward_mean | majority_ratio_mean | tie_rate | geo_hm | geo_support | geo_skip |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.406250 | 0.250000 | 1.000000 | 0.000000 | 0.375000 | 0.562500 | 0.500000 | 0.000000 | 0.000000 | 0.000000 |

## Latest Prediction Distribution

| prediction | count |
| --- | ---: |
| A | 1875 |
| B | 1987 |
| C | 2302 |
| D | 4302 |
| unknown | 106 |

## Early Stop Details

```json
{
  "enabled": true,
  "triggered": true,
  "reason": "accuracy_drop",
  "step": 1,
  "baseline": {
    "acc": 0.3976541808550889,
    "invalid_rate": 0.010877790389708665,
    "hit_max_response_length_rate": 0.09998108210367007,
    "qtype_acc": {
      "joint_type": 0.4190611976310594,
      "movable_part": 0.26341127922971114
    }
  },
  "current": {
    "acc": 0.386587211502081,
    "invalid_rate": 0.0100264850548619,
    "hit_max_response_length_rate": 0.1021566401816118,
    "qtype_acc": {
      "joint_type": 0.4063391094538276,
      "movable_part": 0.2627235213204952
    }
  },
  "thresholds": {
    "acc_drop": 0.004,
    "invalid_increase": 0.008,
    "hitmax_increase": 0.01,
    "best_acc_drop": 0.003,
    "patience": 0,
    "min_delta": 0.0,
    "qtype_acc_drop": 0.02,
    "qtype_drop_count": 1
  },
  "state": {
    "best_acc": 0.3976541808550889,
    "best_step": 0,
    "stale_count": 1,
    "history": [
      {
        "step": 0,
        "snapshot": {
          "acc": 0.3976541808550889,
          "invalid_rate": 0.010877790389708665,
          "hit_max_response_length_rate": 0.09998108210367007,
          "qtype_acc": {
            "joint_type": 0.4190611976310594,
            "movable_part": 0.26341127922971114
          }
        },
        "best": true
      },
      {
        "step": 1,
        "snapshot": {
          "acc": 0.386587211502081,
          "invalid_rate": 0.0100264850548619,
          "hit_max_response_length_rate": 0.1021566401816118,
          "qtype_acc": {
            "joint_type": 0.4063391094538276,
            "movable_part": 0.2627235213204952
          }
        },
        "best": false,
        "best_acc": 0.3976541808550889,
        "best_step": 0,
        "stale_count": 1
      }
    ]
  },
  "qtype_drops": [],
  "collapse_step": 1,
  "recovery": {
    "enabled": true,
    "restored": true,
    "trigger_reason": "accuracy_drop",
    "restored_from_step": 0,
    "restored_at_step": 1,
    "checkpoint_path": "/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/runs/20260526_cleaned_ttrl_majority_combined_r8_lr5e10_4step_cache/checkpoints/global_step_0",
    "best": {
      "step": 0,
      "metric": 0.3976541808550889,
      "snapshot": {
        "acc": 0.3976541808550889,
        "invalid_rate": 0.010877790389708665,
        "hit_max_response_length_rate": 0.09998108210367007,
        "qtype_acc": {
          "joint_type": 0.4190611976310594,
          "movable_part": 0.26341127922971114
        }
      },
      "checkpoint_path": "/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/runs/20260526_cleaned_ttrl_majority_combined_r8_lr5e10_4step_cache/checkpoints/global_step_0"
    },
    "history": [
      {
        "step": 0,
        "metric": 0.3976541808550889,
        "snapshot": {
          "acc": 0.3976541808550889,
          "invalid_rate": 0.010877790389708665,
          "hit_max_response_length_rate": 0.09998108210367007,
          "qtype_acc": {
            "joint_type": 0.4190611976310594,
            "movable_part": 0.26341127922971114
          }
        }
      }
    ]
  }
}
```

## Weight Recovery Details

```json
{
  "enabled": true,
  "restored": true,
  "trigger_reason": "accuracy_drop",
  "restored_from_step": 0,
  "restored_at_step": 1,
  "checkpoint_path": "/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/runs/20260526_cleaned_ttrl_majority_combined_r8_lr5e10_4step_cache/checkpoints/global_step_0",
  "best": {
    "step": 0,
    "metric": 0.3976541808550889,
    "snapshot": {
      "acc": 0.3976541808550889,
      "invalid_rate": 0.010877790389708665,
      "hit_max_response_length_rate": 0.09998108210367007,
      "qtype_acc": {
        "joint_type": 0.4190611976310594,
        "movable_part": 0.26341127922971114
      }
    },
    "checkpoint_path": "/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1/runs/20260526_cleaned_ttrl_majority_combined_r8_lr5e10_4step_cache/checkpoints/global_step_0"
  },
  "history": [
    {
      "step": 0,
      "metric": 0.3976541808550889,
      "snapshot": {
        "acc": 0.3976541808550889,
        "invalid_rate": 0.010877790389708665,
        "hit_max_response_length_rate": 0.09998108210367007,
        "qtype_acc": {
          "joint_type": 0.4190611976310594,
          "movable_part": 0.26341127922971114
        }
      }
    }
  ]
}
```
