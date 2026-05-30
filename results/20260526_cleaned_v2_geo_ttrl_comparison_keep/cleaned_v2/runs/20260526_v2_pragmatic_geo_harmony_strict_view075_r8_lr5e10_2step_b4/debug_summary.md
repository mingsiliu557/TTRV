# Geo-Harmony Strict Debug Summary

Run: `/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates/runs/20260526_v2_pragmatic_geo_harmony_strict_view075_r8_lr5e10_2step_b4`

| step | acc@1 | invalid | hit_max | joint_acc | movable_acc | net_vs_step0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.500538 | 0.015780 | 0.057621 | 0.522363 | 0.260057 | baseline |
| 1 | 0.503288 | 0.013270 | 0.060610 | 0.525492 | 0.258621 | 23 |
| 2 | 0.498267 | 0.011476 | 0.057262 | 0.520668 | 0.251437 | -19 |

## Training Selector

| step | pass@8 | majority@8 | response_acc | selected | selected_precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000000 | 0.500000 | 0.406250 | 0 | n/a |
| 2 | 0.500000 | 0.000000 | 0.125000 | 1 | 0.000000 |

Best step by acc@1: `1`.
