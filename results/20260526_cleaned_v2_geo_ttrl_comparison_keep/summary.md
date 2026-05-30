# Cleaned PhysX-MCQ Geo/TTRL Cleanup Summary

Created: `2026-05-26T11:32:59.402448+00:00`

## What Was Kept

- Existing `outputs/*_keep` logs were preserved.
- Cleaned v1 static/TTRL/Geo-Harmony comparison artifacts were archived here and will be retained under `outputs/physx_cleaned_options_v1_keep`.
- Cleaned v2 pragmatic balanced dataset, static baseline, loose Geo-Harmony diagnostic, and strict Geo-Harmony diagnostic were archived here and will be retained under `outputs/physx_cleaned_options_v2_candidates_keep`.
- Checkpoint directories are not kept; metrics, reports, configs, run logs, and prediction JSONL files are kept in the retained cleaned output roots.

## Main Results

| result | n | acc@1 | invalid | hit_max | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 original retained step0 | 10572 | 0.335887 | 0.013999 | 0.039444 | baseline |
| v1 cleaned step0 | 10572 | 0.397654 | 0.010878 | 0.099981 | +0.061767 |
| v2 pragmatic balanced step0 | 8365 | 0.500538 | 0.015780 | 0.057621 | fresh subset |

Notes: v1 cleaned is directly compared on the retained original subset. v2 is a stricter pragmatic subset with rephrased/reordered options, so its 0.500538 acc@1 is the current clean baseline but not a same-example direct delta against v1.

## V1 TTRL / Geo-Harmony Comparison

| method | step | acc@1 | joint_acc | movable_acc | delta_vs_step0 | early_stop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| static_step0_cache | 0 | 0.397654 | 0.419061 | 0.263411 | +0.000000 | False  |
| geo_harmony | 1 | 0.389803 | 0.410178 | 0.262036 | -0.007851 | True accuracy_drop |
| ttrl_majority | 1 | 0.386587 | 0.406339 | 0.262724 | -0.011067 | True accuracy_drop |

Conclusion: Both Geo-Harmony and TTRL majority under the tested update settings dropped validation accuracy at step 1; early stop selected static step0 as best and recovery restored step0 checkpoints.

## V2 Geo-Harmony Diagnostics

| run | step | acc@1 | invalid | hit_max | joint_acc | movable_acc | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| loose HM0.60 | 0 | 0.500538 | 0.015780 | 0.057621 | 0.522363 | 0.260057 | unstable small drift |
| loose HM0.60 | 1 | 0.501614 | 0.014585 | 0.059175 | 0.523797 | 0.257184 | unstable small drift |
| loose HM0.60 | 2 | 0.499821 | 0.012074 | 0.059414 | 0.521320 | 0.262931 | unstable small drift |
| strict view0.75 | 0 | 0.500538 | 0.015780 | 0.057621 | 0.522363 | 0.260057 | baseline |
| strict view0.75 | 1 | 0.503288 | 0.013270 | 0.060610 | 0.525492 | 0.258621 | best but zero reward/no robust signal |
| strict view0.75 | 2 | 0.498267 | 0.011476 | 0.057262 | 0.520668 | 0.251437 | collapse/early-stop |

Best cleaned v2 acc@1 observed: strict step1 = `0.503288`, +0.002750 over static step0, but it is not considered a stable method improvement because training reward was zero at step1 and step2 collapsed to 0.498267 after one wrong pseudo-label update.

## Existing QType Keep Reference

| question_type | strategy | baseline | best/majority | delta |
| --- | --- | ---: | ---: | ---: |
| object_category | weight_update | 0.395107 | 0.403002 | +0.007895 |
| joint_type | test_time_vote8_majority | 0.351393 | 0.365760 | +0.014367 |
| movable_part | test_time_vote8_majority | 0.267610 | 0.273028 | +0.005418 |

## Cleanup Manifest

- Deleted non-keep smoke/target-part/Hydra/prepared-cache outputs from `/root/autodl-tmp/TTRV/outputs`.
- Deleted checkpoint directories from retained cleaned runs to save space; no prediction JSONL/metrics/report files from retained cleaned runs were deleted.
- Renamed retained cleaned roots and key run dirs with `_keep`.

See `summary.json` for machine-readable details.
