# Important Logs And Results

## Main DTD20 2D TTRV Runs

| Run | Log | Final mean@1 | Final maj@1 | Note |
| --- | --- | ---: | ---: | --- |
| TTRV official | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0529_232251.log` | 0.88099 | 0.88110 | frequency + entropy |
| Frequency only | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_001507.log` | 0.89964 | 0.89973 | entropy coefficient 0 |
| Soft pseudo-label | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_010725.log` | 0.89645 | 0.89654 | soft pseudo-label reward |
| TTRL majority vote | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_103923.log` | 0.89041 | 0.89051 | majority hard pseudo-label |

## Vision Self-Harmony Initial Runs

| Run | Log | Output dir | Final mean@1 | Final maj@1 |
| --- | --- | --- | ---: | ---: |
| photometric | `verl/logs/dtd20_vision_self_harmony_0530_151326.out` | `verl/outputs/dtd20_vision_self_harmony_0530_151326/` | 0.88757 | 0.88767 |
| center_crop_resize | `verl/logs/dtd20_vision_self_harmony_center_crop_resize_0530_151326.out` | `verl/outputs/dtd20_vision_self_harmony_0530_151326/` | 0.90000 | 0.90009 |

Large output JSONL in that output directory is not committed because final validation JSONL files are about 119MB each.

## Vision Self-Harmony Transform Ablation

Master log:

```text
verl/logs/dtd20_vsh_transform_ablation_0530_173616.log
```

Output directory:

```text
verl/outputs/dtd20_vsh_transform_ablation_0530_173616/
```

The run was stopped manually to save GPU cost. Completed experiments: 7/12.

| Order | Transform | Per-run log | Final mean@1 | Final maj@1 | Status |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `center_crop_s098` | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_173616.log` | 0.90053 | 0.90062 | complete |
| 2 | `center_crop_s095` | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_182702.log` | 0.89929 | 0.89938 | complete |
| 3 | `center_crop_s092` | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_191545.log` | 0.90000 | 0.90009 | complete |
| 4 | `center_crop_s088` | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_200441.log` | 0.90124 | 0.90133 | best completed |
| 5 | `center_crop_s084` | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_205339.log` | 0.89503 | 0.89512 | complete |
| 6 | `photometric_weak` | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_214726.log` | 0.89787 | 0.89796 | complete |
| 7 | `photometric_medium` | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_223937.log` | 0.89503 | 0.89512 | complete |
| 8 | `photometric_strong` | `verl/logs/dtd_20_OpenGVLab_InternVL3-2B_2e_0530_232831.log` | - | - | incomplete |

Not yet run before stopping:

```text
cotta_weak_noflip
cotta_strong_noflip
multi_aug_safe
cotta_strong_dtd_flip
```

## Transform Ablation Rollout Stability

Each completed transform has 40 train rollout records. Every record has 32 original samples and 32 transform samples.

| Transform | Harmony label acc | Paired agreement | TV distance | Harmony margin | Original invalid | Transform invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| center_crop_s098 | 0.9500 | 0.7555 | 0.0338 | 0.8944 | 0.1008 | 0.1031 |
| center_crop_s095 | 0.9500 | 0.7641 | 0.0405 | 0.9054 | 0.0992 | 0.1187 |
| center_crop_s092 | 0.9500 | 0.7875 | 0.0321 | 0.9133 | 0.0914 | 0.0977 |
| center_crop_s088 | 0.9500 | 0.7695 | 0.0372 | 0.9095 | 0.0859 | 0.1117 |
| center_crop_s084 | 1.0000 | 0.7664 | 0.0546 | 0.8937 | 0.0859 | 0.1062 |
| photometric_weak | 0.9500 | 0.7953 | 0.0323 | 0.9103 | 0.0883 | 0.1000 |
| photometric_medium | 0.9500 | 0.7430 | 0.0265 | 0.8984 | 0.1148 | 0.1273 |

## Interpretation

- Best completed transform: `center_crop_s088`, final mean@1 0.90124.
- The gain over frequency-only 0.89964 is small, about +0.16 percentage points.
- Crop strength has a useful range: `s098`, `s092`, and `s088` are all near 0.900+, while `s084` drops to 0.89503.
- Photometric-only transforms underperform crop-only.
- The most important unanswered question is whether CoTTA-style combined safe transforms outperform crop-only. Those transforms were not reached before stopping.

## Large Output Files Not In Git

These are intentionally kept on disk but excluded from Git:

```text
verl/outputs/dtd20_vsh_transform_ablation_0530_173616/*_final_eval_flat.jsonl
verl/outputs/dtd20_vision_self_harmony_0530_151326/*_final_eval_flat.jsonl
verl/outputs/dtd20_fullval_ttrv_0530_122004/*fullval_flat.jsonl
```

Most of these final validation JSONL files are about 119MB each, and one combined fullval JSONL is about 249MB.

Small train rollout JSONL files under `verl/outputs` are also excluded because the whole `verl/outputs` directory is ignored upstream. If needed, copy selected train rollout files into `results/` before committing.
