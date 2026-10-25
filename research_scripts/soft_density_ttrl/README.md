# Soft-Density / Density-Only TTRL Diagnostics

This directory contains D0 offline diagnostics for the density-only prototype reward. It reuses the current feature-center representation: `response_embeddings` returned during actor old-log-prob recomputation. It does not introduce a new VLM hidden-state extractor.

## Generate rollout JSONL with embeddings

Run a tiny smoke or a normal density/feature-center run with:

```bash
TTRL_LOG_RESPONSE_EMBEDDINGS=1 \
RUN_DENSITY_PEAK=1 \
DENSITY_PEAK_STYLES="density_peak_hard" \
RUN_OFFICIAL=0 RUN_FREQUENCY_ONLY=0 RUN_VSH=0 RUN_FEATURE_CENTER_HARD=0 RUN_FEATURE_CENTER_HSR=0 \
bash run_mme20_feature_center_gate.sh
```

`TTRL_LOG_RESPONSE_EMBEDDINGS=1` is intentionally optional because it makes `train_rollouts.jsonl` large.

## Recompute D0 rewards

```bash
python research_scripts/soft_density_ttrl/compute_density_rewards.py \
  --input-jsonl /path/to/train_rollouts.jsonl \
  --output-jsonl /path/to/density_diagnostics.jsonl \
  --summary-json /path/to/density_summary.json \
  --choice-labels A-F
```

## Analyze buckets

```bash
python research_scripts/soft_density_ttrl/analyze_density_diagnostics.py \
  --diagnostics-jsonl /path/to/density_diagnostics.jsonl \
  --output-json /path/to/density_bucket_summary.json \
  --output-md /path/to/density_bucket_summary.md \
  --choice-labels A-F
```

Important D0 checks:

- density reward is not constant;
- density peak accuracy is not below majority/centroid overall;
- high-entropy or minority-correct buckets show positive signal;
- density advantage is not identical to frequency advantage.
