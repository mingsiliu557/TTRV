# TTRV Handoff - 2026-05-31

This folder records the current experiment/code state before moving machines.

## Repository State

- Git remote: `git@github.com:mingsiliu557/TTRV.git`
- Target branch: `main`
- Git user: `mingsiliu557 <liumingsi915@gmail.com>`
- Main runnable script: `/root/autodl-tmp/TTRV/run.sh`
- Important code changes are in:
  - `verl/verl/trainer/ppo/ray_trainer.py`
  - `verl/verl/utils/reward_score/ttrl/ttt_metrics.py`
  - `verl/verl/workers/reward_manager/ttrl.py`
  - `verl/examples/ttrv/run.sh`
  - `verl/verl/trainer/config/ppo_trainer.yaml`
  - `verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py`

## What Was Not Committed

The following local artifacts are intentionally excluded from Git:

- Dataset/cache roots: `/root/autodl-tmp/TTRV/data/`, `verl/data/*.parquet`, generated large datasets.
- Checkpoints: any `checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`.
- Large generated validation JSONL under `verl/outputs/`; several files are 119MB+ and exceed GitHub's normal single-file limit.

The paths and metrics for the important excluded output files are recorded in `IMPORTANT_LOGS_AND_RESULTS.md`.

## SQA3D

`SQA3D/` is an external nested repository. It is recorded as a submodule:

```text
path: SQA3D
url: https://github.com/SilongYong/SQA3D.git
commit: a98da8fda65f81f614bdfed48c27aae34975c566
```

After cloning on a new machine, restore it with:

```bash
git submodule update --init --recursive
```

## Resume Commands

Run the full Vision Self-Harmony transform ablation:

```bash
cd /root/autodl-tmp/TTRV
bash run.sh
```

Disable shutdown for debugging:

```bash
cd /root/autodl-tmp/TTRV
AUTO_SHUTDOWN=0 bash run.sh
```

The current `run.sh` defaults to `AUTO_SHUTDOWN=1`, so it shuts down the machine after completing all transform ablations.
