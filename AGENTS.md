# AGENTS.md

## Scope

This file defines the working boundary for agents inside:

```text
/root/autodl-tmp/TTRV
```

Current branch/context: `ttrv-sqa3d`. The next task is to run PointLLM on the PhysXNet-derived 3D MCQ dataset and connect it to a TTRV-style test-time adaptation workflow.

This is not a data download or model conversion task. Use local files and local model caches only.

## Objective

Build a reproducible PointLLM + PhysX-MCQ TTRV experiment:

1. keep the existing PointLLM freeform MCQ evaluation path working;
2. use the parsed `prediction` (`A/B/C/D/unknown`) for TTRV frequency statistics, not the raw response string;
3. implement or adapt the reward path so that sampled raw responses are parsed before frequency/entropy reward is computed;
4. save per-example raw outputs, parsed predictions, reward components, and evaluation metrics;
5. report overall accuracy and accuracy by `question_type`.

## Key Inputs

Dataset:

```text
/root/autodl-tmp/TTRV/physx_mcq_workspace/PhysX-3D/outputs/physxnet_mcq_verl.json
/root/autodl-tmp/TTRV/physx_mcq_workspace/PhysX-3D/outputs/physxnet_mcq.jsonl
```

PointLLM code and outputs:

```text
/root/autodl-tmp/TTRV/pointllm-src/
/root/autodl-tmp/TTRV/pointllm-src/scripts/eval_physx_mcq_pointllm.py
/root/autodl-tmp/TTRV/pointllm-src/outputs/
/root/autodl-tmp/TTRV/pointllm-src/logs/
```

Local PhysXNet OBJ assets:

```text
/root/autodl-tmp/TTRV/physx_mcq_workspace/PhysX-3D/dataset_toolkits/physxnet/version_1/partseg/<object_id>/objs/*.obj
```

TTRV / veRL framework:

```text
/root/autodl-tmp/TTRV/verl/
```

Observed local PointLLM 7B cache roots:

```text
/root/.cache/huggingface/hub/models--RunsenXu--PointLLM_7B_v1.2
/root/autodl-tmp/.cache/huggingface/hub/models--RunsenXu--PointLLM_7B_v1.2
/root/autodl-tmp/.cache/huggingface/transformers/models--RunsenXu--PointLLM_7B_v1.2
```

Use `local_files_only=True` or equivalent offline Hugging Face behavior.

## Dataset Notes

`physxnet_mcq_verl.json` fields:

```text
prompt, image_path, answer, source, id
```

Important:

- `answer` is one of `A/B/C/D`.
- `prompt` already contains `<image>`, ABCD options, and an instruction to output only the option letter.
- `image_path` is a compatibility field. PointLLM must not consume the PNG as point-cloud input.
- Use `physxnet_mcq.jsonl` as the sidecar for `object_id`, `question_type`, part labels, true answer text, and debug metadata.
- Current validation found 44,487 examples, 0 invalid examples, and question types:
  - `object_category`: 32,047
  - `joint_type`: 9,118
  - `movable_part`: 3,322

## PointLLM Input Rules

PointLLM should receive a point cloud derived from local OBJ files:

- default `point_scope=full_object`;
- sample 8192 surface points;
- normalize XYZ to unit sphere;
- append zero RGB so the shape is `[8192, 6]`;
- cache under `pointllm-src/outputs/physxnet_point_cache/`;
- record `point_scope`, `pointnum`, `seed`, cache path, and preprocessing notes in every run.

Do not modify the original PhysXNet dataset or OBJ files.

## MCQ Generation And Parsing

For the main TTRV path, use freeform generation:

```bash
cd /root/autodl-tmp/TTRV/pointllm-src
python scripts/eval_physx_mcq_pointllm.py --max_examples 20 --answer_mode freeform
```

`choice_logits` is allowed only as a diagnostic or upper-bound sanity check. Do not mix it with freeform TTRV results.

The scoring path must preserve:

```text
prediction_raw -> parse_prediction(prediction_raw) -> prediction
```

Where `prediction` is one of:

```text
A, B, C, D, unknown
```

Unparsable outputs are `unknown` and must be counted as invalid/incorrect for eval.

## TTRV Reward Semantics

For each prompt:

```text
sample multiple raw responses
-> parse each raw response to A/B/C/D/unknown
-> compute frequency over parsed predictions
-> compute normalized entropy over the parsed prediction distribution
-> reward = beta * frequency - alpha * normalized_entropy
```

Rules:

- Frequency must be computed on the parsed `prediction`, not on the full raw response text.
- `unknown` should receive a negative reward such as `unknown_reward=-1.0`.
- Pure TTRV reward must not use ground-truth `answer`.
- Ground truth is used only for evaluation accuracy.
- TTRV can reinforce stable but wrong answers; always report accuracy alongside frequency/entropy.

Per-example TTRV logs should include at least:

```json
{
  "id": "...",
  "question_type": "...",
  "answer": "A",
  "prediction_raw": "...",
  "prediction": "A",
  "correct": true,
  "frequency": 0.0,
  "normalized_entropy": 0.0,
  "reward": 0.0,
  "response_token_len": 0,
  "hit_max_response_length": false,
  "point_path": "...",
  "model_path": "..."
}
```

## Baseline References

These are preliminary sanity numbers, not final benchmark claims:

| run | mode | n | acc | invalid |
| --- | --- | ---: | ---: | ---: |
| `physx_mcq_pointllm7b_default_verl_smoke20` | freeform | 20 | 0.4500 | 3 |
| `physx_mcq_pointllm7b_default_verl_200` | freeform | 200 | 0.2700 | 68 |
| `physx_mcq_pointllm7b_report` | choice_logits | 200 | 0.3400 | 0 |
| `physx_mcq_pointllm7b_choice_smoke20` | choice_logits | 20 | 0.4000 | 0 |

## Allowed Work

Agents may inspect and modify code needed for this experiment under:

```text
/root/autodl-tmp/TTRV/pointllm-src/
/root/autodl-tmp/TTRV/verl/
```

Agents may inspect dataset files under:

```text
/root/autodl-tmp/TTRV/physx_mcq_workspace/PhysX-3D/
```

Write experiment outputs under:

```text
/root/autodl-tmp/TTRV/pointllm-src/outputs/
/root/autodl-tmp/TTRV/pointllm-src/logs/
/root/autodl-tmp/TTRV/outputs/
/root/autodl-tmp/TTRV/output/
```

Keep edits scoped to the PointLLM evaluation adapter, TTRV dataset/reward/rollout plumbing, configs, scripts, and reports needed for this experiment.

## Forbidden Work

Do not:

- download checkpoints or external datasets;
- call external LLM APIs or online evaluation services;
- modify the original PhysXNet dataset in place;
- feed PNG `image_path` files directly to PointLLM as point clouds;
- overwrite model caches or original checkpoint weights;
- clean, reset, or delete large existing directories;
- run destructive commands such as `git reset --hard`, `git clean -fdx`, or recursive deletion of user data;
- report TTRV reward improvements as benchmark accuracy without evaluation.

## Ask Before

Ask the user before:

2. downloading anything;
6. using more than one GPU or changing system CUDA settings;
7. deleting or overwriting previous logs/results;
8. changing the dataset schema.

## Recommended Workflow

1. Re-run a 20/200 example PointLLM freeform smoke to confirm model, cache, and parser behavior.
2. Run a 200 example `choice_logits` diagnostic if useful, but keep it separate.
3. Inspect `prediction_raw`, invalid rate, and accuracy by `question_type`.
4. Implement the MCQ TTRV reward path in `verl/`: parse raw responses first, then compute frequency/entropy.
5. Run a tiny TTRV smoke subset and save all rollout/eval JSONL artifacts.
6. After each TTRV step, run evaluation and save `validation_predictions_step*.jsonl`.
7. Track best step separately from final step.

Use timestamped run names with key parameters, for example:

```text
20260522_153000-physx-freeform-r16-v32-a0p5
```

