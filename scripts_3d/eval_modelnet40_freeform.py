#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/root/autodl-tmp/.cache")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/root/autodl-tmp/.cache/huggingface/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/.cache/huggingface/datasets")
os.environ.setdefault("TORCH_HOME", "/root/autodl-tmp/.cache/torch")
os.environ.setdefault("PIP_CACHE_DIR", "/root/autodl-tmp/.cache/pip")
os.environ.pop("TRANSFORMERS_CACHE", None)

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from verl.utils.reward_score.modelnet40_freeform import map_to_modelnet40_class


def build_prompt(model, question):
    from pointllm.conversation import conv_templates

    conv = conv_templates["vicuna_v1_1"].copy()
    point_config = model.get_model().point_backbone_config
    point_tokens = point_config["default_point_patch_token"] * point_config["point_token_len"]
    if point_config["mm_use_point_start_end"]:
        point_tokens = point_config["default_point_start_token"] + point_tokens + point_config["default_point_end_token"]
    conv.append_message(conv.roles[0], point_tokens + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model-name", default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--data-file", default=str(repo_root / "data/modelnet40_20/test.parquet"))
    parser.add_argument("--output-dir", default=str(repo_root / "outputs/pointllm-modelnet40-freeform-20/baseline_eval"))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    from pointllm.model import PointLLMLlamaForCausalLM

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("parquet", data_files=args.data_file)["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        use_cache=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()

    prompt = build_prompt(model, "What is this?")
    input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()
    results = []
    correct = 0
    unknown = 0

    rows = list(dataset)
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            if args.log_every and start % (args.batch_size * args.log_every) == 0:
                print(f"Evaluating {start}/{len(rows)}", flush=True)
            batch_input_ids = input_ids.repeat(len(batch_rows), 1)
            pc = torch.stack(
                [
                    torch.from_numpy(np.load(row["extra_info"]["pc_path"]).astype(np.float32))
                    for row in batch_rows
                ],
                dim=0,
            ).cuda().to(model.dtype)
            output_ids = model.generate(
                input_ids=batch_input_ids,
                point_clouds=pc,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            for row, output in zip(batch_rows, output_ids):
                extra = row["extra_info"]
                text = tokenizer.decode(output[input_ids.shape[1] :], skip_special_tokens=True).strip()
                pred = map_to_modelnet40_class(text)
                label_id = str(extra["label_id"])
                is_correct = pred == label_id
                correct += int(is_correct)
                unknown += int(pred == "unknown")
                results.append(
                    {
                        "index": extra["index"],
                        "label_id": extra["label_id"],
                        "label_name": extra["label_name"],
                        "prediction": text,
                        "mapped_prediction": pred,
                        "correct": is_correct,
                    }
                )

    accuracy = correct / len(dataset)
    unknown_rate = unknown / len(dataset)
    summary = {"n": len(dataset), "accuracy": accuracy, "unknown_rate": unknown_rate}
    with (output_dir / "predictions.json").open("w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    with (output_dir / "results.md").open("w") as f:
        f.write("# ModelNet40 Free-Form Greedy Baseline\n\n")
        f.write(f"- N: {len(dataset)}\n")
        f.write(f"- Accuracy: {accuracy:.4f}\n")
        f.write(f"- Unknown rate: {unknown_rate:.4f}\n")
    print(summary)


if __name__ == "__main__":
    main()
