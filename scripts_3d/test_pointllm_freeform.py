#!/usr/bin/env python3
import argparse
import os
import pickle
from collections import Counter
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
from transformers import AutoTokenizer

from verl.utils.reward_score.modelnet40_freeform import MODELNET40_CLASSES, map_to_modelnet40_class


def pc_norm(pc):
    xyz = pc[:, :3].astype(np.float32, copy=True)
    xyz -= np.mean(xyz, axis=0)
    scale = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if scale > 0:
        xyz /= scale
    return xyz


def load_points(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        points = next((obj[key] for key in ("data", "points", "list_of_points") if key in obj), None)
        labels = next((obj[key] for key in ("label", "labels", "list_of_labels") if key in obj), None)
    else:
        points, labels = obj
    labels = [int(np.asarray(label).reshape(-1)[0]) for label in labels]
    return list(points), labels


def build_prompt(model, question):
    from pointllm.conversation import SeparatorStyle, conv_templates

    conv = conv_templates["vicuna_v1_1"].copy()
    point_config = model.get_model().point_backbone_config
    point_tokens = point_config["default_point_patch_token"] * point_config["point_token_len"]
    if point_config["mm_use_point_start_end"]:
        point_tokens = point_config["default_point_start_token"] + point_tokens + point_config["default_point_end_token"]
    conv.append_message(conv.roles[0], point_tokens + "\n" + question)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_str


def main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model-name", default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--data-file", default=str(repo_root / "data/modelnet40_data/modelnet40_test_8192pts_fps.dat"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    from pointllm.model import PointLLMLlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        use_cache=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()

    points, labels = load_points(args.data_file)
    point_cloud = np.concatenate([pc_norm(points[args.sample_index]), np.zeros((8192, 3), dtype=np.float32)], axis=1)
    point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda().to(model.dtype)
    label = labels[args.sample_index]

    prompt, _stop_str = build_prompt(model, "What is this?")
    input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()
    outputs = []
    with torch.inference_mode():
        for _ in range(args.num_rollouts):
            output_ids = model.generate(
                input_ids=input_ids,
                point_clouds=point_cloud,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            text = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
            outputs.append(text)

    mapped = [map_to_modelnet40_class(text) for text in outputs]
    print(f"GT: label_id={label}, name={MODELNET40_CLASSES[label]}")
    print("\nfree-form rollouts:")
    for i, text in enumerate(outputs, start=1):
        print(f"  {i}: {text}")
    print(f"\nNormalized: {mapped}")
    print(f"Counter: {dict(Counter(mapped))}")


if __name__ == "__main__":
    main()
