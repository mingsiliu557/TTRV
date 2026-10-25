#!/usr/bin/env python3
"""Minimal Qwen2.5-VL/vLLM image-generation diagnostic."""

import os

from qwen_vl_utils import fetch_image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODEL = os.environ.get(
    "MODEL_PATH",
    "/data/ttrv-datasets/models/Qwen2.5-VL-7B-Instruct",
)
IMAGE = os.environ.get(
    "IMAGE_PATH",
    "/data/ttrv/MME/images/code_reasoning/0020.png",
)


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL, use_fast=False)
    image = fetch_image(
        {
            "image": IMAGE,
            "min_pixels": 3136,
            "max_pixels": 1003520,
        },
        image_patch_size=14,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Is a python code shown in the picture? Please answer "
                        "with yes or no. Please respond with only the "
                        "corresponding option letter (A or B). Do not include "
                        "any explanation or extra text.\nOptions:\nA. No\nB. Yes."
                    ),
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
    print(
        "input",
        {
            "image_mode": image.mode,
            "image_size": image.size,
            "prompt_tokens": len(prompt_token_ids),
            "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        },
        flush=True,
    )

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.40,
        max_model_len=8548,
        max_num_batched_tokens=8548,
        limit_mm_per_prompt={"image": 1},
    )
    outputs = llm.generate(
        [
            {
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": {"image": [image]},
            }
        ],
        SamplingParams(temperature=0.0, max_tokens=8),
        use_tqdm=False,
    )
    print("output", outputs[0].outputs[0].text, flush=True)


if __name__ == "__main__":
    main()
