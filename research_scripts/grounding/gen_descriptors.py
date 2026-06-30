#!/usr/bin/env python
"""Generate short visual evidence descriptors for PAPO grounding masks."""

import argparse
import re
from pathlib import Path

try:
    from ._common import (
        descriptor_mode,
        descriptor_phrases,
        grounding_key,
        image_path_from_spec,
        image_specs,
        is_whole_image_descriptor,
        load_image_from_spec,
        prompt_to_text,
        read_rows,
        row_data_source,
        row_index,
        write_jsonl,
    )
except ImportError:
    from _common import (  # type: ignore
        descriptor_mode,
        descriptor_phrases,
        grounding_key,
        image_path_from_spec,
        image_specs,
        is_whole_image_descriptor,
        load_image_from_spec,
        prompt_to_text,
        read_rows,
        row_data_source,
        row_index,
        write_jsonl,
    )


DEFAULT_DESCRIPTOR_PROMPT = """<image>
Question:
{question}

You are NOT answering the question. Your task is to name the visible image evidence that should be inspected.
Return exactly one JSON object and no extra text:
{{"mode":"localized","descriptors":["short visible phrase"]}}

Rules:
- Use mode "localized" when one or more visible objects, labels, chart/diagram elements, text snippets, numbers, axes, regions, arrows, lines, or spatial relations can be named.
- Use mode "whole_image" only when no local visible evidence can be named.
- Do not output the answer.
- Do not output option letters A/B/C/D/E/F unless the letter itself is visibly printed in the image or diagram.
- Descriptors must be short noun phrases grounded in the image, not explanations.
- Prefer concrete phrases such as "red car", "left bottle", "x-axis label", "angle x", "line segment AB", "shaded region", "table value 5", or "Chinese text at center".
- Return 1 to 5 descriptors.
"""

ANSWER_INSTRUCTION_PATTERNS = [
    r"<image>",
    r"Hint:\s*Please answer[^\n]*",
    r"Please answer directly with only[^\n.]*[.\n]?",
    r"Please respond with only[^\n.]*[.\n]?",
    r"Do not include any explanation[^\n.]*[.\n]?",
    r"Choose the correct answer from the options below and respond with only[^\n.]*[.\n]?",
    r"Please answer with yes or no[.\n]?",
]


def clean_question_for_descriptor(question: str) -> str:
    text = str(question or "")
    for pattern in ANSWER_INSTRUCTION_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip() or str(question or "").strip()



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True, help="Training parquet used by verl/examples/ttrv/run.sh")
    parser.add_argument("--output-jsonl", required=True, help="Descriptor JSONL output path")
    parser.add_argument("--model-path", default="OpenGVLab/InternVL3-2B", help="InternVL model path or cached HF id")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or an explicit torch device")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files")
    parser.add_argument("--descriptor-prompt-template", default=DEFAULT_DESCRIPTOR_PROMPT)
    parser.add_argument(
        "--internvl-img-context",
        action="store_true",
        help="Deprecated compatibility flag. The current InternVL chat path keeps <image> and passes pixel_values.",
    )
    parser.add_argument("--image-size", type=int, default=448, help="InternVL image tile size.")
    parser.add_argument("--max-num", type=int, default=12, help="Maximum dynamic image tiles for InternVL.")
    parser.add_argument("--use-flash-attn", action="store_true", help="Request flash attention when loading InternVL.")
    return parser.parse_args()


def load_descriptor_model(args):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Descriptor generation requires torch and transformers. Install them and ensure the InternVL model is cached."
        ) from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": args.local_files_only,
        }
        try:
            model = AutoModel.from_pretrained(
                args.model_path,
                use_flash_attn=bool(args.use_flash_attn),
                **model_kwargs,
            ).eval()
        except TypeError:
            model = AutoModel.from_pretrained(args.model_path, **model_kwargs).eval()
        model.to(device)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load InternVL descriptor model from {args.model_path!r}. "
            "Install model dependencies and pre-cache the model, or pass a local --model-path. "
            f"Original error: {exc}"
        ) from exc
    return model, tokenizer, device, dtype


def build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    tiles_per_row = target_width // image_size
    for i in range(blocks):
        box = (
            (i % tiles_per_row) * image_size,
            (i // tiles_per_row) * image_size,
            ((i % tiles_per_row) + 1) * image_size,
            ((i // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_to_pixel_values(image, input_size: int, max_num: int):
    import torch

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image.convert("RGB"), image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in images])


def generate_descriptor(model, tokenizer, device, dtype, image, question: str, args) -> str:
    import torch

    prompt = args.descriptor_prompt_template.format(question=clean_question_for_descriptor(question))
    try:
        pixel_values = image_to_pixel_values(
            image,
            input_size=int(args.image_size),
            max_num=int(args.max_num),
        ).to(dtype=dtype, device=device)
        generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_config["temperature"] = args.temperature
        with torch.no_grad():
            descriptor = model.chat(tokenizer, pixel_values, prompt, generation_config)
    except Exception as exc:
        raise RuntimeError(
            "InternVL descriptor generation failed. Check model.chat(pixel_values, prompt) support and local model files. "
            f"Original error: {exc}"
        ) from exc
    return descriptor or "WHOLE_IMAGE"


def main():
    args = parse_args()
    rows = read_rows(args.input_parquet, limit=args.limit)
    model, tokenizer, device, dtype = load_descriptor_model(args)
    outputs = []
    for ordinal, row in enumerate(rows):
        specs = image_specs(row, image_key=args.image_key)
        key = grounding_key(row, ordinal)
        descriptor = "WHOLE_IMAGE"
        fallback = False
        fallback_reason = ""
        image_path = image_path_from_spec(specs[0]) if specs else None
        if not specs:
            fallback = True
            fallback_reason = "missing_image"
        else:
            image = load_image_from_spec(specs[0])
            question = prompt_to_text(row.get(args.prompt_key, ""))
            descriptor = generate_descriptor(model, tokenizer, device, dtype, image, question, args)
            fallback = is_whole_image_descriptor(descriptor)
            fallback_reason = "whole_image_descriptor" if fallback else ""
        phrases = descriptor_phrases(descriptor)
        outputs.append(
            {
                "data_source": row_data_source(row),
                "index": row_index(row, ordinal),
                "grounding_key": key,
                "descriptor": descriptor,
                "descriptor_mode": descriptor_mode(descriptor),
                "descriptor_phrases": phrases,
                "image_path": image_path,
                "boxes_norm": [],
                "confidence": None,
                "fallback": fallback,
                "fallback_reason": fallback_reason,
            }
        )
    write_jsonl(args.output_jsonl, outputs)
    print(f"wrote {len(outputs)} descriptors to {Path(args.output_jsonl)}")


if __name__ == "__main__":
    main()
