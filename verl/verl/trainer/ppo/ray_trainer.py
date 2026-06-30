# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import hashlib
import json
import random
import re
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import ray
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


def _harmony_seed(global_step, sample_index, image_index):
    raw = f"{global_step}:{sample_index}:{image_index}".encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16)


def _canonical_harmony_transform_type(transform_type):
    aliases = {
        "photometric": "photometric_weak",
        "center_crop_resize": "center_crop_s092",
    }
    return aliases.get(transform_type, transform_type)


def _center_crop_resize(image, scale):
    width, height = image.size
    crop_width = max(1, int(width * scale))
    crop_height = max(1, int(height * scale))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image.resize((width, height))


def _random_crop_resize(image, rng, min_scale):
    width, height = image.size
    scale = min(1.0, min_scale + (1.0 - min_scale) * rng.random())
    crop_width = max(1, int(width * scale))
    crop_height = max(1, int(height * scale))
    max_left = max(0, width - crop_width)
    max_top = max(0, height - crop_height)
    left = rng.randint(0, max_left) if max_left else 0
    top = rng.randint(0, max_top) if max_top else 0
    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image.resize((width, height)), scale


def _apply_photometric_jitter(image, rng, strength):
    from PIL import ImageEnhance

    if strength == "weak":
        ranges = {
            "brightness": (0.9, 1.1),
            "contrast": (0.9, 1.1),
            "color": (0.95, 1.05),
            "sharpness": (0.95, 1.10),
        }
    elif strength == "medium":
        ranges = {
            "brightness": (0.8, 1.2),
            "contrast": (0.85, 1.15),
            "color": (0.85, 1.15),
            "sharpness": (0.85, 1.25),
        }
    elif strength == "strong":
        ranges = {
            "brightness": (0.65, 1.35),
            "contrast": (0.7, 1.3),
            "color": (0.7, 1.3),
            "sharpness": (0.7, 1.4),
        }
    else:
        raise ValueError(f"Unsupported photometric strength: {strength}")

    enhancers = [
        ("brightness", ImageEnhance.Brightness),
        ("contrast", ImageEnhance.Contrast),
        ("color", ImageEnhance.Color),
        ("sharpness", ImageEnhance.Sharpness),
    ]
    factors = {}
    for name, enhancer_cls in enhancers:
        low, high = ranges[name]
        factor = low + (high - low) * rng.random()
        image = enhancer_cls(image).enhance(factor)
        factors[name] = factor
    return image, factors


def _add_gaussian_noise(image, rng, noise_std):
    if noise_std <= 0:
        return image
    arr = np.asarray(image).astype(np.float32)
    noise_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    arr = np.clip(arr + noise_rng.normal(0.0, noise_std * 255.0, arr.shape), 0, 255).astype(np.uint8)
    from PIL import Image

    return Image.fromarray(arr, mode="RGB")


def _transform_harmony_image(image, transform_type, seed, return_metadata=False):
    from PIL import ImageFilter, ImageOps

    image = image.convert("RGB").copy()
    canonical_type = _canonical_harmony_transform_type(transform_type)
    rng = random.Random(seed)
    metadata = {
        "harmony_transform_type": canonical_type,
        "requested_harmony_transform_type": transform_type,
        "direction_safe": True,
        "crop_scale": None,
        "photometric_strength": None,
        "photometric_factors": {},
        "blur_sigma": 0.0,
        "noise_std": 0.0,
        "flip_applied": False,
    }

    if canonical_type.startswith("center_crop_s"):
        try:
            scale = int(canonical_type.removeprefix("center_crop_s")) / 100.0
        except ValueError as exc:
            raise ValueError(f"Unsupported vision self-harmony transform type: {transform_type}") from exc
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"Invalid center crop scale for transform type: {transform_type}")
        metadata["crop_scale"] = scale
        image = _center_crop_resize(image, scale)
    elif canonical_type.startswith("photometric_"):
        strength = canonical_type.removeprefix("photometric_")
        metadata["photometric_strength"] = strength
        image, metadata["photometric_factors"] = _apply_photometric_jitter(image, rng, strength)
    elif canonical_type in {"cotta_weak_noflip", "cotta_strong_noflip", "multi_aug_safe", "cotta_strong_dtd_flip"}:
        strong = canonical_type in {"cotta_strong_noflip", "multi_aug_safe", "cotta_strong_dtd_flip"}
        metadata["photometric_strength"] = "strong" if strong else "medium"
        image, metadata["photometric_factors"] = _apply_photometric_jitter(
            image, rng, metadata["photometric_strength"]
        )
        image, crop_scale = _random_crop_resize(image, rng, 0.88 if strong else 0.94)
        metadata["crop_scale"] = crop_scale
        metadata["blur_sigma"] = (0.001 + (0.50 if strong else 0.25) * rng.random())
        image = image.filter(ImageFilter.GaussianBlur(radius=metadata["blur_sigma"]))
        metadata["noise_std"] = 0.005 if strong else 0.0025
        image = _add_gaussian_noise(image, rng, metadata["noise_std"])
        if canonical_type == "cotta_strong_dtd_flip":
            metadata["direction_safe"] = False
            metadata["flip_applied"] = rng.random() < 0.5
            if metadata["flip_applied"]:
                image = ImageOps.mirror(image)
    else:
        raise ValueError(f"Unsupported vision self-harmony transform type: {transform_type}")

    if return_metadata:
        return image, metadata
    return image


def _build_harmony_transform_batch(gen_batch: DataProto, source_batch: DataProto, transform_type: str, global_step: int):
    transform_batch = deepcopy(gen_batch)
    if "multi_modal_data" not in transform_batch.non_tensor_batch:
        raise ValueError("vision_self_harmony requires multi_modal_data")

    transformed_multi_modal_data = []
    transform_metadata = []
    for sample_i, multi_modal_data in enumerate(transform_batch.non_tensor_batch["multi_modal_data"]):
        new_multi_modal_data = {}
        sample_metadata = []
        for key, value in multi_modal_data.items():
            if key != "image":
                new_multi_modal_data[key] = deepcopy(value)
                continue
            transformed_images = []
            for image_i, image in enumerate(value):
                transformed_image, metadata = _transform_harmony_image(
                    image,
                    transform_type=transform_type,
                    seed=_harmony_seed(global_step, sample_i, image_i),
                    return_metadata=True,
                )
                transformed_images.append(transformed_image)
                sample_metadata.append(metadata)
            new_multi_modal_data[key] = transformed_images
        transformed_multi_modal_data.append(new_multi_modal_data)
        transform_metadata.append(sample_metadata[0] if len(sample_metadata) == 1 else sample_metadata)
    transform_batch.non_tensor_batch["multi_modal_data"] = np.array(transformed_multi_modal_data, dtype=object)
    return transform_batch, transform_metadata


DENSITY_EMBEDDING_REWARD_STYLES = {
    "density_peak_hard",
    "density_peak_soft",
    "density_peak_answer_entropy",
    "density_peak_density_entropy",
    "density_cluster_soft",
    "density_cluster_answer_entropy",
    "density_cluster_density_entropy",
}


_DENSITY_EVIDENCE_DEFAULT_TEMPLATE = (
    "Candidate answer: {response}\n"
    "Visual evidence supporting this candidate answer:"
)

_DENSITY_CANONICAL_EVIDENCE_DEFAULT_TEMPLATE = (
    "Candidate option: {candidate}\n"
    "Visual evidence needed:"
)

_DENSITY_CANONICAL_OPTION_DEFAULT_TEMPLATE = "Candidate option: {candidate}"


def _decode_response_for_evidence(tokenizer, response_ids, response_mask):
    valid_len = int(response_mask.sum().detach().cpu().item())
    if valid_len <= 0:
        return "unknown"
    token_ids = response_ids[:valid_len].detach().cpu().tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return text if text else "unknown"


def _decode_prompt_for_evidence(tokenizer, prompt_ids, prompt_mask):
    valid_ids = prompt_ids[prompt_mask.to(dtype=torch.bool)]
    if valid_ids.numel() <= 0:
        return ""
    return tokenizer.decode(valid_ids.detach().cpu().tolist(), skip_special_tokens=True)


def _extract_choice_options_from_prompt(prompt_text: str):
    if not prompt_text:
        return {}

    marker_match = None
    for marker in re.finditer(r"\b(?:options|choices)\s*:", prompt_text, flags=re.IGNORECASE):
        marker_match = marker
    option_text = prompt_text[marker_match.end():] if marker_match else prompt_text
    option_text = re.sub(r"\s+", " ", option_text).strip()

    pattern = re.compile(r"(?:^|\s)(?:\(([A-F])\)|([A-F])\s*[\.:])\s*")
    matches = list(pattern.finditer(option_text))
    options = {}
    for idx, match in enumerate(matches):
        label = (match.group(1) or match.group(2) or "").upper()
        if not label:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(option_text)
        value = option_text[start:end].strip()
        value = re.sub(r"\s*assistant\s*$", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"<\|im_end\|>.*$", "", value).strip()
        if value:
            options[label] = value
    return options


def _extract_choice_label_from_response(response_text: str, valid_labels):
    if not response_text:
        return None
    labels = "".join(sorted(valid_labels or set("ABCDEF")))
    if not labels:
        labels = "ABCDEF"
    escaped = re.escape(labels)
    text = response_text.strip()

    patterns = [
        rf"^[\s\(\[\{{:;,\.\-\|/\\$]*([{escaped}])(?:\b|[\)\]\}}\.:\-,\|/\\$])",
        rf"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([{escaped}])\)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _canonicalize_density_evidence_text(prompt_text: str, response_text: str, template: str):
    options = _extract_choice_options_from_prompt(prompt_text)
    label = _extract_choice_label_from_response(response_text, set(options) or set("ABCDEF"))
    option = options.get(label) if label else None

    if label and option:
        candidate = f"{label}. {option}"
    elif label:
        candidate = label
    else:
        candidate = "unknown"

    return template.format(
        response=response_text,
        answer=label or response_text,
        candidate=candidate,
        option_label=label or "unknown",
        option_text=option or "",
    )


def _build_density_evidence_query_batch(
    batch: DataProto,
    tokenizer,
    template: str = None,
    canonicalize: bool = False,
):
    if "prompts" not in batch.batch.keys():
        raise ValueError("density evidence embedding requires prompts in batch")
    if "responses" not in batch.batch.keys():
        raise ValueError("density evidence embedding requires responses in batch")

    if not template:
        template = _DENSITY_CANONICAL_EVIDENCE_DEFAULT_TEMPLATE if canonicalize else _DENSITY_EVIDENCE_DEFAULT_TEMPLATE
    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    prompt_length = prompts.size(-1)
    response_length = responses.size(-1)
    prompt_attention_mask = batch.batch["attention_mask"][:, :prompt_length]
    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        response_mask = batch.batch["attention_mask"][:, -response_length:]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    device = responses.device
    evidence_responses = torch.full_like(responses, fill_value=int(pad_token_id))
    evidence_response_mask = torch.zeros_like(response_mask)
    evidence_queries = []
    evidence_token_lengths = []

    for row_idx in range(responses.size(0)):
        decoded_response = _decode_response_for_evidence(tokenizer, responses[row_idx], response_mask[row_idx])
        if canonicalize:
            decoded_prompt = _decode_prompt_for_evidence(tokenizer, prompts[row_idx], prompt_attention_mask[row_idx])
            evidence_text = _canonicalize_density_evidence_text(decoded_prompt, decoded_response, template)
        else:
            evidence_text = template.format(
                response=decoded_response,
                answer=decoded_response,
                candidate=decoded_response,
                option_label="unknown",
                option_text="",
            )
        token_ids = tokenizer.encode(evidence_text, add_special_tokens=False)[:response_length]
        if not token_ids:
            token_ids = [int(pad_token_id)]
        token_tensor = torch.tensor(token_ids, dtype=responses.dtype, device=device)
        token_count = token_tensor.numel()
        evidence_responses[row_idx, :token_count] = token_tensor
        evidence_response_mask[row_idx, :token_count] = 1
        evidence_queries.append(evidence_text)
        evidence_token_lengths.append(token_count)

    evidence_attention_mask = torch.cat([prompt_attention_mask, evidence_response_mask], dim=-1)
    evidence_position_ids = compute_position_id_with_mask(evidence_attention_mask)
    if batch.batch["position_ids"].dim() == 3:
        mrope_dim = batch.batch["position_ids"].size(1)
        evidence_position_ids = evidence_position_ids.unsqueeze(1).expand(-1, mrope_dim, -1)

    non_tensor_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in batch.non_tensor_batch else []
    evidence_batch = batch.select(
        batch_keys=["responses", "input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=non_tensor_keys,
        deepcopy=True,
    )
    evidence_batch.batch["responses"] = evidence_responses
    evidence_batch.batch["input_ids"] = torch.cat([prompts, evidence_responses], dim=-1)
    evidence_batch.batch["attention_mask"] = evidence_attention_mask
    evidence_batch.batch["position_ids"] = evidence_position_ids
    evidence_batch.meta_info = deepcopy(batch.meta_info)
    evidence_batch.meta_info["return_response_embeddings"] = True
    return evidence_batch, evidence_queries, evidence_token_lengths


def _canonical_papo_mask_type(mask_type: str) -> str:
    mask_type = str(mask_type or "black").lower().strip()
    aliases = {
        "random_patch_blackening": "black",
        "blackening": "black",
        "black": "black",
        "random_patch_gray": "gray",
        "grey": "gray",
        "gray": "gray",
        "random_patch_blur": "blur",
        "blur": "blur",
    }
    if mask_type not in aliases:
        raise ValueError(f"Unsupported PAPO mask_type={mask_type!r}; expected one of {sorted(aliases)}")
    return aliases[mask_type]


def _random_patch_papo_perturbation(
    image,
    patch_size: int,
    mask_prob: float,
    seed: int,
    mask_type: str = "black",
):
    from PIL import ImageDraw, ImageFilter

    patch_size = int(patch_size)
    mask_prob = float(mask_prob)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if not (0.0 <= mask_prob <= 1.0):
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")

    mask_type = _canonical_papo_mask_type(mask_type)

    image = image.convert("RGB").copy()
    width, height = image.size
    rng = random.Random(seed)
    draw = ImageDraw.Draw(image)

    total_patches = 0
    perturbed_patches = 0
    blur_radius = max(1.0, patch_size / 3.0)
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            total_patches += 1
            if rng.random() >= mask_prob:
                continue
            right = min(width, left + patch_size)
            bottom = min(height, top + patch_size)
            if mask_type == "black":
                draw.rectangle((left, top, right - 1, bottom - 1), fill=(0, 0, 0))
            elif mask_type == "gray":
                draw.rectangle((left, top, right - 1, bottom - 1), fill=(127, 127, 127))
            elif mask_type == "blur":
                patch = image.crop((left, top, right, bottom)).filter(ImageFilter.GaussianBlur(radius=blur_radius))
                image.paste(patch, (left, top))
            perturbed_patches += 1

    perturbed_ratio = perturbed_patches / total_patches if total_patches else 0.0
    metadata = {
        "papo_mask_strategy": "random",
        "papo_mask_type": f"random_patch_{mask_type}",
        "papo_mask_patch_size": patch_size,
        "papo_mask_prob": mask_prob,
        "papo_mask_total_patches": total_patches,
        "papo_mask_perturbed_patches": perturbed_patches,
        "papo_mask_perturbed_ratio": perturbed_ratio,
        "papo_masked_fraction": perturbed_ratio,
        "papo_grounding_used": False,
        "papo_fallback": False,
        "papo_grounding_missing": False,
        # Kept for backward-compatible metrics and scripts.
        "papo_mask_blackened_patches": perturbed_patches,
        "papo_mask_blackened_ratio": perturbed_ratio,
    }
    return image, metadata

def _random_patch_blackening(image, patch_size: int, black_prob: float, seed: int):
    return _random_patch_papo_perturbation(
        image,
        patch_size=patch_size,
        mask_prob=black_prob,
        seed=seed,
        mask_type="black",
    )


def _jsonable_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_sample_value(values, sample_i: int, default=None):
    if values is None:
        return default
    try:
        return _jsonable_scalar(values[sample_i])
    except Exception:
        return default


def _truthy_config_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _grounding_key_from_parts(data_source, index, extra_info=None) -> Optional[str]:
    if index is None and isinstance(extra_info, dict):
        index = extra_info.get("index")
    data_source = _jsonable_scalar(data_source)
    index = _jsonable_scalar(index)
    if data_source is None or index is None:
        return None
    return f"{data_source}::{index}"


def _grounding_key_from_record(record: dict[str, Any]) -> Optional[str]:
    key = record.get("grounding_key")
    if key:
        return str(key)
    extra_info = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else None
    return _grounding_key_from_parts(record.get("data_source"), record.get("index"), extra_info=extra_info)


def _load_papo_grounding_table(path: str) -> dict[str, dict[str, Any]]:
    if path is None or str(path).strip() == "":
        return {}
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"PAPO grounding file not found: {path}")

    records = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "records" in payload:
            records = payload["records"]
        elif isinstance(payload, dict):
            records = [dict(value, grounding_key=key) for key, value in payload.items() if isinstance(value, dict)]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported PAPO grounding JSON payload in {path}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc

    table: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = _grounding_key_from_record(record)
        if key is None:
            continue
        table[key] = record
    return table


def _is_number_like(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_whole_image_sentinel(value) -> bool:
    if isinstance(value, str):
        return value.strip().upper() == "WHOLE_IMAGE"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _is_whole_image_sentinel(value[0])
    return False


def _coerce_grounding_boxes(raw_boxes):
    if isinstance(raw_boxes, np.ndarray):
        raw_boxes = raw_boxes.tolist()
    if _is_whole_image_sentinel(raw_boxes):
        return "WHOLE_IMAGE"
    if raw_boxes is None:
        return []
    if isinstance(raw_boxes, dict):
        raw_boxes = raw_boxes.get("boxes_norm", raw_boxes.get("box_norm", raw_boxes.get("boxes", raw_boxes.get("bbox"))))
        if isinstance(raw_boxes, np.ndarray):
            raw_boxes = raw_boxes.tolist()
        if _is_whole_image_sentinel(raw_boxes):
            return "WHOLE_IMAGE"
    if not isinstance(raw_boxes, (list, tuple)):
        return []

    candidates = []
    if len(raw_boxes) == 4 and all(_is_number_like(value) for value in raw_boxes):
        candidates = [raw_boxes]
    else:
        for item in raw_boxes:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if _is_whole_image_sentinel(item):
                return "WHOLE_IMAGE"
            if isinstance(item, dict):
                item = item.get("box_norm", item.get("boxes_norm", item.get("box", item.get("bbox"))))
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, (list, tuple)) and len(item) == 4 and all(_is_number_like(value) for value in item):
                candidates.append(item)

    boxes = []
    for candidate in candidates:
        x0, y0, x1, y1 = [float(value) for value in candidate]
        if max(x0, y0, x1, y1) > 1.0 + 1e-6 or min(x0, y0, x1, y1) < -1e-6:
            continue
        x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
        y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes


def _select_grounding_boxes_for_image(record: dict[str, Any], image_i: int):
    if "boxes_by_image" in record:
        raw_boxes = record["boxes_by_image"]
        if isinstance(raw_boxes, dict):
            raw_boxes = raw_boxes.get(str(image_i), raw_boxes.get(image_i, raw_boxes.get(f"image_{image_i}")))
        elif isinstance(raw_boxes, (list, tuple)) and image_i < len(raw_boxes):
            raw_boxes = raw_boxes[image_i]
        else:
            raw_boxes = None
    else:
        raw_boxes = record.get("boxes_norm", record.get("boxes"))
        if isinstance(raw_boxes, dict):
            raw_boxes = raw_boxes.get(str(image_i), raw_boxes.get(image_i, raw_boxes.get(f"image_{image_i}", raw_boxes)))
    return _coerce_grounding_boxes(raw_boxes)


def _confidence_is_low(confidence) -> bool:
    if confidence is None or confidence == "":
        return False
    if isinstance(confidence, np.ndarray):
        confidence = confidence.tolist()
    if isinstance(confidence, dict):
        confidence = list(confidence.values())
    if isinstance(confidence, (list, tuple)):
        numeric_values = [float(value) for value in confidence if _is_number_like(value)]
        return bool(numeric_values) and max(numeric_values) <= 0.0
    return _is_number_like(confidence) and float(confidence) <= 0.0


def _grounding_record_fallback_reason(record: Optional[dict[str, Any]]) -> Optional[str]:
    if record is None:
        return "missing_key"
    if _truthy_config_value(record.get("audit_disabled", False)):
        return "audit_disabled"
    if _truthy_config_value(record.get("fallback", False)):
        return str(record.get("fallback_reason") or "record_fallback")
    if _confidence_is_low(record.get("confidence")):
        return "low_confidence"
    return None


def _dilate_norm_box(box, box_dilate: float):
    x0, y0, x1, y1 = box
    if box_dilate <= 0:
        return [x0, y0, x1, y1]
    width = x1 - x0
    height = y1 - y0
    dx = width * box_dilate
    dy = height * box_dilate
    return [max(0.0, x0 - dx), max(0.0, y0 - dy), min(1.0, x1 + dx), min(1.0, y1 + dy)]


def _boxes_to_pil_mask(image_size, boxes_norm, box_dilate: float):
    from PIL import Image, ImageDraw

    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes_norm:
        x0, y0, x1, y1 = _dilate_norm_box(box, box_dilate)
        left = max(0, min(width, int(np.floor(x0 * width))))
        top = max(0, min(height, int(np.floor(y0 * height))))
        right = max(0, min(width, int(np.ceil(x1 * width))))
        bottom = max(0, min(height, int(np.ceil(y1 * height))))
        if right > left and bottom > top:
            draw.rectangle((left, top, right - 1, bottom - 1), fill=255)
    masked_fraction = float(np.asarray(mask, dtype=np.uint8).mean() / 255.0) if width and height else 0.0
    return mask, masked_fraction


def _apply_papo_mask_to_regions(image, region_mask, mask_type: str, blur_radius: float):
    from PIL import Image, ImageFilter

    mask_type = _canonical_papo_mask_type(mask_type)
    image = image.convert("RGB").copy()
    if region_mask.getbbox() is None:
        return image
    if mask_type == "black":
        image.paste(Image.new("RGB", image.size, (0, 0, 0)), (0, 0), region_mask)
    elif mask_type == "gray":
        image.paste(Image.new("RGB", image.size, (127, 127, 127)), (0, 0), region_mask)
    elif mask_type == "blur":
        image.paste(image.filter(ImageFilter.GaussianBlur(radius=blur_radius)), (0, 0), region_mask)
    return image


def _sample_patch_mask_inside_region(image_size, region_mask, patch_size: int, mask_prob: float, seed: int):
    from PIL import Image, ImageChops, ImageDraw

    patch_size = int(patch_size)
    mask_prob = float(mask_prob)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if not (0.0 <= mask_prob <= 1.0):
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")

    width, height = image_size
    region_arr = np.asarray(region_mask, dtype=np.uint8)
    patch_boxes = []
    total_patches = 0
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            total_patches += 1
            right = min(width, left + patch_size)
            bottom = min(height, top + patch_size)
            if region_arr[top:bottom, left:right].max(initial=0) > 0:
                patch_boxes.append((left, top, right, bottom))

    selected_patch_count = int(round(mask_prob * len(patch_boxes)))
    selected_patch_count = max(0, min(len(patch_boxes), selected_patch_count))
    rng = random.Random(seed)
    selected_patch_boxes = list(patch_boxes)
    rng.shuffle(selected_patch_boxes)
    selected_patch_boxes = selected_patch_boxes[:selected_patch_count]

    patch_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(patch_mask)
    for left, top, right, bottom in selected_patch_boxes:
        draw.rectangle((left, top, right - 1, bottom - 1), fill=255)

    sampled_region_mask = ImageChops.multiply(region_mask, patch_mask)
    masked_fraction = float(np.asarray(sampled_region_mask, dtype=np.uint8).mean() / 255.0) if width and height else 0.0
    region_fraction = float(region_arr.mean() / 255.0) if width and height else 0.0
    return sampled_region_mask, {
        "papo_mask_patch_size": patch_size,
        "papo_mask_prob": mask_prob,
        "papo_mask_total_patches": total_patches,
        "papo_grounding_region_patch_count": len(patch_boxes),
        "papo_mask_perturbed_patches": selected_patch_count,
        "papo_mask_total_patch_ratio": selected_patch_count / total_patches if total_patches else 0.0,
        "papo_mask_region_patch_ratio": selected_patch_count / len(patch_boxes) if patch_boxes else 0.0,
        "papo_grounding_region_fraction": region_fraction,
        "papo_masked_fraction": masked_fraction,
        "papo_masked_fraction_in_grounding": masked_fraction / region_fraction if region_fraction > 0.0 else 0.0,
        # Kept for the existing logger name. For region-random masks this is the
        # selected patch ratio over the full image grid, not the pixel fraction.
        "papo_mask_blackened_ratio": selected_patch_count / total_patches if total_patches else 0.0,
    }


def _grounding_box_papo_perturbation(
    image,
    boxes_norm,
    grounding_direction: str = "evidence",
    box_dilate: float = 0.0,
    mask_type: str = "black",
    region_patch_sample: bool = False,
    patch_size: int = 14,
    mask_prob: float = 0.6,
    seed: int = 0,
):
    from PIL import ImageChops

    grounding_direction = str(grounding_direction or "evidence").lower().strip()
    if grounding_direction not in {"evidence", "context"}:
        raise ValueError("PAPO_GROUNDING_DIRECTION must be 'evidence' or 'context'")
    box_dilate = float(box_dilate)
    if box_dilate < 0.0:
        raise ValueError(f"PAPO_GROUNDING_BOX_DILATE must be non-negative, got {box_dilate}")

    image = image.convert("RGB")
    box_mask, box_fraction = _boxes_to_pil_mask(image.size, boxes_norm, box_dilate=box_dilate)
    if grounding_direction == "evidence":
        active_mask = box_mask
        masked_fraction = box_fraction
    else:
        active_mask = ImageChops.invert(box_mask)
        masked_fraction = 1.0 - box_fraction

    mask_type = _canonical_papo_mask_type(mask_type)
    blur_radius = max(1.0, min(image.size) / 32.0)
    patch_metadata = {}
    if region_patch_sample:
        active_mask, patch_metadata = _sample_patch_mask_inside_region(
            image_size=image.size,
            region_mask=active_mask,
            patch_size=patch_size,
            mask_prob=mask_prob,
            seed=seed,
        )
        masked_fraction = patch_metadata["papo_masked_fraction"]
    masked_image = _apply_papo_mask_to_regions(image, active_mask, mask_type=mask_type, blur_radius=blur_radius)
    metadata = {
        "papo_mask_strategy": "grounding_region_random" if region_patch_sample else "grounding",
        "papo_mask_type": (
            f"grounding_{grounding_direction}_patch_{mask_type}"
            if region_patch_sample
            else f"grounding_{grounding_direction}_{mask_type}"
        ),
        "papo_grounding_direction": grounding_direction,
        "papo_grounding_box_count": len(boxes_norm),
        "papo_grounding_box_dilate": box_dilate,
        "papo_grounding_box_fraction": box_fraction,
        "papo_masked_fraction": masked_fraction,
        "papo_grounding_used": True,
        "papo_fallback": False,
        "papo_grounding_missing": False,
    }
    metadata.update(patch_metadata)
    return masked_image, metadata


def _resolution_papo_perturbation(image, downscale: float = 0.25):
    from PIL import Image

    downscale = float(downscale)
    if not (0.0 < downscale <= 1.0):
        raise ValueError(f"PAPO_FALLBACK_DOWNSCALE must be in (0, 1], got {downscale}")
    image = image.convert("RGB")
    width, height = image.size
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    low_size = (max(1, int(round(width * downscale))), max(1, int(round(height * downscale))))
    masked_image = image.resize(low_size, resampling).resize((width, height), resampling)
    metadata = {
        "papo_mask_type": "resolution_downscale",
        "papo_fallback_mask": "resolution",
        "papo_fallback_downscale": downscale,
        "papo_masked_fraction": 1.0,
    }
    return masked_image, metadata


def _papo_grounding_fallback_perturbation(
    image,
    fallback_reason: str,
    fallback_mask: str,
    fallback_downscale: float,
    patch_size: int,
    mask_prob: float,
    seed: int,
    mask_type: str,
    grounding_key: Optional[str],
):
    fallback_mask = str(fallback_mask or "resolution").lower().strip()
    if fallback_mask == "resolution":
        masked_image, metadata = _resolution_papo_perturbation(image, downscale=fallback_downscale)
    elif fallback_mask == "random":
        masked_image, metadata = _random_patch_papo_perturbation(
            image,
            patch_size=patch_size,
            mask_prob=mask_prob,
            seed=seed,
            mask_type=mask_type,
        )
        metadata["papo_fallback_mask"] = "random"
    else:
        raise ValueError("PAPO_FALLBACK_MASK must be 'resolution' or 'random'")

    metadata.update(
        {
            "papo_mask_strategy": "grounding",
            "papo_grounding_key": grounding_key,
            "papo_grounding_used": False,
            "papo_fallback": True,
            "papo_fallback_reason": str(fallback_reason),
            "papo_grounding_missing": str(fallback_reason) in {"missing_key", "missing_batch_key"},
        }
    )
    metadata.setdefault("papo_masked_fraction", metadata.get("papo_mask_perturbed_ratio", 0.0))
    return masked_image, metadata


def _iter_papo_mask_metadata(metadata_batch):
    for sample_metadata in metadata_batch:
        if isinstance(sample_metadata, dict):
            yield sample_metadata
        elif isinstance(sample_metadata, np.ndarray):
            for item in sample_metadata.tolist():
                if isinstance(item, dict):
                    yield item
        elif isinstance(sample_metadata, (list, tuple)):
            for item in sample_metadata:
                if isinstance(item, dict):
                    yield item


def _processor_prompt_for_masked_branch(prompt: str) -> str:
    # InternVL stores <image> in prompts but expects <IMG_CONTEXT> during processor encoding.
    return prompt.replace("<image>", "<IMG_CONTEXT>")


def _build_papo_masked_multi_modal_inputs(
    batch: DataProto,
    processor,
    patch_size: int,
    black_prob: float,
    global_step: int,
    mask_type: str = "black",
    strategy: str = "random",
    grounding_table: Optional[dict[str, dict[str, Any]]] = None,
    grounding_direction: str = "evidence",
    box_dilate: float = 0.0,
    fallback_mask: str = "resolution",
    fallback_downscale: float = 0.25,
):
    if processor is None:
        raise ValueError("PAPO perception loss requires a multimodal processor")
    if "multi_modal_data" not in batch.non_tensor_batch:
        raise ValueError("PAPO perception loss requires multi_modal_data in the training batch")
    if "full_prompts" not in batch.non_tensor_batch:
        raise ValueError("PAPO perception loss requires data.return_full_prompt=True")

    strategy = str(strategy or "random").lower().strip()
    grounding_strategies = {"grounding", "grounding_region_random", "grounding_patch_sample"}
    if strategy not in {"random", *grounding_strategies}:
        raise ValueError(
            "PAPO_MASK_STRATEGY must be 'random', 'grounding', "
            "'grounding_region_random', or 'grounding_patch_sample'"
        )
    patch_size = int(patch_size)
    black_prob = float(black_prob)
    box_dilate = float(box_dilate)
    fallback_downscale = float(fallback_downscale)
    grounding_table = grounding_table or {}

    masked_inputs = []
    masked_metadata = []
    full_prompts = batch.non_tensor_batch["full_prompts"]
    multi_modal_data_batch = batch.non_tensor_batch["multi_modal_data"]
    data_sources = batch.non_tensor_batch.get("data_source")
    indexes = batch.non_tensor_batch.get("index")
    extra_infos = batch.non_tensor_batch.get("extra_info")

    for sample_i, (prompt, multi_modal_data) in enumerate(zip(full_prompts, multi_modal_data_batch)):
        if not isinstance(multi_modal_data, dict) or "image" not in multi_modal_data:
            raise ValueError("PAPO perception loss currently supports image multimodal data only")

        grounding_key = None
        grounding_record = None
        grounding_record_reason = None
        sample_data_source = None
        sample_index = None
        if strategy in grounding_strategies:
            sample_data_source = _safe_sample_value(data_sources, sample_i)
            sample_index = _safe_sample_value(indexes, sample_i)
            sample_extra_info = _safe_sample_value(extra_infos, sample_i)
            grounding_key = _grounding_key_from_parts(sample_data_source, sample_index, extra_info=sample_extra_info)
            if grounding_key is None:
                grounding_record_reason = "missing_batch_key"
            else:
                grounding_record = grounding_table.get(grounding_key)
                grounding_record_reason = _grounding_record_fallback_reason(grounding_record)

        masked_images = []
        sample_metadata = []
        for image_i, image in enumerate(multi_modal_data["image"]):
            seed = _harmony_seed(global_step, sample_i, image_i)
            if strategy == "random":
                masked_image, metadata = _random_patch_papo_perturbation(
                    image,
                    patch_size=patch_size,
                    mask_prob=black_prob,
                    seed=seed,
                    mask_type=mask_type,
                )
            else:
                fallback_reason = grounding_record_reason
                boxes_norm = []
                if fallback_reason is None:
                    boxes_norm = _select_grounding_boxes_for_image(grounding_record, image_i)
                    if boxes_norm == "WHOLE_IMAGE":
                        fallback_reason = "whole_image"
                    elif not boxes_norm:
                        fallback_reason = "no_boxes"

                if fallback_reason is not None:
                    masked_image, metadata = _papo_grounding_fallback_perturbation(
                        image,
                        fallback_reason=fallback_reason,
                        fallback_mask=fallback_mask,
                        fallback_downscale=fallback_downscale,
                        patch_size=patch_size,
                        mask_prob=black_prob,
                        seed=seed,
                        mask_type=mask_type,
                        grounding_key=grounding_key,
                    )
                else:
                    masked_image, metadata = _grounding_box_papo_perturbation(
                        image,
                        boxes_norm=boxes_norm,
                        grounding_direction=grounding_direction,
                        box_dilate=box_dilate,
                        mask_type=mask_type,
                        region_patch_sample=strategy in {"grounding_region_random", "grounding_patch_sample"},
                        patch_size=patch_size,
                        mask_prob=black_prob,
                        seed=seed,
                    )
                    metadata.update(
                        {
                            "papo_grounding_key": grounding_key,
                            "papo_grounding_data_source": sample_data_source,
                            "papo_grounding_index": sample_index,
                            "papo_grounding_confidence": grounding_record.get("confidence"),
                        }
                    )
            masked_images.append(masked_image)
            sample_metadata.append(metadata)

        prompt_text = _processor_prompt_for_masked_branch(str(prompt))
        model_inputs = processor(text=[prompt_text], images=masked_images, return_tensors="pt")
        model_inputs.pop("input_ids", None)
        model_inputs.pop("attention_mask", None)
        model_inputs.pop("second_per_grid_ts", None)
        masked_inputs.append(dict(model_inputs))
        masked_metadata.append(sample_metadata[0] if len(sample_metadata) == 1 else sample_metadata)

    return np.array(masked_inputs, dtype=object), np.array(masked_metadata, dtype=object)


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch

from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self._papo_grounding_table = None
        self._papo_grounding_table_path = None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        
        if self.config.reward_model.reward_manager == "ttrl":
            self.use_ttrl = True
            self.n_samples_per_prompt = self.config.reward_model.reward_kwargs.n_samples_per_prompt
            self.n_votes_per_prompt = self.config.reward_model.reward_kwargs.n_votes_per_prompt
        else:
            self.use_ttrl = False

        self._validate_config()
        self._create_dataloader()

    def _get_papo_grounding_table(self):
        grounding_file = self.config.data.get("papo_grounding_file", None)
        if grounding_file is None or str(grounding_file).strip() == "":
            return {}
        grounding_file = os.path.expanduser(str(grounding_file))
        if self._papo_grounding_table is None or self._papo_grounding_table_path != grounding_file:
            self._papo_grounding_table = _load_papo_grounding_table(grounding_file)
            self._papo_grounding_table_path = grounding_file
            print(f"[papo] loaded {len(self._papo_grounding_table)} grounding records from {grounding_file}")
        return self._papo_grounding_table

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            if (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
            ):
                assert config.actor_rollout_ref.model.use_remove_padding, (
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        papo_mask_strategy = str(config.data.get("papo_mask_strategy", "random") or "random").lower().strip()
        if papo_mask_strategy not in {"random", "grounding", "grounding_region_random", "grounding_patch_sample"}:
            raise ValueError(
                "data.papo_mask_strategy must be 'random', 'grounding', "
                "'grounding_region_random', or 'grounding_patch_sample'"
            )
        papo_grounding_direction = str(config.data.get("papo_grounding_direction", "evidence") or "evidence").lower().strip()
        if papo_grounding_direction not in {"evidence", "context"}:
            raise ValueError("data.papo_grounding_direction must be 'evidence' or 'context'")
        papo_fallback_mask = str(config.data.get("papo_fallback_mask", "resolution") or "resolution").lower().strip()
        if papo_fallback_mask not in {"resolution", "random"}:
            raise ValueError("data.papo_fallback_mask must be 'resolution' or 'random'")

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, seed=1):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type

        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(
                    f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                    f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset"
                )
        else:
            dataset_cls = RLHFDataset

        self.train_dataset = dataset_cls(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
            # suffix_prompt=self.config.data.suffix_prompt,
        )

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", seed))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
            # suffix_prompt=self.config.data.suffix_prompt,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=20, # double check this ==> reduce it to 400
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1, (
            "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."
        )

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )
        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)


    def _validate_try(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        # kamla
        # sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            # output_ids = test_output_gen_batch.batch["responses"]
            # output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            # sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        # self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # add ttrl metrics
        if self.use_ttrl and "ttrl_info" in result:
            for key, val in result["ttrl_info"].items():
                metric_dict[f"val-ttrl/{key}"] = val

        return metric_dict

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # add ttrl metrics
        if self.use_ttrl and "ttrl_info" in result:
            for key, val in result["ttrl_info"].items():
                metric_dict[f"val-ttrl/{key}"] = val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        # ensure_dir_exists(default_local_dir)
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        # ensure_dir_exists(local_global_step_folder)
        
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        # ensure_dir_exists(dataloader_local_path)
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    # def ensure_dir_exists(path):
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _select_top_k_per_prompt(self, data, n_votes_per_prompt, n_samples_per_prompt):
        assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
        num_prompts = len(data) // n_votes_per_prompt

        selected_indices = []
        for i in range(num_prompts):
            start = i * n_votes_per_prompt
            selected_indices.extend(range(start, start + n_samples_per_prompt))

        return data[selected_indices]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.meta_info["do_vote"] = False
                if self.use_ttrl:
                    self.config.actor_rollout_ref.rollout.n = self.n_votes_per_prompt
                    batch.meta_info["do_vote"] = True

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                        meta_info_keys=["do_vote"]
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                        meta_info_keys=["do_vote"]
                    )

                reward_style = self.config.reward_model.reward_kwargs.get("reward_style", None)
                if (
                    self.config.actor_rollout_ref.actor.get("use_papo_prcp_loss", False)
                    and "multi_modal_data" in gen_batch.non_tensor_batch
                ):
                    batch.non_tensor_batch["multi_modal_data"] = deepcopy(
                        gen_batch.non_tensor_batch["multi_modal_data"]
                    )
                use_vision_self_harmony = self.use_ttrl and reward_style == "vision_self_harmony"
                if use_vision_self_harmony:
                    harmony_transform_type = self.config.reward_model.reward_kwargs.get(
                        "harmony_transform_type", "photometric"
                    )
                    harmony_transform_gen_batch, harmony_transform_metadata = _build_harmony_transform_batch(
                        gen_batch=gen_batch,
                        source_batch=batch,
                        transform_type=harmony_transform_type,
                        global_step=self.global_steps,
                    )
                else:
                    harmony_transform_gen_batch = None
                    harmony_transform_metadata = None

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        if self.use_ttrl:
                            assert len(gen_batch_output) == len(batch) * self.n_votes_per_prompt
                        else:
                            pass
                    if harmony_transform_gen_batch is not None:
                        with _timer("gen_harmony_transform", timing_raw):
                            harmony_transform_output = self.actor_rollout_wg.generate_sequences(harmony_transform_gen_batch)
                            assert len(harmony_transform_output) == len(batch) * self.n_votes_per_prompt
                    else:
                        harmony_transform_output = None
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    if harmony_transform_output is not None:
                        batch.batch["harmony_transform_responses"] = harmony_transform_output.batch["responses"]
                        batch.batch["harmony_transform_attention_mask"] = harmony_transform_output.batch["attention_mask"]
                        batch.non_tensor_batch["harmony_transform_metadata"] = np.array(
                            [
                                deepcopy(metadata)
                                for metadata in harmony_transform_metadata
                                for _ in range(self.config.actor_rollout_ref.rollout.n)
                            ],
                            dtype=object,
                        )

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        use_density_embeddings = bool(
                            self.use_ttrl and reward_style in DENSITY_EMBEDDING_REWARD_STYLES
                        )
                        use_response_embeddings = bool(
                            self.use_ttrl
                            and (
                                reward_style in {"feature_center_hard", "feature_center_hsr"}
                                or use_density_embeddings
                            )
                        )
                        density_embedding_scope = str(
                            self.config.reward_model.reward_kwargs.get(
                                "density_embedding_scope", "response_mean_pool"
                            )
                        )
                        batch.meta_info["return_response_embeddings"] = use_response_embeddings
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item(), "train/entropy": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        #old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        if use_density_embeddings:
                            evidence_scopes = {
                                "evidence_query_mean_pool",
                                "canonical_evidence_query_mean_pool",
                                "canonical_option_query_mean_pool",
                            }
                            metrics["train/density_embedding_scope_evidence"] = float(
                                density_embedding_scope in {"evidence_query_mean_pool", "canonical_evidence_query_mean_pool"}
                            )
                            metrics["train/density_embedding_scope_canonical"] = float(
                                density_embedding_scope in {"canonical_evidence_query_mean_pool", "canonical_option_query_mean_pool"}
                            )
                            metrics["train/density_embedding_scope_option_only"] = float(
                                density_embedding_scope == "canonical_option_query_mean_pool"
                            )
                            if density_embedding_scope in evidence_scopes:
                                canonicalize_evidence = density_embedding_scope in {
                                    "canonical_evidence_query_mean_pool",
                                    "canonical_option_query_mean_pool",
                                }
                                if density_embedding_scope == "canonical_option_query_mean_pool":
                                    default_evidence_template = _DENSITY_CANONICAL_OPTION_DEFAULT_TEMPLATE
                                elif canonicalize_evidence:
                                    default_evidence_template = _DENSITY_CANONICAL_EVIDENCE_DEFAULT_TEMPLATE
                                else:
                                    default_evidence_template = _DENSITY_EVIDENCE_DEFAULT_TEMPLATE
                                evidence_template = self.config.reward_model.reward_kwargs.get(
                                    "density_evidence_template", default_evidence_template
                                )
                                evidence_batch, evidence_queries, evidence_token_lengths = _build_density_evidence_query_batch(
                                    batch=batch,
                                    tokenizer=self.tokenizer,
                                    template=evidence_template,
                                    canonicalize=canonicalize_evidence,
                                )
                                evidence_log_prob = self.actor_rollout_wg.compute_log_prob(evidence_batch)
                                batch.batch["response_embeddings"] = evidence_log_prob.batch["response_embeddings"]
                                batch.non_tensor_batch["density_evidence_query"] = np.array(evidence_queries, dtype=object)
                                if evidence_token_lengths:
                                    metrics["train/density_evidence_query_token_mean"] = float(
                                        np.mean(evidence_token_lengths)
                                    )
                                    metrics["train/density_evidence_query_token_max"] = float(
                                        np.max(evidence_token_lengths)
                                    )
                            elif density_embedding_scope in {"response_mean_pool", "full_response_mean_pool"}:
                                batch.non_tensor_batch["density_evidence_query"] = np.array(
                                    [""] * len(batch), dtype=object
                                )
                            else:
                                raise ValueError(f"Unsupported density_embedding_scope={density_embedding_scope}")

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_ttrl:
                            sorted_indices = sorted(range(len(batch)), key=lambda i: batch[i].non_tensor_batch["extra_info"]["index"])
                            batch = batch[sorted_indices]
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                            if self.use_ttrl:
                                ttrl_metrics = reward_result["ttrl_info"]
                                for k, v in ttrl_metrics.items():
                                    metrics.update({f"train/{k}": v})
                                
                                # Down Sampling
                                batch = self._select_top_k_per_prompt(
                                    batch, self.n_votes_per_prompt, self.n_samples_per_prompt
                                )
                                self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt

                                # Recompute ttrl metrics
                                post_reward_result = self.reward_fn.compute_post_ttrl_metrics(batch)
                                for k, v in post_reward_result.items():
                                    metrics.update({f"train/{k}": v})

                                # Recompute Entropy
                                post_entropy_loss = agg_loss(
                                    loss_mat=batch.batch["entropys"], loss_mask=batch.batch["response_mask"], loss_agg_mode=loss_agg_mode
                                )
                                metrics.update({"train/post_entropy": post_entropy_loss.detach().item()})
                                
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}

                        batch.batch["token_level_scores"] = reward_tensor

                        if (
                            self.config.actor_rollout_ref.actor.get("use_papo_prcp_loss", False)
                            and self.config.actor_rollout_ref.actor.get("papo_valid_only", False)
                        ):
                            response_mask = batch.batch.get("response_mask")
                            if response_mask is None:
                                response_mask = batch.batch["attention_mask"][:, -reward_tensor.size(-1):]
                            valid_response = reward_tensor.detach().abs().sum(dim=-1) > 1e-8
                            batch.batch["papo_valid_response_mask"] = (
                                response_mask * valid_response.to(response_mask.dtype).unsqueeze(-1)
                            )
                            metrics["train/papo_valid_response_ratio"] = float(
                                valid_response.float().mean().detach().item()
                            )

                        if self.use_ttrl:
                            self._balance_batch(batch, metrics=metrics)

                        if self.config.actor_rollout_ref.actor.get("use_papo_prcp_loss", False):
                            with _timer("papo_aug_log_prob", timing_raw):
                                papo_mask_strategy = self.config.data.get("papo_mask_strategy", "random")
                                papo_grounding_table = (
                                    self._get_papo_grounding_table()
                                    if str(papo_mask_strategy or "random").lower().strip()
                                    in {"grounding", "grounding_region_random", "grounding_patch_sample"}
                                    else None
                                )
                                papo_masked_inputs, papo_mask_metadata = _build_papo_masked_multi_modal_inputs(
                                    batch=batch,
                                    processor=self.processor,
                                    patch_size=self.config.data.get("papo_mask_patch_size", 14),
                                    black_prob=self.config.data.get("papo_mask_prob", 0.6),
                                    global_step=self.global_steps,
                                    mask_type=self.config.data.get("papo_mask_type", "black"),
                                    strategy=papo_mask_strategy,
                                    grounding_table=papo_grounding_table,
                                    grounding_direction=self.config.data.get("papo_grounding_direction", "evidence"),
                                    box_dilate=self.config.data.get("papo_grounding_box_dilate", 0.0),
                                    fallback_mask=self.config.data.get("papo_fallback_mask", "resolution"),
                                    fallback_downscale=self.config.data.get("papo_fallback_downscale", 0.25),
                                )
                                papo_non_tensor_keys = ["multi_modal_inputs"]
                                for key in ["data_source", "index", "extra_info"]:
                                    if key in batch.non_tensor_batch:
                                        papo_non_tensor_keys.append(key)
                                papo_aug_batch = batch.select(
                                    batch_keys=["responses", "input_ids", "attention_mask", "position_ids"],
                                    non_tensor_batch_keys=papo_non_tensor_keys,
                                )
                                papo_aug_batch.non_tensor_batch["multi_modal_inputs"] = papo_masked_inputs
                                papo_aug_batch.meta_info = deepcopy(batch.meta_info)
                                papo_aug_batch.meta_info["return_response_embeddings"] = False
                                aug_log_prob = self.actor_rollout_wg.compute_log_prob(papo_aug_batch)
                                batch.batch["aug_log_probs"] = aug_log_prob.batch["old_log_probs"]

                                metadata_rows = list(_iter_papo_mask_metadata(papo_mask_metadata))
                                mask_ratios = [
                                    metadata["papo_mask_blackened_ratio"]
                                    for metadata in metadata_rows
                                    if "papo_mask_blackened_ratio" in metadata
                                ]
                                if mask_ratios:
                                    metrics["train/papo_mask_blackened_ratio"] = float(np.mean(mask_ratios))
                                masked_fractions = [
                                    metadata["papo_masked_fraction"]
                                    for metadata in metadata_rows
                                    if "papo_masked_fraction" in metadata
                                ]
                                if masked_fractions:
                                    metrics["train/papo_masked_fraction_mean"] = float(np.mean(masked_fractions))
                                if str(papo_mask_strategy or "random").lower().strip() in {
                                    "grounding",
                                    "grounding_region_random",
                                    "grounding_patch_sample",
                                }:
                                    grounding_used = [
                                        1.0 if metadata.get("papo_grounding_used", False) else 0.0
                                        for metadata in metadata_rows
                                        if "papo_grounding_used" in metadata
                                    ]
                                    if grounding_used:
                                        metrics["train/papo_grounding_used_ratio"] = float(np.mean(grounding_used))
                                    fallback_used = [
                                        1.0 if metadata.get("papo_fallback", False) else 0.0
                                        for metadata in metadata_rows
                                        if "papo_fallback" in metadata
                                    ]
                                    if fallback_used:
                                        metrics["train/papo_fallback_ratio"] = float(np.mean(fallback_used))
                                    grounding_missing = [
                                        1.0 if metadata.get("papo_grounding_missing", False) else 0.0
                                        for metadata in metadata_rows
                                        if "papo_grounding_missing" in metadata
                                    ]
                                    if grounding_missing:
                                        metrics["train/papo_grounding_missing_ratio"] = float(np.mean(grounding_missing))

                                    grounding_box_fractions = [
                                        metadata["papo_grounding_box_fraction"]
                                        for metadata in metadata_rows
                                        if "papo_grounding_box_fraction" in metadata
                                    ]
                                    if grounding_box_fractions:
                                        metrics["train/papo_grounding_box_fraction_mean"] = float(np.mean(grounding_box_fractions))
                                    grounding_region_fractions = [
                                        metadata["papo_grounding_region_fraction"]
                                        for metadata in metadata_rows
                                        if "papo_grounding_region_fraction" in metadata
                                    ]
                                    if grounding_region_fractions:
                                        metrics["train/papo_grounding_region_fraction_mean"] = float(np.mean(grounding_region_fractions))
                                    masked_in_grounding = [
                                        metadata["papo_masked_fraction_in_grounding"]
                                        for metadata in metadata_rows
                                        if "papo_masked_fraction_in_grounding" in metadata
                                    ]
                                    if masked_in_grounding:
                                        metrics["train/papo_masked_fraction_in_grounding_mean"] = float(
                                            np.mean(masked_in_grounding)
                                        )
                                    grounding_region_patch_counts = [
                                        metadata["papo_grounding_region_patch_count"]
                                        for metadata in metadata_rows
                                        if "papo_grounding_region_patch_count" in metadata
                                    ]
                                    if grounding_region_patch_counts:
                                        metrics["train/papo_grounding_region_patch_count_mean"] = float(
                                            np.mean(grounding_region_patch_counts)
                                        )
                                    mask_region_patch_ratios = [
                                        metadata["papo_mask_region_patch_ratio"]
                                        for metadata in metadata_rows
                                        if "papo_mask_region_patch_ratio" in metadata
                                    ]
                                    if mask_region_patch_ratios:
                                        metrics["train/papo_mask_region_patch_ratio_mean"] = float(
                                            np.mean(mask_region_patch_ratios)
                                        )
                                    mask_total_patch_ratios = [
                                        metadata["papo_mask_total_patch_ratio"]
                                        for metadata in metadata_rows
                                        if "papo_mask_total_patch_ratio" in metadata
                                    ]
                                    if mask_total_patch_ratios:
                                        metrics["train/papo_mask_total_patch_ratio_mean"] = float(
                                            np.mean(mask_total_patch_ratios)
                                        )

                                metrics["train/papo_aug_log_prob_mean"] = float(
                                    agg_loss(
                                        loss_mat=aug_log_prob.batch["old_log_probs"],
                                        loss_mask=batch.batch["response_mask"],
                                        loss_agg_mode=loss_agg_mode,
                                    ).detach().item()
                                )

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
