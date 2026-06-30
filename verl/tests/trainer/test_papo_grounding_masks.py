import json

import numpy as np
from PIL import Image

from verl.trainer.ppo.ray_trainer import (
    _grounding_box_papo_perturbation,
    _grounding_key_from_parts,
    _load_papo_grounding_table,
    _random_patch_papo_perturbation,
    _resolution_papo_perturbation,
)


def test_grounding_evidence_masks_inside_box():
    image = Image.new("RGB", (10, 10), "white")
    masked, metadata = _grounding_box_papo_perturbation(
        image,
        boxes_norm=[[0.0, 0.0, 0.5, 1.0]],
        grounding_direction="evidence",
        mask_type="black",
    )

    assert masked.getpixel((1, 5)) == (0, 0, 0)
    assert masked.getpixel((8, 5)) == (255, 255, 255)
    assert metadata["papo_grounding_used"] is True
    assert metadata["papo_grounding_direction"] == "evidence"
    assert metadata["papo_masked_fraction"] == 0.5


def test_grounding_context_masks_outside_box():
    image = Image.new("RGB", (10, 10), "white")
    masked, metadata = _grounding_box_papo_perturbation(
        image,
        boxes_norm=[[0.0, 0.0, 0.5, 1.0]],
        grounding_direction="context",
        mask_type="black",
    )

    assert masked.getpixel((1, 5)) == (255, 255, 255)
    assert masked.getpixel((8, 5)) == (0, 0, 0)
    assert metadata["papo_grounding_direction"] == "context"
    assert metadata["papo_masked_fraction"] == 0.5


def test_resolution_fallback_downscales_and_restores_size():
    pattern = np.indices((16, 16)).sum(axis=0) % 2
    image = Image.fromarray(np.repeat((pattern * 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    masked, metadata = _resolution_papo_perturbation(image, downscale=0.25)

    assert masked.size == image.size
    assert np.asarray(masked).shape == np.asarray(image).shape
    assert not np.array_equal(np.asarray(masked), np.asarray(image))
    assert metadata["papo_fallback_mask"] == "resolution"
    assert metadata["papo_masked_fraction"] == 1.0


def test_random_patch_mask_compatibility():
    image = Image.new("RGB", (8, 8), "white")
    masked, metadata = _random_patch_papo_perturbation(
        image,
        patch_size=4,
        mask_prob=1.0,
        seed=0,
        mask_type="black",
    )

    assert masked.getpixel((7, 7)) == (0, 0, 0)
    assert metadata["papo_mask_blackened_ratio"] == 1.0
    assert metadata["papo_mask_perturbed_ratio"] == 1.0


def test_grounding_table_lookup_uses_data_source_and_index(tmp_path):
    path = tmp_path / "grounding.jsonl"
    record = {
        "data_source": "AI2D-TTT",
        "index": "ai2d_20-3",
        "descriptor": "label A",
        "boxes_norm": [[0.1, 0.2, 0.3, 0.4]],
        "confidence": 0.9,
        "fallback": False,
        "fallback_reason": "",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    table = _load_papo_grounding_table(str(path))
    key = _grounding_key_from_parts("AI2D-TTT", "ai2d_20-3")

    assert key == "AI2D-TTT::ai2d_20-3"
    assert table[key]["boxes_norm"] == [[0.1, 0.2, 0.3, 0.4]]
