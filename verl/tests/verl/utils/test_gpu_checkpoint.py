"""Small offline tests; CPU exercises byte-identical streaming without a GPU job."""

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

# Do not import Ray/veRL's distributed runtime for a checkpoint-loader unit test.
spec = importlib.util.spec_from_file_location(
    "gpu_checkpoint", Path(__file__).resolve().parents[3] / "verl/utils/gpu_checkpoint.py"
)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class Toy(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.linear = torch.nn.Linear(7, 5, dtype=dtype)
        self.register_buffer("count", torch.tensor(3, dtype=torch.int64))
        self.register_buffer("empty", torch.empty(0))
        self.register_buffer("rotary", torch.tensor([0.1, 0.2]), persistent=False)


def empty_model(dtype=torch.float32):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        return Toy(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_exact_weights_and_forward(tmp_path, dtype):
    original = Toy(dtype).eval()
    save_file(original.state_dict(), str(tmp_path / "model.safetensors"))
    target = empty_model(torch.float32)
    loader.assign_checkpoint(target, tmp_path, "cpu", buffer_bytes=16)
    for key, value in original.state_dict().items():
        assert torch.equal(target.state_dict()[key], value.to(target.state_dict()[key].dtype))
    assert torch.equal(target.rotary, original.rotary)
    x = torch.ones(2, 7)
    torch.testing.assert_close(target.linear(x), original.float().linear(x))
    target.linear(x).sum().backward()
    assert target.linear.weight.grad is not None
    assert not any(p.is_meta for p in target.parameters())


def test_shards(tmp_path):
    original = Toy()
    items = list(original.state_dict().items())
    weights = {}
    for i, (name, value) in enumerate(items):
        filename = f"model-{i}.safetensors"
        save_file({name: value}, str(tmp_path / filename))
        weights[name] = filename
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    model = loader.assign_checkpoint(empty_model(), tmp_path, "cpu", buffer_bytes=24)
    assert torch.equal(model.linear.weight, original.linear.weight)
    weights["linear.weight"] = "../outside.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    with pytest.raises(ValueError, match="local safetensors"):
        loader.checkpoint_layout(tmp_path)


def test_hf_legacy_key_mapping(tmp_path):
    class LegacyToy(Toy):
        _checkpoint_conversion_mapping = {"old.": "linear."}

        def _get_key_renaming_mapping(self, keys, key_mapping=None):
            assert key_mapping == self._checkpoint_conversion_mapping
            return {name: name.replace("old.", "linear.") for name in keys}

    original = Toy().state_dict()
    save_file({name.replace("linear.", "old."): value for name, value in original.items()},
              str(tmp_path / "model.safetensors"))
    target = LegacyToy().to("meta")
    target.rotary = torch.tensor([0.1, 0.2])
    loader.assign_checkpoint(target, tmp_path, "cpu", buffer_bytes=16)
    assert torch.equal(target.linear.weight, original["linear.weight"])


def test_missing_extra_and_shape_fail(tmp_path):
    original = Toy().state_dict()
    for replacement, match in [
        ({key: val for key, val in original.items() if key != "linear.bias"}, "Missing"),
        ({**original, "extra": torch.ones(1)}, "Unexpected"),
        ({**original, "linear.bias": torch.ones(6)}, "shape mismatch"),
    ]:
        save_file(replacement, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match=match):
            loader.assign_checkpoint(empty_model(), tmp_path, "cpu")


def test_tied_missing_alias(tmp_path):
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 3, device="meta")
            self.head = torch.nn.Linear(3, 5, bias=False, device="meta")

        def tie_weights(self):
            self.head.weight = self.embedding.weight

    expected = torch.arange(15).reshape(5, 3).float()
    save_file({"embedding.weight": expected}, str(tmp_path / "model.safetensors"))
    model = loader.assign_checkpoint(Tied(), tmp_path, "cpu", buffer_bytes=8)
    assert model.embedding.weight is model.head.weight
    assert torch.equal(model.head.weight, expected)


def test_bounded_readinto_and_short_reads(tmp_path):
    save_file(Toy().state_dict(), str(tmp_path / "model.safetensors"))
    sizes = []
    real_read = loader._read_exact

    def record(file, view):
        sizes.append(len(view))
        return real_read(file, view)

    # Header parsing has its own separate bound; measure only tensor payload reads.
    layout = loader.checkpoint_layout(tmp_path)
    with patch.object(loader, "checkpoint_layout", return_value=layout), patch.object(loader, "_read_exact", record):
        loader.assign_checkpoint(empty_model(), tmp_path, "cpu", buffer_bytes=16)
    assert max(sizes) <= 16

    class ShortReader(io.BytesIO):
        def readinto(self, view):
            return super().readinto(view[:3])

    result = bytearray(8)
    real_read(ShortReader(b"12345678"), memoryview(result))
    assert result == b"12345678"
    with pytest.raises(ValueError, match="Truncated"):
        real_read(ShortReader(b"123"), memoryview(result))


def test_truncated_checkpoint(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file(Toy().state_dict(), str(path))
    with path.open("r+b") as file:
        file.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError, match="Truncated"):
        loader.checkpoint_layout(tmp_path)


def test_tiny_qwen_vl_matches_hf_forward(tmp_path):
    from accelerate import init_empty_weights
    from transformers import AutoModelForImageTextToText, Qwen2_5_VLConfig

    config = Qwen2_5_VLConfig(
        vision_start_token_id=29, vision_end_token_id=28, image_token_id=30, video_token_id=31,
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, patch_size=2, temporal_patch_size=1, spatial_merge_size=1),
    )
    original = AutoModelForImageTextToText.from_config(config, attn_implementation="eager").eval()
    # HF save_pretrained writes backward-compatible VLM names, just like the real checkpoint.
    original.save_pretrained(tmp_path)
    with init_empty_weights(include_buffers=False):
        target = AutoModelForImageTextToText.from_config(config, attn_implementation="eager")
    target = loader.assign_checkpoint(target, tmp_path, "cpu", buffer_bytes=32).eval()
    with torch.no_grad():
        inputs = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(target(input_ids=inputs).logits, original(input_ids=inputs).logits,
                                   rtol=0, atol=0)
    if torch.cuda.is_available():
        # Real VLM checkpoints retain bf16 in their nested text config.
        config.text_config.torch_dtype = torch.bfloat16
        gpu_model = loader.load_model_on_gpu(AutoModelForImageTextToText, tmp_path, config, torch.float32)
        assert all(p.dtype == torch.float32 for p in gpu_model.parameters())
        with torch.no_grad():
            torch.testing.assert_close(gpu_model(input_ids=inputs.cuda()).logits.cpu(),
                                       original(input_ids=inputs).logits, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_stream_and_assignment(tmp_path):
    source = Toy(torch.bfloat16)
    save_file(source.state_dict(), str(tmp_path / "model.safetensors"))
    target = loader.assign_checkpoint(empty_model(), tmp_path, "cuda:0", buffer_bytes=8)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key].cpu(), value.to(target.state_dict()[key].dtype))
    assert target.rotary.is_cuda
