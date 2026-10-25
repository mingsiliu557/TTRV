"""Offline, single-GPU loader check; does not start training or allocate Slurm jobs."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

spec = importlib.util.spec_from_file_location(
    "gpu_checkpoint", Path(__file__).resolve().parents[1] / "verl/utils/gpu_checkpoint.py"
)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    factory = AutoModelForCausalLM
    for candidate in (AutoModelForVision2Seq, AutoModelForImageTextToText):
        if type(config) in candidate._model_mapping:
            factory = candidate
            break
    start = time.monotonic()
    model = loader.load_model_on_gpu(
        factory, args.model_path, config,
        torch.bfloat16 if args.dtype == "bf16" else torch.float32,
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    print(f"Loaded {config.model_type} in {elapsed:.3f}s; checking checkpoint values", flush=True)
    loaded = model.state_dict()
    layout = loader.checkpoint_layout(args.model_path)
    key_mapping = loader.checkpoint_key_mapping(model, [name for _, _, entries in layout for name in entries])
    checked = 0
    # Spot-check both ends of EVERY stored tensor without CPU mmap or full copies.
    for path, offset, entries in layout:
        with path.open("rb", buffering=0) as file:
            for stored_name, (dtype, shape, first, last) in entries.items():
                name = key_mapping[stored_name]
                size = torch.empty((), dtype=dtype).element_size()
                if last == first:
                    continue
                for byte_offset, index in ((first, 0), (last - size, -1)):
                    file.seek(offset + byte_offset)
                    data = bytearray(size)
                    loader._read_exact(file, memoryview(data))
                    expected = torch.frombuffer(data, dtype=dtype)[0].to(loaded[name].dtype)
                    actual = loaded[name].reshape(-1)[index].cpu()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                checked += 1
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # A text-only forward checks loaded language weights and rotary buffers.
        # This is not a multimodal accuracy or training test.
        language_model = model.language_model if config.model_type.startswith("internvl") else model
        logits = language_model(input_ids=torch.tensor([[1, 2]], device="cuda"), use_cache=False).logits
        assert torch.isfinite(logits).all(), "Nonfinite logits"
    print(json.dumps({
        "model_type": config.model_type,
        "dtype": args.dtype,
        "load_seconds": round(elapsed, 3),
        "checked_tensors": checked,
        "cpu_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
        "text_forward_finite": True,
        "forward_autocast_dtype": "bf16",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
