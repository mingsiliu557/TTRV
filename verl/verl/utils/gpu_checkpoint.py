"""Local HF checkpoints without a full CPU state dict.

Both InternVL and Qwen use a reusable 16 MiB buffer, without mmap, to bound
host-side weight staging. load_file(device="cuda") can still page whole shards
into host RAM, so it is intentionally not used under a small host-memory quota.
Format: https://github.com/huggingface/safetensors#format
"""

import json
import math
import os
from pathlib import Path
import struct
import sys

import torch

BUFFER_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint metadata key: {key}")
        result[key] = value
    return result


def _read_exact(file, view):
    offset = 0
    while offset < len(view):
        n = file.readinto(view[offset:])
        if not n:
            raise ValueError("Truncated safetensors payload")
        offset += n


def _header(path):
    with open(path, "rb", buffering=0) as file:
        prefix = bytearray(8)
        _read_exact(file, memoryview(prefix))
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= MAX_HEADER_BYTES:
            raise ValueError(f"Invalid or oversized safetensors header: {path.name}")
        raw = bytearray(size)
        _read_exact(file, memoryview(raw))
        header = json.loads(raw, object_pairs_hook=_unique_object)
    offset = 8 + size
    entries = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported checkpoint dtype {info['dtype']} for {name}")
        shape = info["shape"]
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid tensor shape for {name}")
        start, end = info["data_offsets"]
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if type(start) is not int or type(end) is not int or start < 0 or end - start != expected:
            raise ValueError(f"Invalid tensor byte offsets for {name}")
        entries[name] = (dtype, tuple(shape), start, end)
    cursor = 0
    for _, _, start, end in sorted(entries.values(), key=lambda x: (x[2], x[3])):
        if start != cursor:
            raise ValueError(f"Overlapping or non-contiguous payload in {path.name}")
        cursor = end
    if offset + cursor != path.stat().st_size:
        raise ValueError(f"Truncated or trailing payload in {path.name}")
    return offset, entries


def checkpoint_layout(model_path):
    """Read only bounded headers; never download weights or map their payload."""
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError("GPU loading requires an existing local checkpoint directory")
    index = root / "model.safetensors.index.json"
    weight_map = None
    if index.exists():
        if index.stat().st_size > MAX_HEADER_BYTES:
            raise ValueError("Oversized checkpoint index")
        with index.open() as file:
            weight_map = json.load(file, object_pairs_hook=_unique_object)["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = ["model.safetensors"]
    if not filenames:
        raise ValueError("Empty checkpoint index")
    layout, found = [], {}
    for filename in filenames:
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Checkpoint index must reference local safetensors shard filenames")
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"GPU loading requires local safetensors weights; missing {filename}")
        offset, entries = _header(path)
        for name in entries:
            if name in found:
                raise ValueError(f"Duplicate tensor across checkpoint shards: {name}")
            found[name] = filename
        layout.append((path, offset, entries))
    if weight_map is not None and found != weight_map:
        raise ValueError("Checkpoint index does not match shard headers")
    return layout


def _stream_tensors(path, offset, entries, expected, device, buffer):
    view = memoryview(buffer)
    with open(path, "rb", buffering=0) as file:
        for name, (dtype, shape, start, end) in sorted(entries.items(), key=lambda item: item[1][2]):
            target_dtype = expected[name].dtype
            tensor = torch.empty(shape, dtype=target_dtype, device=device)
            flat = tensor.reshape(-1)
            itemsize = torch.empty((), dtype=dtype).element_size()
            capacity = len(buffer) // itemsize * itemsize
            if capacity == 0:
                raise ValueError("Streaming buffer is smaller than one checkpoint element")
            file.seek(offset + start)
            consumed = 0
            while consumed < end - start:
                size = min(capacity, end - start - consumed)
                _read_exact(file, view[:size])
                cpu_chunk = torch.frombuffer(buffer, dtype=dtype, count=size // itemsize)
                # Blocking transfer is essential: the CPU buffer is reused next iteration.
                # Convert dtype on the GPU, never expand a whole tensor on the CPU.
                gpu_chunk = cpu_chunk.to(device=device, non_blocking=False)
                flat[consumed // itemsize:(consumed + size) // itemsize].copy_(gpu_chunk)
                del gpu_chunk, cpu_chunk
                consumed += size
                # Best effort: release clean file-cache pages charged to the job cgroup.
                if hasattr(os, "posix_fadvise"):
                    try:
                        os.posix_fadvise(file.fileno(), offset + start + consumed - size,
                                         size, os.POSIX_FADV_DONTNEED)
                    except OSError:
                        pass
            yield name, tensor


def checkpoint_key_mapping(model, keys):
    """Retain HF's legacy VLM key conversion (e.g. old Qwen visual/model names)."""
    if hasattr(model, "_get_key_renaming_mapping"):
        mapping = model._get_key_renaming_mapping(
            list(keys), key_mapping=getattr(model, "_checkpoint_conversion_mapping", None)
        )
    else:
        mapping = {name: name for name in keys}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Checkpoint key conversion produced duplicate target names")
    return mapping


def assign_checkpoint(model, model_path, device, *, buffer_bytes=BUFFER_BYTES):
    """Assign checkpoint tensors to a meta model before constructing its optimizer.

    CPU device is supported for small unit tests; production always passes CUDA.
    Missing/extra weights fail rather than silently initializing random parameters.
    """
    if sys.byteorder != "little":
        raise RuntimeError("Streaming safetensors requires a little-endian host")
    if buffer_bytes < 8:
        raise ValueError("buffer_bytes must be at least 8")
    device = torch.device(device)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    expected = model.state_dict()
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
    layout = checkpoint_layout(model_path)
    key_mapping = checkpoint_key_mapping(model, [name for _, _, entries in layout for name in entries])
    layout = [(path, offset, {key_mapping[name]: info for name, info in entries.items()})
              for path, offset, entries in layout]
    keys = set()
    for _, _, entries in layout:
        for name, (_, shape, _, _) in entries.items():
            if name not in expected or tuple(expected[name].shape) != shape:
                raise ValueError(f"Unexpected checkpoint key or shape mismatch: {name}")
            keys.add(name)
    missing = set(expected) - keys
    for group in aliases.values():
        if keys.intersection(group):
            missing.difference_update(group)
    if missing:
        raise ValueError(f"Missing checkpoint weights: {sorted(missing)[:10]}")

    weights = {}
    buffer = bytearray(buffer_bytes)
    for path, offset, entries in layout:
        weights.update(_stream_tensors(path, offset, entries, expected, device, buffer))
    for group in aliases.values():
        source = next((name for name in group if name in weights), None)
        if source is not None:
            for name in group:
                weights.setdefault(name, weights[source])
    model.load_state_dict(weights, strict=True, assign=True)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    # Preserve initialized nonpersistent buffers such as rotary frequencies.
    model.to(device=device)
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("GPU checkpoint loading left unresolved meta tensors")
    return model


def load_model_on_gpu(model_class, model_path, config, torch_dtype, trust_remote_code=True):
    """Direct replacement for AutoModel.from_pretrained in actor/ref initialization."""
    from accelerate import init_empty_weights

    if not Path(model_path).is_dir():
        raise ValueError("GPU loading requires local model files")
    device = torch.device("cuda", torch.cuda.current_device())
    kwargs = dict(torch_dtype=torch_dtype, attn_implementation="eager")
    factory = model_class.from_config
    if trust_remote_code and model_class.__name__ in getattr(config, "auto_map", {}):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.models.auto.auto_factory import add_generation_mixin_to_remote_model

        class_ref = config.auto_map[model_class.__name__]
        if "--" in class_ref:
            raise ValueError("External remote-code repositories are not supported by the local GPU loader")
        local_class = get_class_from_dynamic_module(class_ref, model_path, local_files_only=True)
        local_class = add_generation_mixin_to_remote_model(local_class)
        factory = local_class._from_config
    else:
        kwargs["trust_remote_code"] = trust_remote_code
    # Parameters are meta; temporary constructor tensors and real rotary buffers
    # use CUDA, not CPU. include_buffers=False avoids uninitialized meta RoPE.
    with torch.device(device), init_empty_weights(include_buffers=False):
        model = factory(config, **kwargs)
    # Nested VLM text configs may retain a checkpoint's bf16 dtype even when the
    # outer model requests fp32. Cast meta parameters before allocating weights.
    model.to(dtype=torch_dtype)
    model = assign_checkpoint(model, model_path, device)
    return model.eval()
