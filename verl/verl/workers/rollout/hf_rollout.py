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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""

import contextlib
import hashlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]

POINT_CLOUD_MODEL_KEYS = ("point_clouds", "box_query", "box_mask", "click_query", "click_mask")


def _get_model_type(module: nn.Module) -> str | None:
    module = getattr(module, "module", module)
    return getattr(getattr(module, "config", None), "model_type", None)


def _point_cloud_model_keys(model_type: str | None):
    if model_type == "pointllm":
        return ("point_clouds",)
    return POINT_CLOUD_MODEL_KEYS


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _stable_int(value) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _normalize_xyz_inplace(point_clouds: torch.Tensor) -> torch.Tensor:
    xyz = point_clouds[..., :3]
    xyz.sub_(xyz.mean(dim=1, keepdim=True))
    radius = torch.sqrt(torch.sum(xyz * xyz, dim=-1)).amax(dim=1, keepdim=True).clamp_min_(1e-6)
    xyz.div_(radius.unsqueeze(-1))
    return point_clouds


def _apply_downsample(point_cloud: torch.Tensor, ratio: float, generator: torch.Generator) -> torch.Tensor:
    num_points = int(point_cloud.shape[0])
    keep = max(1, min(num_points, int(round(num_points * ratio))))
    perm = torch.randperm(num_points, device=point_cloud.device, generator=generator)[:keep]
    choice = torch.randint(0, keep, (num_points,), device=point_cloud.device, generator=generator)
    return point_cloud.index_select(0, perm.index_select(0, choice))


def _policy_for_view(policy: str, view_id: int, seed: int) -> str:
    policy = str(policy or "none").strip().lower()
    if policy != "mixed":
        return policy
    policies = ("sparsity", "sensor_noise", "rigid_pose", "composite")
    return policies[(view_id + seed) % len(policies)]


def _apply_point_cloud_reframes(point_clouds: torch.Tensor, *, n: int, config, meta_info: dict) -> torch.Tensor:
    policy = str(config.get("point_cloud_reframe_policy", "none") or "none").strip().lower()
    if policy in {"", "none", "off", "false"} or n <= 1:
        return point_clouds
    if point_clouds.ndim != 3 or point_clouds.shape[-1] < 3:
        return point_clouds

    num_views = max(1, int(config.get("point_cloud_reframe_num_views", 1) or 1))
    samples_per_view = int(config.get("point_cloud_reframe_samples_per_view", 0) or 0)
    if samples_per_view <= 0:
        samples_per_view = max(1, (n + num_views - 1) // num_views)
    seed = int(config.get("point_cloud_reframe_seed", 0) or 0)
    seed += int(meta_info.get("global_step", 0) or 0) * 1000003

    out = point_clouds.clone()
    batch_size = int(out.shape[0])
    base_seed = seed + _stable_int((batch_size, n, policy))
    scale_min = float(config.get("point_cloud_reframe_scale_min", 0.95) or 0.95)
    scale_max = float(config.get("point_cloud_reframe_scale_max", 1.05) or 1.05)
    translate = float(config.get("point_cloud_reframe_translate", 0.03) or 0.0)
    jitter_sigma = float(config.get("point_cloud_reframe_jitter_sigma", 0.01) or 0.0)
    jitter_clip = float(config.get("point_cloud_reframe_jitter_clip", 0.03) or 0.0)
    downsample_min = float(config.get("point_cloud_reframe_downsample_min", 0.60) or 0.60)
    downsample_max = float(config.get("point_cloud_reframe_downsample_max", 0.85) or 0.85)
    renormalize = _as_bool(config.get("point_cloud_reframe_renormalize", False), default=False)

    for row in range(batch_size):
        repeat_idx = row % n
        view_id = min(repeat_idx // samples_per_view, num_views - 1)
        if view_id <= 0:
            continue
        generator = torch.Generator(device=out.device)
        generator.manual_seed(base_seed + row * 9176 + view_id * 101)
        row_policy = _policy_for_view(policy, view_id, seed)

        if row_policy in {"sparsity", "composite"}:
            ratio = downsample_min + (downsample_max - downsample_min) * torch.rand(
                (), device=out.device, generator=generator
            ).item()
            out[row] = _apply_downsample(out[row], ratio, generator)

        if row_policy in {"sensor_noise", "rigid_pose", "composite"}:
            scale = scale_min + (scale_max - scale_min) * torch.rand((), device=out.device, generator=generator).item()
            out[row, :, :3].mul_(scale)

        if row_policy in {"rigid_pose", "composite"} and translate > 0:
            shift = torch.empty((1, 3), device=out.device, dtype=out.dtype).uniform_(
                -translate, translate, generator=generator
            )
            out[row, :, :3].add_(shift)

        if row_policy in {"sensor_noise", "composite"} and jitter_sigma > 0:
            noise = torch.randn(out[row, :, :3].shape, device=out.device, dtype=out.dtype, generator=generator)
            noise.mul_(jitter_sigma)
            if jitter_clip > 0:
                noise.clamp_(-jitter_clip, jitter_clip)
            out[row, :, :3].add_(noise)

    if renormalize:
        _normalize_xyz_inplace(out)
    return out


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto, n: int = None) -> DataProto:
        if n is None:
            n = 1 if prompts.meta_info.get("validate", False) else self.config.get("n", 1)
        if n > 1:
            prompts = prompts.repeat(repeat_times=n, interleave=True)
            if "point_clouds" in prompts.batch:
                prompts.batch["point_clouds"] = _apply_point_cloud_reframes(
                    prompts.batch["point_clouds"], n=n, config=self.config, meta_info=prompts.meta_info
                )
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)
        generate_kwargs = {}
        model_type = _get_model_type(self.module)
        if "point_clouds" in prompts.batch:
            try:
                model_dtype = next(self.module.parameters()).dtype
            except StopIteration:
                model_dtype = torch.bfloat16
            for key in _point_cloud_model_keys(model_type):
                if key not in prompts.batch:
                    continue
                tensor = prompts.batch[key].to(device=idx.device)
                if key == "point_clouds":
                    tensor = tensor.to(dtype=model_dtype)
                else:
                    tensor = tensor.to(dtype=torch.float32)
                generate_kwargs[key] = tensor

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", self.config.get("top_k", 0))

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)  # to be compatible with vllm

        temperature = prompts.meta_info.get("temperature", self.config.temperature)

        sampling_kwargs = {}
        if do_sample:
            sampling_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

        if isinstance(self.module, FSDP):
            recurse = model_type in {"pointllm", "ll3da"}
            # Point-cloud models use non-LLM backbones during generation, so recursively expose wrapped params.
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=recurse)
        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    # max_length=max_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    # renormalize_logits=True,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                    **sampling_kwargs,
                    **generate_kwargs,
                )
        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (bs, prompt_length)
        response = seq[:, prompt_length:]  # (bs, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        for key in _point_cloud_model_keys(model_type):
            if key in prompts.batch:
                batch[key] = prompts.batch[key]

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch)
