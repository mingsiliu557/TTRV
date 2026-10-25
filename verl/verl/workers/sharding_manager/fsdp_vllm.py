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

import inspect
import logging
import os
import types

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import log_gpu_memory_usage

from .base import BaseShardingManager
from .patch import patched_ds_v3_load_weights, patched_qwen_moe_load_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


def _normalize_qwen25_vl_weight_name(name: str) -> str:
    """Convert transformers 4.52 in-memory names to HF checkpoint names."""
    if name.startswith("model.visual."):
        return name[len("model.") :]
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    return name


def _qwen25_vl_process_image_input(self, image_input):
    """Process images without reusing the CUDA grid after vision execution.

    vLLM 0.7.3 can invalidate the multimodal grid storage after the vision
    forward when its sleep-mode allocator is used alongside FSDP. Preserve
    the split sizes as CPU integers before entering the vision tower.
    """
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2
    merge_size = self.visual.spatial_merge_size
    split_sizes = (
        grid_thw.detach().to(device="cpu", dtype=torch.int64).prod(-1) // merge_size // merge_size
    ).tolist()
    debug = os.environ.get("QWEN_VLLM_SYNC_DEBUG") == "1"
    if debug:
        print(
            "[qwen-vllm-debug] image input",
            {
                "type": image_input["type"],
                "grid_shape": tuple(grid_thw.shape),
                "grid_dtype": str(grid_thw.dtype),
                "grid_device": str(grid_thw.device),
                "split_sizes": split_sizes,
            },
            flush=True,
        )
        torch.cuda.synchronize()

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        if debug:
            print(
                "[qwen-vllm-debug] pixel values",
                {
                    "shape": tuple(pixel_values.shape),
                    "dtype": str(pixel_values.dtype),
                    "device": str(pixel_values.device),
                    "contiguous": pixel_values.is_contiguous(),
                },
                flush=True,
            )
            torch.cuda.synchronize()
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        if debug:
            torch.cuda.synchronize()
            print(
                "[qwen-vllm-debug] visual forward complete",
                {"shape": tuple(image_embeds.shape), "finite": bool(torch.isfinite(image_embeds).all().item())},
                flush=True,
            )

    return image_embeds.split(split_sizes)


def _install_qwen25_vl_compat_forward(model) -> None:
    if getattr(model, "_verl_qwen_vllm_compat_installed", False):
        return
    model._process_image_input = types.MethodType(_qwen25_vl_process_image_input, model)
    model._verl_qwen_vllm_compat_installed = True


def _validate_qwen25_vl_synced_weight(model, updated_params, world_size: int) -> None:
    if os.environ.get("QWEN_VLLM_SYNC_DEBUG") != "1":
        return
    source_name = "model.visual.patch_embed.proj.weight"
    target_name = _normalize_qwen25_vl_weight_name(source_name)
    source_param = updated_params.get(source_name)
    target_param = dict(model.named_parameters()).get(target_name)
    if source_param is None or target_param is None:
        print(
            "[qwen-vllm-debug] weight check unavailable",
            {"source_found": source_param is not None, "target_found": target_param is not None},
            flush=True,
        )
        return

    source = source_param.full_tensor() if world_size != 1 and hasattr(source_param, "full_tensor") else source_param
    source = source.detach().to(device=target_param.device, dtype=target_param.dtype)
    target = target_param.detach()
    torch.cuda.synchronize()
    print(
        "[qwen-vllm-debug] weight check",
        {
            "source_shape": tuple(source.shape),
            "target_shape": tuple(target.shape),
            "max_abs_diff": float((source - target).abs().max().item()),
            "source_finite": bool(torch.isfinite(source).all().item()),
            "target_finite": bool(torch.isfinite(target).all().item()),
        },
        flush=True,
    )
    del source


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(
                self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
            )
        else:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if vllm_version in ("0.4.2", "0.5.4", "0.6.3"):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params)
            torch.cuda.synchronize()
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            if model.config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
                _validate_qwen25_vl_synced_weight(model, params, torch.distributed.get_world_size())
                _install_qwen25_vl_compat_forward(model)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            torch.cuda.empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])
            torch.cuda.synchronize()

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before vllm offload in sharding manager", logger=logger)
        torch.cuda.synchronize()
        # TODO(ZSL): check this
        if vllm_version in ("0.4.2", "0.5.4", "0.6.3"):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
        torch.cuda.synchronize()
        log_gpu_memory_usage("After vllm offload in sharding manager", logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3"):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params):
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        world_size = torch.distributed.get_world_size()
        if model.config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
            loaded_params = patched_ds_v3_load_weights(
                model,
                (
                    (name, param.full_tensor() if world_size != 1 and hasattr(param, "full_tensor") else param)
                    for name, param in updated_params.items()
                ),
            )
        elif model.config.architectures[0] in ["Qwen2MoeForCausalLM"]:
            loaded_params = patched_qwen_moe_load_weights(
                model,
                (
                    (name, param.full_tensor() if world_size != 1 and hasattr(param, "full_tensor") else param)
                    for name, param in updated_params.items()
                ),
            )
        elif model.config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
            loaded_params = model.load_weights(
                (
                    (
                        _normalize_qwen25_vl_weight_name(name),
                        param.full_tensor() if world_size != 1 and hasattr(param, "full_tensor") else param,
                    )
                    for name, param in updated_params.items()
                )
            )
        else:
            loaded_params = model.load_weights(
                ((name, param.full_tensor() if world_size != 1 else param) for name, param in updated_params.items())
            )
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")
