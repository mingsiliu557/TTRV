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
Single Process Actor
"""
import re
import math
import itertools
from typing import Dict, Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor


def _safe_agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    if torch.sum(loss_mask) <= 0:
        return (loss_mat * 0.0).sum()
    if loss_agg_mode == "token-mean":
        return verl_F.masked_mean(loss_mat, loss_mask)
    if loss_agg_mode == "seq-mean-token-sum":
        valid_seq = torch.sum(loss_mask, dim=-1) > 0
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.mean(seq_losses[valid_seq])
    if loss_agg_mode == "seq-mean-token-mean":
        token_count = torch.sum(loss_mask, dim=-1)
        valid_seq = token_count > 0
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / token_count.clamp_min(1.0)
        return torch.mean(seq_losses[valid_seq])
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, return_response_embeddings=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                image_flags = None
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )
                    if re.match("internvl", self.actor_module.config.model_type):
                        # The image_flags is used for InternVL's github version
                        if key == "pixel_values":
                            image_flags = torch.ones(multi_modal_inputs[key].size(0), dtype=torch.long)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            response_embeddings = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.preprocessor.minicpmo import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )
                
                # extra_args = {}
                # if self.use_fused_kernels:
                #     extra_args["temperature"] = temperature
                #     extra_args["return_dict"] = True
                if image_flags is not None:
                    multi_modal_inputs["image_flags"] = image_flags

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_response_embeddings,
                    return_dict=True,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward
                )

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                if return_response_embeddings:
                    hidden_rmpad = output.hidden_states[-1].squeeze(0)
                    if self.use_ulysses_sp:
                        hidden_rmpad = gather_outpus_and_unpad(
                            hidden_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    full_hidden = pad_input(
                        hidden_states=hidden_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                    response_hidden = full_hidden[:, -response_length:, :]
                    response_token_mask = attention_mask[:, -response_length:].to(response_hidden.dtype).unsqueeze(-1)
                    pooled = (response_hidden * response_token_mask).sum(dim=1) / response_token_mask.sum(dim=1).clamp_min(1.0)
                    response_embeddings = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_response_embeddings,
                    return_dict=True,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                if return_response_embeddings:
                    response_hidden = output.hidden_states[-1][:, -response_length:, :]
                    response_token_mask = attention_mask[:, -response_length:].to(response_hidden.dtype).unsqueeze(-1)
                    pooled = (response_hidden * response_token_mask).sum(dim=1) / response_token_mask.sum(dim=1).clamp_min(1.0)
                    response_embeddings = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)

            return entropy, log_probs, response_embeddings

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @staticmethod
    def _build_attention_cf_mask(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_context_token_id: int,
        ratio: float,
        cut_inter_visual: bool,
        dtype: torch.dtype,
        style: str = "hard_cut",
        scale: float = 0.3,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Build a 4D additive causal mask with text-query -> image-key links weakened.

        The returned mask follows the HuggingFace Qwen2 4D additive attention
        mask contract. ``hard_cut`` preserves the old behavior by setting
        selected links to dtype min. ``soft_scale`` adds log(scale), which
        down-weights selected image-key links before softmax without deleting
        the visual path completely.
        """
        if input_ids.dim() != 2 or attention_mask.dim() != 2:
            raise ValueError("attention counterfactual expects 2D input_ids and attention_mask")
        if not (0.0 <= ratio < 1.0):
            raise ValueError(f"papo_attn_cf_ratio must be in [0, 1), got {ratio}")
        style = str(style).lower().strip()
        if style not in ("hard_cut", "soft_scale"):
            raise ValueError(f"Unsupported papo_attn_cf_style {style!r}; expected hard_cut or soft_scale")
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"papo_attn_cf_scale must be in (0, 1], got {scale}")

        batch_size, seqlen = input_ids.shape
        device = input_ids.device
        min_dtype = torch.finfo(dtype).min
        selected_bias = min_dtype if style == "hard_cut" else float(math.log(scale))

        cf_mask = torch.zeros((batch_size, 1, seqlen, seqlen), dtype=dtype, device=device)
        future_mask = torch.triu(torch.ones((seqlen, seqlen), dtype=torch.bool, device=device), diagonal=1)
        cf_mask = cf_mask.masked_fill(future_mask.view(1, 1, seqlen, seqlen), min_dtype)
        key_padding_mask = attention_mask.to(dtype=torch.bool).logical_not()
        cf_mask = cf_mask.masked_fill(key_padding_mask.view(batch_size, 1, 1, seqlen), min_dtype)

        image_counts = []
        selected_counts = []
        text_counts = []
        no_image = 0
        for row_idx in range(batch_size):
            valid_mask = attention_mask[row_idx].to(dtype=torch.bool)
            image_idx = torch.nonzero(
                (input_ids[row_idx] == img_context_token_id) & valid_mask,
                as_tuple=False,
            ).flatten()
            text_idx = torch.nonzero(valid_mask & (input_ids[row_idx] != img_context_token_id), as_tuple=False).flatten()
            num_image = int(image_idx.numel())
            num_text = int(text_idx.numel())
            image_counts.append(float(num_image))
            text_counts.append(float(num_text))
            if num_image == 0 or num_text == 0 or ratio <= 0.0:
                selected_counts.append(0.0)
                if num_image == 0:
                    no_image += 1
                continue

            selected_count = min(num_image, max(1, int(round(num_image * ratio))))
            # Deterministic uniform subset: cheap and reproducible, while avoiding
            # the extra attention-ranking forward needed for top-attention selection.
            uniform_offsets = torch.div(
                torch.arange(selected_count, device=device) * num_image,
                selected_count,
                rounding_mode="floor",
            )
            selected_image_idx = image_idx[uniform_offsets]
            selected_counts.append(float(selected_count))
            query_idx = image_idx if cut_inter_visual else text_idx
            if query_idx.numel() > 0:
                cf_mask[row_idx, 0, query_idx[:, None], selected_image_idx[None, :]] = selected_bias

        total_image = sum(image_counts)
        total_selected = sum(selected_counts)
        metrics = {
            "papo_attn_cf_image_token_count_mean": sum(image_counts) / max(len(image_counts), 1),
            "papo_attn_cf_text_token_count_mean": sum(text_counts) / max(len(text_counts), 1),
            "papo_attn_cf_selected_image_ratio": total_selected / max(total_image, 1.0),
            "papo_attn_cf_no_image_ratio": float(no_image) / max(batch_size, 1),
            "papo_attn_cf_soft_scale_active": float(style == "soft_scale"),
            "papo_attn_cf_scale": float(scale),
        }
        return cf_mask, metrics

    def _forward_attn_cf_micro_batch(
        self,
        micro_batch,
        temperature: float,
        ratio: float,
        cut_inter_visual: bool,
        style: str,
        scale: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                image_flags = None
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )
                    if re.match("internvl", self.actor_module.config.model_type):
                        if key == "pixel_values":
                            image_flags = torch.ones(
                                multi_modal_inputs[key].size(0),
                                dtype=torch.long,
                                device=multi_modal_inputs[key].device,
                            )
                if image_flags is not None:
                    multi_modal_inputs["image_flags"] = image_flags

        attr_module = getattr(self.actor_module, "module", self.actor_module)
        img_context_token_id = getattr(self.actor_module, "img_context_token_id", None)
        if img_context_token_id is None:
            img_context_token_id = getattr(attr_module, "img_context_token_id", None)
        if img_context_token_id is None:
            raise ValueError("attention counterfactual requires a non-null img_context_token_id")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask_2d = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            cf_attention_mask, mask_metrics = self._build_attention_cf_mask(
                input_ids=input_ids,
                attention_mask=attention_mask_2d,
                img_context_token_id=img_context_token_id,
                ratio=ratio,
                cut_inter_visual=cut_inter_visual,
                dtype=torch.bfloat16,
                style=style,
                scale=scale,
            )
            attr_module = getattr(self.actor_module, "module", self.actor_module)
            language_model = getattr(attr_module, "language_model", None)
            attn_impl_stack = []
            for module in (attr_module, language_model, getattr(language_model, "model", None)):
                config = getattr(module, "config", None)
                if config is not None and hasattr(config, "_attn_implementation"):
                    attn_impl_stack.append((config, config._attn_implementation))
                    config._attn_implementation = "eager"
            try:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=cf_attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            finally:
                for config, old_impl in reversed(attn_impl_stack):
                    config._attn_implementation = old_impl
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]
            log_probs = logprobs_from_logits(logits, micro_batch["responses"])
            return log_probs, mask_metrics

    def compute_attn_cf_log_prob(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute response log-probs under an attention-space image counterfactual.

        This intentionally bypasses the remove-padding/varlen path so a normal
        4D additive attention mask can be injected into the LLM.
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        ratio = float(data.meta_info.get("papo_attn_cf_ratio", self.config.get("papo_attn_cf_ratio", 0.6)))
        cut_inter_visual = bool(
            data.meta_info.get("papo_attn_cf_cut_iv", self.config.get("papo_attn_cf_cut_iv", False))
        )
        style = str(data.meta_info.get("papo_attn_cf_style", self.config.get("papo_attn_cf_style", "hard_cut")))
        scale = float(data.meta_info.get("papo_attn_cf_scale", self.config.get("papo_attn_cf_scale", 0.3)))

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        batch_size = data.batch["responses"].size(0)
        if has_multi_modal_inputs:
            micro_batches = data.select(select_keys, ["multi_modal_inputs"]).chunk(batch_size)
        else:
            micro_batches = data.select(batch_keys=select_keys).batch.split(1)

        log_probs_lst = []
        metric_sums: Dict[str, float] = {}
        metric_count = 0
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            else:
                micro_batch = micro_batch.to(torch.cuda.current_device())

            with torch.no_grad():
                log_probs, mask_metrics = self._forward_attn_cf_micro_batch(
                    micro_batch=micro_batch,
                    temperature=temperature,
                    ratio=ratio,
                    cut_inter_visual=cut_inter_visual,
                    style=style,
                    scale=scale,
                )
            log_probs_lst.append(log_probs)
            for key, value in mask_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            metric_count += 1

        log_probs = torch.concat(log_probs_lst, dim=0)
        metrics = {key: value / max(metric_count, 1) for key, value in metric_sums.items()}
        metrics["papo_attn_cf_ratio"] = ratio
        metrics["papo_attn_cf_cut_iv"] = float(cut_inter_visual)
        metrics["papo_attn_cf_style_soft_scale"] = float(str(style).lower().strip() == "soft_scale")
        metrics["papo_attn_cf_scale"] = scale
        return log_probs, metrics

    def compute_log_prob(self, data: DataProto, calculate_entropy=False, return_response_embeddings=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        response_embeddings_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            response_mask = micro_batch["attention_mask"][:, -micro_batch["responses"].size(-1) :]
            with torch.no_grad():
                entropy, log_probs, response_embeddings = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_response_embeddings=return_response_embeddings,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if return_response_embeddings:
                response_embeddings_lst.append(response_embeddings)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        response_embeddings = None
        if return_response_embeddings:
            response_embeddings = torch.concat(response_embeddings_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]
            if response_embeddings is not None:
                response_embeddings = response_embeddings[revert_indices]

        if return_response_embeddings:
            return log_probs, entropys, response_embeddings
        return log_probs, entropys

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error

        use_papo_prcp_loss = bool(self.config.get("use_papo_prcp_loss", False))
        use_papo_valid_only = use_papo_prcp_loss and bool(self.config.get("papo_valid_only", False))
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if use_papo_prcp_loss:
            select_keys.append("aug_log_probs")
            if use_papo_valid_only:
                select_keys.append("papo_valid_response_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    papo_ori_entropy_coef = self.config.get("papo_ori_entropy_coef", 0.0) if use_papo_prcp_loss else 0.0
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, _ = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode
                        )

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if use_papo_prcp_loss:
                        aug_log_prob = data["aug_log_probs"]
                        papo_loss_mask = response_mask
                        if use_papo_valid_only:
                            papo_loss_mask = data["papo_valid_response_mask"].to(response_mask.device)
                            papo_loss_mask = papo_loss_mask.to(response_mask.dtype) * response_mask
                            metrics["actor/papo_valid_token_ratio"] = verl_F.masked_mean(
                                (papo_loss_mask > 0).float(), response_mask
                            ).detach().item()

                        log_ratio = torch.clamp(log_prob - aug_log_prob, min=-20.0, max=20.0)
                        kl_prcp_raw_mat = torch.exp(log_ratio) - log_ratio - 1.0
                        kl_prcp_raw_mat = torch.clamp(kl_prcp_raw_mat, min=0.0)
                        kl_prcp_clip = self.config.get("papo_kl_prcp_clip", 0.2)
                        kl_prcp_mat = torch.clamp(kl_prcp_raw_mat, min=0.0, max=kl_prcp_clip)
                        kl_prcp_raw = _safe_agg_loss(
                            loss_mat=kl_prcp_raw_mat,
                            loss_mask=papo_loss_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        kl_prcp = _safe_agg_loss(
                            loss_mat=kl_prcp_mat,
                            loss_mask=papo_loss_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        kl_prcp_clip_frac = verl_F.masked_mean(
                            (kl_prcp_raw_mat > kl_prcp_clip).float(), papo_loss_mask
                        )
                        log_ratio_mean = _safe_agg_loss(
                            loss_mat=log_ratio,
                            loss_mask=papo_loss_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        log_ratio_abs_mean = _safe_agg_loss(
                            loss_mat=log_ratio.abs(),
                            loss_mask=papo_loss_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        policy_loss = policy_loss - self.config.get("papo_kl_prcp_coef", 0.01) * kl_prcp
                        metrics["actor/kl_prcp_raw"] = kl_prcp_raw.detach().item()
                        metrics["actor/kl_prcp_clipped"] = kl_prcp.detach().item()
                        metrics["actor/kl_prcp_clip_frac"] = kl_prcp_clip_frac.detach().item()
                        metrics["actor/papo_log_ratio_mean"] = log_ratio_mean.detach().item()
                        metrics["actor/papo_log_ratio_abs_mean"] = log_ratio_abs_mean.detach().item()
                        metrics["actor/papo_orig_logprob_mean"] = _safe_agg_loss(
                            loss_mat=log_prob,
                            loss_mask=papo_loss_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                        ).detach().item()
                        metrics["actor/papo_masked_logprob_mean"] = _safe_agg_loss(
                            loss_mat=aug_log_prob,
                            loss_mask=papo_loss_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                        ).detach().item()

                        if papo_ori_entropy_coef != 0:
                            papo_ori_nll = _safe_agg_loss(
                                loss_mat=-log_prob,
                                loss_mask=papo_loss_mask,
                                loss_agg_mode=self.config.loss_agg_mode,
                            )
                            policy_loss = policy_loss + papo_ori_entropy_coef * papo_ori_nll
                            metrics["actor/papo_ori_nll"] = papo_ori_nll.detach().item()
                            metrics["actor/papo_ori_nll_coef"] = papo_ori_entropy_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics