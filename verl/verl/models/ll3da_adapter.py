import contextlib
import os
import sys
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace
from typing import Iterable

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


@contextlib.contextmanager
def _prefer_ll3da_datasets_package(ll3da_root: str):
    """Let LL3DA import its local datasets package despite HF datasets already being loaded."""
    saved_datasets = sys.modules.pop("datasets", None)
    helper_paths = [
        ll3da_root,
        os.path.join(ll3da_root, "third_party", "pointnet2"),
        os.path.join(ll3da_root, "utils"),
    ]
    for path in reversed(helper_paths):
        sys.path.insert(0, path)
    cwd = os.getcwd()
    os.chdir(ll3da_root)
    try:
        yield
    finally:
        os.chdir(cwd)
        for path in helper_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        if saved_datasets is not None:
            sys.modules["datasets"] = saved_datasets


def _strip_checkpoint_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "module", "state_dict", "model_state_dict", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _checkpoint_bucket(key: str) -> str:
    if key.startswith("detector."):
        return "detector"
    if key.startswith("captioner.qformer."):
        return "qformer"
    if key.startswith("captioner.transformer."):
        return "opt_transformer"
    if key.startswith(
        (
            "captioner.encoder_to_qformer_projection.",
            "captioner.prompt_encoder.",
            "captioner.latent_query.",
            "captioner.qformer_to_language_projection.",
        )
    ):
        return "qformer_projection"
    if key.startswith("captioner."):
        return "captioner_other"
    return key.split(".", 1)[0]


def _summarize_keys(keys: Iterable[str]) -> dict[str, int]:
    summary = defaultdict(int)
    for key in keys:
        summary[_checkpoint_bucket(key)] += 1
    return dict(sorted(summary.items()))


def _load_matching_state_dict(module: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_checkpoint_prefixes(_extract_state_dict(checkpoint))
    model_state = module.state_dict()
    matched = {}
    skipped = []
    shape_mismatch = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
            if key in model_state and hasattr(value, "shape"):
                shape_mismatch.append(
                    {
                        "key": key,
                        "checkpoint": tuple(value.shape),
                        "model": tuple(model_state[key].shape),
                    }
                )
    missing, unexpected = module.load_state_dict(matched, strict=False)
    return {
        "loaded": len(matched),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "loaded_by_module": _summarize_keys(matched.keys()),
        "missing_by_module": _summarize_keys(missing),
        "skipped_by_module": _summarize_keys(skipped),
        "skipped_examples": skipped[:20],
        "shape_mismatch_examples": shape_mismatch[:20],
    }


def _as_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


class LL3DAForCausalLMAdapter(nn.Module):
    """Small HF-style wrapper around Open3DA/LL3DA CaptionNet for veRL PPO."""

    def __init__(self, caption_net: nn.Module, tokenizer, qtokenizer, config, torch_dtype, max_prompts: int = 1):
        super().__init__()
        self.caption_net = caption_net
        self.tokenizer = tokenizer
        self.qtokenizer = qtokenizer
        self.config = config
        self.config.model_type = "ll3da"
        self.torch_dtype = torch_dtype
        self.max_prompts = max(1, int(max_prompts))
        self.train_visual_prefix = False

    @property
    def detector(self):
        return self.caption_net.detector

    @property
    def captioner(self):
        return self.caption_net.captioner

    def get_input_embeddings(self):
        return self.captioner.transformer.get_input_embeddings()

    def enable_input_require_grads(self):
        if hasattr(self.captioner.transformer, "enable_input_require_grads"):
            self.captioner.transformer.enable_input_require_grads()

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.captioner.transformer, "gradient_checkpointing_enable"):
            self.captioner.transformer.gradient_checkpointing_enable(*args, **kwargs)

    def _decode_instruction_texts(self, input_ids, attention_mask=None, response_length=None):
        if response_length is not None and response_length > 0:
            input_ids = input_ids[:, :-response_length]
            attention_mask = attention_mask[:, :-response_length] if attention_mask is not None else None
        texts = []
        for row_idx, ids in enumerate(input_ids):
            if attention_mask is not None:
                valid = attention_mask[row_idx].bool()
                ids = ids[valid]
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
        return texts

    def _tokenize_for_qformer(self, texts: list[str], device):
        encoded = self.qtokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        return {
            "qformer_input_ids": encoded["input_ids"].to(device=device),
            "qformer_attention_mask": encoded["attention_mask"].to(device=device, dtype=torch.float32),
        }

    def _default_prompt_tensors(self, point_clouds):
        batch_size = point_clouds.shape[0]
        device = point_clouds.device
        return {
            "box_query": torch.zeros((batch_size, self.max_prompts, 8, 3), device=device, dtype=torch.float32),
            "box_mask": torch.zeros((batch_size, self.max_prompts), device=device, dtype=torch.float32),
            "click_query": torch.zeros((batch_size, self.max_prompts, 3), device=device, dtype=torch.float32),
            "click_mask": torch.zeros((batch_size, self.max_prompts), device=device, dtype=torch.float32),
        }

    def _prepare_prompt_tensor(self, value, point_clouds, target_shape):
        if value is None:
            return None
        tensor = value.to(device=point_clouds.device, dtype=torch.float32)
        if tuple(tensor.shape[1:]) != target_shape:
            raise ValueError(f"Unexpected LL3DA prompt tensor shape {tuple(tensor.shape)}; expected [B,{','.join(map(str, target_shape))}]")
        return tensor

    @staticmethod
    def _has_active_prompt(prompt_mask) -> bool:
        if prompt_mask is None:
            return False
        return bool(prompt_mask.detach().float().sum().item() > 0)

    def _build_ll3da_inputs(
        self,
        point_clouds,
        instruction_texts,
        box_query=None,
        box_mask=None,
        click_query=None,
        click_mask=None,
    ):
        device = point_clouds.device
        point_clouds = point_clouds.float()
        dims_min = point_clouds[..., :3].amin(dim=1)
        dims_max = point_clouds[..., :3].amax(dim=1)
        prompt_tensors = self._default_prompt_tensors(point_clouds)
        for key, value, shape in (
            ("box_query", box_query, (self.max_prompts, 8, 3)),
            ("box_mask", box_mask, (self.max_prompts,)),
            ("click_query", click_query, (self.max_prompts, 3)),
            ("click_mask", click_mask, (self.max_prompts,)),
        ):
            prepared = self._prepare_prompt_tensor(value, point_clouds, shape)
            if prepared is not None:
                prompt_tensors[key] = prepared
        inputs = {
            "point_clouds": point_clouds,
            "point_cloud_dims_min": dims_min,
            "point_cloud_dims_max": dims_max,
            **prompt_tensors,
        }
        inputs.update(self._tokenize_for_qformer(instruction_texts, device=device))
        return inputs

    def _visual_prefix(self, point_clouds, instruction_texts, box_query=None, box_mask=None, click_query=None, click_mask=None):
        inputs = self._build_ll3da_inputs(
            point_clouds,
            instruction_texts,
            box_query=box_query,
            box_mask=box_mask,
            click_query=click_query,
            click_mask=click_mask,
        )
        with torch.no_grad():
            detector_output = self.detector(inputs, is_eval=True)
        use_box_prompt = self._has_active_prompt(inputs["box_mask"])
        use_click_prompt = self._has_active_prompt(inputs["click_mask"])

        grad_context = contextlib.nullcontext() if self.train_visual_prefix and torch.is_grad_enabled() else torch.no_grad()
        with grad_context:
            prefix_tokens = self.captioner._get_instruction_response(
                detector_output,
                inputs,
                box_query=inputs["box_query"] if use_box_prompt else None,
                box_qmask=inputs["box_mask"] if use_box_prompt else None,
                click_query=inputs["click_query"] if use_click_prompt else None,
                click_qmask=inputs["click_mask"] if use_click_prompt else None,
            )
        prefix_tokens = prefix_tokens.to(dtype=self.torch_dtype)
        if not (self.train_visual_prefix and torch.is_grad_enabled()):
            prefix_tokens = prefix_tokens.detach()
        return prefix_tokens

    def _generation_prefix_inputs(self, input_ids, attention_mask, prefix_tokens):
        embedding_layer = self.get_input_embeddings()
        batch_embeds = []
        batch_lengths = []
        for row_idx, ids in enumerate(input_ids):
            if attention_mask is not None:
                valid_ids = ids[attention_mask[row_idx].bool()]
            else:
                valid_ids = ids
            token_embeds = embedding_layer(valid_ids.unsqueeze(0)).squeeze(0)
            row_embeds = torch.cat([prefix_tokens[row_idx], token_embeds], dim=0).to(dtype=self.torch_dtype)
            batch_embeds.append(row_embeds)
            batch_lengths.append(row_embeds.shape[0])

        max_length = max(batch_lengths)
        hidden_size = batch_embeds[0].shape[-1]
        inputs_embeds = torch.zeros(
            (len(batch_embeds), max_length, hidden_size),
            device=input_ids.device,
            dtype=self.torch_dtype,
        )
        prefix_attention_mask = torch.zeros(
            (len(batch_embeds), max_length),
            device=input_ids.device,
            dtype=attention_mask.dtype if attention_mask is not None else torch.long,
        )
        for row_idx, row_embeds in enumerate(batch_embeds):
            row_length = row_embeds.shape[0]
            inputs_embeds[row_idx, -row_length:] = row_embeds
            prefix_attention_mask[row_idx, -row_length:] = 1
        return inputs_embeds, prefix_attention_mask

    @staticmethod
    def _sample_next_token(logits, do_sample: bool, temperature: float, top_p: float, top_k: int):
        if not do_sample or temperature is None or temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / max(float(temperature), 1e-6)
        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), logits.shape[-1])
            threshold = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)

        if top_p is not None and 0 < float(top_p) < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > float(top_p)
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove, torch.finfo(logits.dtype).min)

        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all():
            return logits.argmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _decode_from_embeds(
        self,
        inputs_embeds,
        attention_mask,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_token_id: int,
    ):
        generated = torch.full(
            (inputs_embeds.shape[0], max_new_tokens),
            fill_value=eos_token_id,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        finished = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)
        current_attention_mask = attention_mask
        next_input_ids = None
        past_key_values = None

        for token_idx in range(max_new_tokens):
            if token_idx == 0:
                outputs = self.captioner.transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask=current_attention_mask,
                    use_cache=True,
                )
            else:
                outputs = self.captioner.transformer(
                    input_ids=next_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            next_token = self._sample_next_token(logits, do_sample, temperature, top_p, top_k)
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            generated[:, token_idx] = next_token.long()
            finished |= next_token.eq(eos_token_id)

            next_mask = torch.ones(
                (current_attention_mask.shape[0], 1),
                dtype=current_attention_mask.dtype,
                device=current_attention_mask.device,
            )
            current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)
            next_input_ids = next_token.unsqueeze(1)

            if bool(finished.all()):
                break

        return generated

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        point_clouds=None,
        box_query=None,
        box_mask=None,
        click_query=None,
        click_mask=None,
        response_length=None,
        use_cache=False,
        **kwargs,
    ):
        if point_clouds is None:
            return self.captioner.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )

        instruction_texts = self._decode_instruction_texts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_length=response_length,
        )
        prefix_tokens = self._visual_prefix(
            point_clouds,
            instruction_texts,
            box_query=box_query,
            box_mask=box_mask,
            click_query=click_query,
            click_mask=click_mask,
        )
        prefix_mask = torch.ones(prefix_tokens.shape[:2], device=input_ids.device, dtype=attention_mask.dtype)
        token_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_tokens, token_embeds], dim=1).to(dtype=self.torch_dtype)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = self.captioner.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        prefix_len = prefix_tokens.shape[1]
        return CausalLMOutputWithPast(logits=outputs.logits[:, prefix_len: prefix_len + input_ids.shape[1], :])

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        point_clouds=None,
        box_query=None,
        box_mask=None,
        click_query=None,
        click_mask=None,
        max_new_tokens=16,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        eos_token_id=None,
        pad_token_id=None,
        return_dict_in_generate=True,
        **kwargs,
    ):
        if point_clouds is None:
            return self.captioner.transformer.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

        generation_config = kwargs.pop("generation_config", None)
        if generation_config is not None:
            temperature = getattr(generation_config, "temperature", temperature)
            top_p = getattr(generation_config, "top_p", top_p)
            top_k = getattr(generation_config, "top_k", top_k)

        instruction_texts = self._decode_instruction_texts(input_ids=input_ids, attention_mask=attention_mask)
        prefix_tokens = self._visual_prefix(
            point_clouds,
            instruction_texts,
            box_query=box_query,
            box_mask=box_mask,
            click_query=click_query,
            click_mask=click_mask,
        )
        inputs_embeds, prefix_attention_mask = self._generation_prefix_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_tokens=prefix_tokens,
        )
        response = self._decode_from_embeds(
            inputs_embeds=inputs_embeds,
            attention_mask=prefix_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=float(temperature) if temperature is not None else 1.0,
            top_p=float(top_p) if top_p is not None else 1.0,
            top_k=int(top_k or 0),
            eos_token_id=eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id,
        )
        sequences = torch.cat([input_ids, response[:, :max_new_tokens]], dim=1)
        return SimpleNamespace(sequences=sequences)


def _ll3da_args(model_config, llm_model_path: str):
    qformer_vocab = model_config.get("ll3da_qformer_vocab", "bert-base-embedding")
    return Namespace(
        use_color=_as_bool(model_config.get("ll3da_use_color"), True),
        use_normal=_as_bool(model_config.get("ll3da_use_normal"), True),
        use_height=not _as_bool(model_config.get("ll3da_no_height"), False),
        no_height=_as_bool(model_config.get("ll3da_no_height"), False),
        use_multiview=_as_bool(model_config.get("ll3da_use_multiview"), False),
        detector=model_config.get("ll3da_detector", "detector_Vote2Cap_DETR"),
        captioner=model_config.get("ll3da_captioner", "ll3da"),
        freeze_detector=True,
        freeze_llm=False,
        use_beam_search=False,
        max_des_len=256,
        max_gen_len=16,
        max_prompts=int(model_config.get("ll3da_max_prompts", 1)),
        grid_size_3d=int(model_config.get("ll3da_grid_size_3d", 255)),
        vocab=llm_model_path,
        qformer_vocab=qformer_vocab,
        batchsize_per_gpu=1,
        dataset="scannet",
        pretrained_weights=model_config.get("ll3da_detector_ckpt_path", None),
    )


def _ll3da_trainable_prefixes(train_scope: str) -> tuple[tuple[str, ...], str]:
    scope = (train_scope or "llm").lower()
    llm_prefix = "caption_net.captioner.transformer."
    qformer_projector_prefixes = (
        "caption_net.captioner.qformer.",
        "caption_net.captioner.encoder_to_qformer_projection.",
        "caption_net.captioner.latent_query.",
        "caption_net.captioner.qformer_to_language_projection.",
    )

    if scope == "llm":
        return (llm_prefix,), "LLM"
    if scope in {"qformer_projector_llm", "qformer+projector+llm", "llm_qformer_projector"}:
        return qformer_projector_prefixes + (llm_prefix,), "Q-Former/projector + LLM"

    raise ValueError(
        "Unsupported ll3da_train_scope="
        f"{train_scope!r}; expected 'llm' or 'qformer_projector_llm'."
    )


def _freeze_except_prefixes(module: nn.Module, trainable_prefixes: Iterable[str]):
    trainable_count = 0
    frozen_count = 0
    for name, param in module.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            param.requires_grad_(True)
            trainable_count += param.numel()
        else:
            param.requires_grad_(False)
            frozen_count += param.numel()
    return trainable_count, frozen_count


def build_ll3da_adapter(model_config, llm_model_path: str, actor_model_config, torch_dtype):
    repo_path = os.path.abspath(os.path.expanduser(model_config.get("ll3da_repo_path", "LL3DA")))
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"LL3DA repo not found at {repo_path}. Clone https://github.com/Open3DA/LL3DA there or set ll3da_repo_path."
        )

    with _prefer_ll3da_datasets_package(repo_path):
        from datasets.scannet_base_dataset import DatasetConfig
        from models.model_general import CaptionNet

        args = _ll3da_args(model_config, llm_model_path)
        qformer_candidate = os.path.join(repo_path, args.qformer_vocab)
        if not os.path.isabs(args.qformer_vocab) and os.path.isdir(qformer_candidate):
            args.qformer_vocab = qformer_candidate
        dataset_config = DatasetConfig()
        caption_net = CaptionNet(args, dataset_config, train_dataset=None)

    qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
    adapter = LL3DAForCausalLMAdapter(
        caption_net,
        caption_net.captioner.tokenizer,
        qtokenizer,
        actor_model_config,
        torch_dtype,
        max_prompts=args.max_prompts,
    )

    ckpt_path = model_config.get("ll3da_ckpt_path", None)
    if ckpt_path:
        load_info = _load_matching_state_dict(adapter.caption_net, os.path.expanduser(ckpt_path))
        print(f"Loaded LL3DA checkpoint {ckpt_path}: {load_info}")

    train_scope = model_config.get("ll3da_train_scope", "llm")
    trainable_prefixes, trainable_label = _ll3da_trainable_prefixes(train_scope)
    trainable_count, frozen_count = _freeze_except_prefixes(adapter, trainable_prefixes)
    adapter.train_visual_prefix = str(train_scope).lower() != "llm"
    print(f"LL3DA train scope: {train_scope}")
    print(f"LL3DA trainable prefixes: {trainable_prefixes}")
    print(f"LL3DA trainable params ({trainable_label}): {trainable_count / 1e9:.2f}B")
    print(f"LL3DA frozen params: {frozen_count / 1e6:.2f}M")
    return adapter
