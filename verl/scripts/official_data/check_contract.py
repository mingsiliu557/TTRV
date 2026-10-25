"""Run the existing preprocessor on every row, one image at a time, on the GPU node.

No CPU model weights are loaded. Optional multimodal forward uses GPU-first loader.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=7524)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from scripts.official_data.prepare import digest
    assert torch.cuda.is_available(), "Run this check inside the existing GPU allocation"
    root = Path(args.data_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest.json").read_text())
    tokenizer = hf_tokenizer(args.model, trust_remote_code=True, local_files_only=True)
    processor = hf_processor(args.model, use_fast=True, local_files_only=True)
    if processor is None:
        raise RuntimeError("Multimodal processor unavailable; refusing text-only fallback")
    config = OmegaConf.create(dict(prompt_key="prompt", image_key="images", video_key="videos",
                                   max_prompt_length=args.max_prompt_length, filter_overlong_prompts=False,
                                   truncation="error", return_full_prompt=True, filter_overlong_prompts_workers=1,
                                   cache_dir=str(output / "dataset_cache")))
    data = RLHFDataset(str(root / "test.parquet"), tokenizer, config, processor)
    assert len(data) == manifest["count"]
    seen, checked_images, max_length = set(), set(), 0
    first = None
    with open(output / "rendered_prompts.jsonl", "x") as stream:
        for i in range(len(data)):
            source = data.dataframe[i]
            if len(source["images"]) != 1 or source["prompt"][0]["content"].count("<image>") != 1:
                raise ValueError(f"Image/prompt contract violated at {i}")
            image = source["images"][0]["image"]
            if image not in checked_images:
                assert digest(image) == source["extra_info"]["image_sha256"]
                checked_images.add(image)
            item = data[i]
            ident = item["extra_info"]["index"]
            assert ident not in seen
            seen.add(ident)
            assert isinstance(item["multi_modal_inputs"], dict)
            assert item["input_ids"].dtype == torch.int64
            length = int(item["attention_mask"].sum())
            max_length = max(max_length, length)
            stream.write(json.dumps(dict(index=ident, length=length, rendered_text=item["full_prompts"],
                                         actual_input_ids=item["input_ids"].tolist()), ensure_ascii=False) + "\n")
            if first is None:
                first = item
                again = data[i]
                assert torch.equal(again["input_ids"], item["input_ids"]), "Repeated __getitem__ mutated source"
                collated = collate_fn([item, again])
                assert collated["input_ids"].shape[0] == 2
            del item
    assert seen == set(manifest["ids"])
    forward_finite = None
    if args.forward:
        from transformers import AutoConfig, AutoModel
        from verl.utils.gpu_checkpoint import load_model_on_gpu
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        model = load_model_on_gpu(AutoModel, args.model, config, torch.float32)
        model.eval()
        model.img_context_token_id = tokenizer.context_image_token_id
        mask = first["attention_mask"].bool()
        inputs = {"input_ids": first["input_ids"][mask].unsqueeze(0).cuda(),
                  "attention_mask": first["attention_mask"][mask].unsqueeze(0).cuda()}
        for key, value in first["multi_modal_inputs"].items():
            inputs[key] = value.cuda() if torch.is_tensor(value) else value
        if "pixel_values" in inputs:
            inputs["image_flags"] = torch.ones(inputs["pixel_values"].shape[0], 1, device="cuda", dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(**inputs, use_cache=False)
        forward_finite = bool(torch.isfinite(result.logits).all())
        assert forward_finite, "Nonfinite multimodal forward"
    (output / "input_contract_report.json").write_text(json.dumps(dict(count=len(data), max_input_length=max_length,
        full_coverage=True, repeated_getitem_stable=True, multimodal_forward_finite=forward_finite,
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        boundary="Existing tokenizer/processor preserved; full official inference equivalence not asserted"), indent=2))


if __name__ == "__main__":
    main()
