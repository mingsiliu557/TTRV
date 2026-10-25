#!/usr/bin/env python3
"""Audit InternVL3 multimodal prompt lengths before Obs1/Obs3 jobs.

The audit uses the repository's real InternVL processor, chat template, and
official patch-grid rule. It computes every row efficiently from image
dimensions, then materializes the longest row in each scope with the real
processor to detect formula or processor drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoConfig
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas


SCRIPT_PATH = Path(__file__).resolve()
VERL_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
for import_root in (VERL_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from attention_visualization.attention_viz import (  # noqa: E402
    IMAGE_CONTEXT_TOKEN,
    _build_messages,
    _prepare_inputs,
    _row_qid,
)
from verl.utils.tokenizer import hf_processor  # noqa: E402


DEFAULT_MODEL = Path(
    "/workspace/ttrv_runtime/models/OpenGVLab/InternVL3-2B"
)


class AuditError(RuntimeError):
    """Raised when audit inputs or processor assumptions are invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs1-dir",
        type=Path,
        help="Directory containing the exact Obs1 train.parquet and test.parquet.",
    )
    parser.add_argument(
        "--obs3-candidates",
        type=Path,
        help="Obs3 candidate JSONL; every non-empty line is audited.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-limit", type=int, default=7524)
    parser.add_argument("--model-len", type=int, default=8548)
    parser.add_argument("--obs1-max-new", type=int, default=1024)
    parser.add_argument("--obs3-max-new", type=int, default=16)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.obs1_dir is None and args.obs3_candidates is None:
        parser.error("at least one of --obs1-dir or --obs3-candidates is required")
    for name in ("prompt_limit", "model_len", "obs1_max_new", "obs3_max_new"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AuditError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise AuditError(f"No candidate records found in {path}")
    return records


def load_obs3_rows(candidates_path: Path) -> list[dict[str, Any]]:
    candidates = read_jsonl(candidates_path)
    table_cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []
    for line_number, candidate in enumerate(candidates, start=1):
        try:
            parquet_path = Path(candidate["test_parquet"]).expanduser().resolve()
            row_index = int(candidate["row_index"])
            expected_qid = str(candidate["question_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(
                f"Malformed Obs3 candidate at {candidates_path}:{line_number}"
            ) from exc
        if parquet_path not in table_cache:
            if not parquet_path.is_file():
                raise AuditError(f"Obs3 parquet is missing: {parquet_path}")
            table_cache[parquet_path] = pq.read_table(parquet_path)
        table = table_cache[parquet_path]
        if not 0 <= row_index < table.num_rows:
            raise AuditError(
                f"Obs3 row_index={row_index} is outside {parquet_path} "
                f"({table.num_rows} rows)"
            )
        row = table.slice(row_index, 1).to_pylist()[0]
        actual_qid = _row_qid(row, row_index)
        if actual_qid != expected_qid:
            raise AuditError(
                f"Obs3 question ID drift at line {line_number}: "
                f"manifest={expected_qid} parquet={actual_qid}"
            )
        rows.append(row)
    return rows


def resolve_image_path(value: Any) -> Path:
    if isinstance(value, Mapping):
        value = value.get("image") or value.get("path")
    if not value:
        raise AuditError("Image entry has no image/path value")
    text = os.fspath(value)
    if text.startswith("file://"):
        text = text[7:]
    path = Path(text).expanduser()
    if not path.is_file():
        raise AuditError(f"Image is missing: {path}")
    return path


def image_patch_count(image_path: Path, image_processor: Any) -> int:
    with Image.open(image_path) as image:
        width, height = image.size
    size = image_processor.size
    patch_size = (int(size["height"]), int(size["width"]))
    columns, rows = get_optimal_tiled_canvas(
        (height, width),
        patch_size,
        int(image_processor.min_patches),
        int(image_processor.max_patches),
    )
    blocks = columns * rows
    # GotOcr2ImageProcessorFast.crop_image_to_patches uses a thumbnail by default.
    return blocks + (1 if blocks != 1 else 0)


def audit_scope(
    *,
    scope: str,
    rows: Iterable[dict[str, Any]],
    max_new_tokens: int,
    processor: Any,
    prompt_limit: int,
    model_len: int,
) -> tuple[dict[str, Any], list[str]]:
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    context_token_length = len(
        tokenizer.encode(processor.image_token, add_special_tokens=False)
    )
    start_token_length = len(
        tokenizer.encode(processor.start_image_token, add_special_tokens=False)
    )
    end_token_length = len(
        tokenizer.encode(processor.end_image_token, add_special_tokens=False)
    )
    if (context_token_length, start_token_length, end_token_length) != (1, 1, 1):
        raise AuditError(
            "InternVL image special tokens are no longer single tokens: "
            f"context/start/end="
            f"{(context_token_length, start_token_length, end_token_length)}"
        )

    metrics: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        messages = _build_messages(row)
        raw_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        image_entries = list(row.get("images") or [])
        video_entries = list(row.get("videos") or [])
        if video_entries:
            raise AuditError(
                f"{scope}[{ordinal}] contains video; this image audit cannot size it"
            )
        placeholder_count = raw_prompt.count("<image>")
        if placeholder_count != len(image_entries):
            raise AuditError(
                f"{scope}[{ordinal}] image placeholder mismatch: "
                f"prompt={placeholder_count} rows={len(image_entries)}"
            )
        if not image_entries:
            raise AuditError(f"{scope}[{ordinal}] contains no image")

        processor_prompt = raw_prompt.replace("<image>", IMAGE_CONTEXT_TOKEN)
        baseline_length = len(
            tokenizer.encode(processor_prompt, add_special_tokens=False)
        )
        total_patches = sum(
            image_patch_count(resolve_image_path(value), image_processor)
            for value in image_entries
        )
        prompt_tokens = (
            baseline_length
            - context_token_length * len(image_entries)
            + (start_token_length + end_token_length) * len(image_entries)
            + context_token_length * int(processor.image_seq_length) * total_patches
        )
        metrics.append(
            {
                "prompt_tokens": prompt_tokens,
                "image_patches": total_patches,
                "question_id": _row_qid(row, ordinal),
                "row": row,
            }
        )

    if not metrics:
        raise AuditError(f"{scope} contains zero rows")
    worst = max(metrics, key=lambda item: item["prompt_tokens"])

    # Materialize the worst row with the runtime processor. This catches drift
    # in token replacement, thumbnail policy, and patch-grid calculation.
    runtime_batch, _ = _prepare_inputs(processor, worst["row"])
    runtime_prompt_tokens = int(runtime_batch["input_ids"].shape[-1])
    runtime_image_patches = int(runtime_batch["pixel_values"].shape[0])
    formula_match = runtime_prompt_tokens == worst["prompt_tokens"]
    patch_match = runtime_image_patches == worst["image_patches"]
    over_prompt = sum(item["prompt_tokens"] > prompt_limit for item in metrics)
    over_context = sum(
        item["prompt_tokens"] + max_new_tokens > model_len for item in metrics
    )

    failures: list[str] = []
    if over_prompt:
        failures.append(f"{scope}: {over_prompt} prompts exceed {prompt_limit}")
    if over_context:
        failures.append(
            f"{scope}: {over_context} prompts plus response exceed {model_len}"
        )
    if not formula_match:
        failures.append(
            f"{scope}: formula={worst['prompt_tokens']} "
            f"runtime={runtime_prompt_tokens}"
        )
    if not patch_match:
        failures.append(
            f"{scope}: formula_patches={worst['image_patches']} "
            f"runtime_patches={runtime_image_patches}"
        )

    summary = {
        "scope": scope,
        "rows": len(metrics),
        "max_prompt_tokens": int(worst["prompt_tokens"]),
        "max_new_tokens": max_new_tokens,
        "max_total_tokens": int(worst["prompt_tokens"] + max_new_tokens),
        "prompt_limit": prompt_limit,
        "model_len": model_len,
        "max_image_patches": max(item["image_patches"] for item in metrics),
        "over_prompt_limit": over_prompt,
        "over_context": over_context,
        "worst_question_id": worst["question_id"],
        "runtime_crosscheck_prompt_tokens": runtime_prompt_tokens,
        "runtime_crosscheck_image_patches": runtime_image_patches,
        "formula_runtime_match": formula_match,
        "patch_runtime_match": patch_match,
        "status": "PASS" if not failures else "FAIL",
    }
    return summary, failures


def write_output(path: Path, payload: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def run_audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    model = args.model.expanduser().resolve()
    if not model.is_dir():
        raise AuditError(f"Model directory is missing: {model}")

    processor = hf_processor(str(model), trust_remote_code=True)
    if processor is None:
        raise AuditError(f"hf_processor returned None for {model}")
    if processor.image_token != IMAGE_CONTEXT_TOKEN:
        raise AuditError(
            f"InternVL image token drift: "
            f"{processor.image_token!r} != {IMAGE_CONTEXT_TOKEN!r}"
        )
    image_processor_name = type(processor.image_processor).__name__
    if image_processor_name != "GotOcr2ImageProcessorFast":
        raise AuditError(
            "Unsupported thumbnail policy for image processor "
            f"{image_processor_name}; update this audit before proceeding"
        )

    config = AutoConfig.from_pretrained(
        str(model), trust_remote_code=True, local_files_only=True
    )
    native_model_len = int(config.llm_config.max_position_embeddings)
    failures: list[str] = []
    if args.model_len > native_model_len:
        failures.append(
            f"configured model_len={args.model_len} exceeds "
            f"native_model_len={native_model_len}"
        )

    scopes: dict[str, dict[str, Any]] = {}
    if args.obs1_dir is not None:
        obs1_dir = args.obs1_dir.expanduser().resolve()
        for split in ("train", "test"):
            parquet_path = obs1_dir / f"{split}.parquet"
            if not parquet_path.is_file():
                raise AuditError(f"Obs1 parquet is missing: {parquet_path}")
            scope = f"obs1_{split}"
            summary, scope_failures = audit_scope(
                scope=scope,
                rows=pq.read_table(parquet_path).to_pylist(),
                max_new_tokens=args.obs1_max_new,
                processor=processor,
                prompt_limit=args.prompt_limit,
                model_len=args.model_len,
            )
            scopes[scope] = summary
            failures.extend(scope_failures)
            print(
                "PROMPT_TOKEN_AUDIT " + json.dumps(summary, sort_keys=True),
                flush=True,
            )

    if args.obs3_candidates is not None:
        candidates_path = args.obs3_candidates.expanduser().resolve()
        if not candidates_path.is_file():
            raise AuditError(f"Obs3 candidate JSONL is missing: {candidates_path}")
        summary, scope_failures = audit_scope(
            scope="obs3_candidates",
            rows=load_obs3_rows(candidates_path),
            max_new_tokens=args.obs3_max_new,
            processor=processor,
            prompt_limit=args.prompt_limit,
            model_len=args.model_len,
        )
        scopes["obs3_candidates"] = summary
        failures.extend(scope_failures)
        print(
            "PROMPT_TOKEN_AUDIT " + json.dumps(summary, sort_keys=True),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "model": str(model),
        "processor": type(processor).__name__,
        "image_processor": image_processor_name,
        "image_seq_length": int(processor.image_seq_length),
        "min_patches": int(processor.image_processor.min_patches),
        "max_patches": int(processor.image_processor.max_patches),
        "prompt_limit": args.prompt_limit,
        "model_len": args.model_len,
        "native_model_len": native_model_len,
        "scopes": scopes,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    return payload, 0 if not failures else 1


def main() -> int:
    args = parse_args()
    try:
        payload, exit_code = run_audit(args)
    except AuditError as exc:
        payload = {
            "schema_version": 1,
            "status": "ERROR",
            "failures": [str(exc)],
            "scopes": {},
        }
        if args.output_json is not None:
            output_path = write_output(args.output_json, payload)
            print(f"PROMPT_TOKEN_AUDIT_OUTPUT {output_path}", flush=True)
        print(f"PROMPT_TOKEN_AUDIT_ERROR {exc}", file=sys.stderr, flush=True)
        return 2

    if args.output_json is not None:
        output_path = write_output(args.output_json, payload)
        print(f"PROMPT_TOKEN_AUDIT_OUTPUT {output_path}", flush=True)
    print(f"PROMPT_TOKEN_AUDIT_OVERALL {payload['status']}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
