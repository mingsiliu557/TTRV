#!/usr/bin/env python
"""Map PAPO evidence descriptors to normalized GroundingDINO/OCR boxes."""

import argparse
from pathlib import Path

try:
    from ._common import (
        descriptor_mode,
        descriptor_phrases,
        grounding_key,
        image_specs,
        is_whole_image_descriptor,
        load_image_from_spec,
        normalize_xyxy,
        ocr_likely_needed,
        read_jsonl,
        read_rows,
        row_data_source,
        row_index,
        write_jsonl,
    )
except ImportError:
    from _common import (  # type: ignore
        descriptor_mode,
        descriptor_phrases,
        grounding_key,
        image_specs,
        is_whole_image_descriptor,
        load_image_from_spec,
        normalize_xyxy,
        ocr_likely_needed,
        read_jsonl,
        read_rows,
        row_data_source,
        row_index,
        write_jsonl,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors-jsonl", required=True)
    parser.add_argument("--input-parquet", required=True, help="Same train parquet used for descriptors")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files")
    parser.add_argument("--require-ocr", action="store_true", help="Fail if pytesseract is unavailable")
    return parser.parse_args()


def load_grounding_model(args):
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Grounding requires torch and transformers with GroundingDINO support. Install dependencies and cache the model."
        ) from exc
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    try:
        processor = AutoProcessor.from_pretrained(
            args.grounding_model,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.grounding_model,
            local_files_only=args.local_files_only,
        ).eval()
        model.to(device)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load GroundingDINO model from {args.grounding_model!r}. "
            "Install dependencies and pre-cache the model, or pass a local --grounding-model. "
            f"Original error: {exc}"
        ) from exc
    return model, processor, device


def maybe_load_ocr(require_ocr: bool):
    try:
        import pytesseract
        from pytesseract import Output
    except Exception as exc:
        if require_ocr:
            raise SystemExit(
                "OCR was requested but pytesseract is unavailable. Install pytesseract and the tesseract binary."
            ) from exc
        return None, None
    return pytesseract, Output


def dino_boxes(image, phrases, model, processor, device, args):
    import torch

    width, height = image.size
    text = ". ".join(phrase for phrase in phrases if phrase.upper() != "WHOLE_IMAGE")
    if not text:
        return [], [], []
    if not text.endswith("."):
        text = text + "."
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [(height, width)]
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        )[0]
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    labels = results.get("labels", [])
    boxes = boxes.detach().cpu().tolist() if hasattr(boxes, "detach") else list(boxes)
    scores = scores.detach().cpu().tolist() if hasattr(scores, "detach") else list(scores)
    norm_boxes = []
    norm_scores = []
    norm_labels = []
    for box, score, label in zip(boxes, scores, labels):
        norm_box = normalize_xyxy(box, width=width, height=height)
        if norm_box:
            norm_boxes.append(norm_box)
            norm_scores.append(float(score))
            norm_labels.append(str(label))
    return norm_boxes, norm_scores, norm_labels


def ocr_boxes(image, phrases, pytesseract, Output):
    if pytesseract is None:
        return [], [], []
    width, height = image.size
    wanted = {phrase.strip().lower() for phrase in phrases if phrase.strip()}
    wanted_tokens = set()
    for phrase in wanted:
        wanted_tokens.update(token for token in phrase.replace(":", " ").split() if token)
    if not wanted_tokens:
        return [], [], []
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    boxes = []
    scores = []
    labels = []
    for i, text in enumerate(data.get("text", [])):
        token = str(text).strip()
        if not token or token.lower() not in wanted_tokens:
            continue
        try:
            conf = float(data.get("conf", [0])[i]) / 100.0
        except Exception:
            conf = 0.0
        left = float(data.get("left", [0])[i])
        top = float(data.get("top", [0])[i])
        w = float(data.get("width", [0])[i])
        h = float(data.get("height", [0])[i])
        norm_box = normalize_xyxy([left, top, left + w, top + h], width=width, height=height)
        if norm_box:
            boxes.append(norm_box)
            scores.append(max(0.0, conf))
            labels.append(f"ocr:{token}")
    return boxes, scores, labels


def build_image_map(input_parquet: str, image_key: str, limit):
    rows = read_rows(input_parquet, limit=limit)
    image_map = {}
    meta_map = {}
    for ordinal, row in enumerate(rows):
        key = grounding_key(row, ordinal)
        specs = image_specs(row, image_key=image_key)
        image_map[key] = specs[0] if specs else None
        meta_map[key] = {"data_source": row_data_source(row), "index": row_index(row, ordinal)}
    return image_map, meta_map


def main():
    args = parse_args()
    records = read_jsonl(args.descriptors_jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    image_map, meta_map = build_image_map(args.input_parquet, args.image_key, args.limit)
    model, processor, device = load_grounding_model(args)
    pytesseract, Output = maybe_load_ocr(args.require_ocr)
    outputs = []
    for record in records:
        key = str(record.get("grounding_key") or "")
        descriptor = str(record.get("descriptor") or "")
        phrases = descriptor_phrases(descriptor)
        meta = meta_map.get(key, {})
        out = {
            "data_source": record.get("data_source", meta.get("data_source", "unknown")),
            "index": record.get("index", meta.get("index")),
            "grounding_key": key,
            "descriptor": descriptor,
            "descriptor_mode": descriptor_mode(descriptor),
            "descriptor_phrases": phrases,
            "boxes_norm": [],
            "confidence": 0.0,
            "fallback": False,
            "fallback_reason": "",
        }
        image_spec = image_map.get(key)
        if is_whole_image_descriptor(descriptor):
            out.update({"boxes_norm": "WHOLE_IMAGE", "fallback": True, "fallback_reason": "whole_image"})
            outputs.append(out)
            continue
        if image_spec is None:
            out.update({"fallback": True, "fallback_reason": "missing_image"})
            outputs.append(out)
            continue
        if not phrases:
            out.update({"fallback": True, "fallback_reason": "empty_descriptor"})
            outputs.append(out)
            continue
        image = load_image_from_spec(image_spec)
        ocr_needed = ocr_likely_needed(phrases)
        dino_norm_boxes, dino_scores, dino_labels = dino_boxes(image, phrases, model, processor, device, args)
        ocr_norm_boxes, ocr_scores, ocr_labels = [], [], []
        ocr_available = pytesseract is not None
        if ocr_needed and ocr_available:
            ocr_norm_boxes, ocr_scores, ocr_labels = ocr_boxes(image, phrases, pytesseract, Output)
        boxes = dino_norm_boxes + ocr_norm_boxes
        scores = dino_scores + ocr_scores
        labels = dino_labels + ocr_labels
        out.update(
            {
                "ocr_needed": bool(ocr_needed),
                "ocr_available": bool(ocr_available),
                "dino_box_count": len(dino_norm_boxes),
                "ocr_box_count": len(ocr_norm_boxes),
            }
        )
        if not boxes:
            reason = "no_boxes"
            if ocr_needed and not ocr_available:
                reason = "no_boxes_ocr_unavailable"
            out.update({"fallback": True, "fallback_reason": reason})
        else:
            out.update(
                {
                    "boxes_norm": boxes,
                    "confidence": max(scores) if scores else 0.0,
                    "box_confidences": scores,
                    "box_labels": labels,
                    "fallback": False,
                    "fallback_reason": "",
                }
            )
        outputs.append(out)
    write_jsonl(args.output_jsonl, outputs)
    fallback_count = sum(1 for row in outputs if row.get("fallback"))
    print(f"wrote {len(outputs)} grounding records to {Path(args.output_jsonl)}; fallbacks={fallback_count}")


if __name__ == "__main__":
    main()
