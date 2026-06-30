import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional


def as_plain(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def parse_maybe_json(value: Any) -> Any:
    value = as_plain(value)
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def read_rows(path: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise SystemExit("pandas/pyarrow are required to read parquet files for grounding scripts.") from exc
        df = pd.read_parquet(path)
        if limit is not None:
            df = df.head(limit)
        return [dict(row) for row in df.to_dict(orient="records")]
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        if limit is not None:
            payload = payload[:limit]
        return [dict(row) for row in payload]
    raise ValueError(f"Unsupported table format {path}; expected .parquet or .json")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = parse_maybe_json(row.get("extra_info", {}))
    return extra_info if isinstance(extra_info, dict) else {}


def row_index(row: dict[str, Any], ordinal: int) -> Any:
    extra_info = row_extra_info(row)
    return as_plain(extra_info.get("index", row.get("index", row.get("id", ordinal))))


def row_data_source(row: dict[str, Any]) -> str:
    return str(as_plain(row.get("data_source", row.get("source", "unknown"))))


def grounding_key(row: dict[str, Any], ordinal: int) -> str:
    return f"{row_data_source(row)}::{row_index(row, ordinal)}"


def prompt_to_text(prompt: Any) -> str:
    prompt = parse_maybe_json(prompt)
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        pieces = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            pieces.append(str(item.get("text", "")))
                        elif isinstance(item, str):
                            pieces.append(item)
                else:
                    pieces.append(str(content))
            else:
                pieces.append(str(message))
        return "\n".join(piece for piece in pieces if piece)
    return str(prompt)


def image_specs(row: dict[str, Any], image_key: str = "images") -> list[Any]:
    images = parse_maybe_json(row.get(image_key))
    if images is None and "image" in row:
        images = row.get("image")
    if images is None and "image_path" in row:
        images = [{"image": row.get("image_path")}]
    if images is None:
        return []
    if isinstance(images, (str, dict)):
        images = [images]
    return list(images)


def image_path_from_spec(spec: Any) -> Optional[str]:
    spec = parse_maybe_json(spec)
    if isinstance(spec, dict):
        spec = spec.get("image", spec.get("path", spec.get("image_path")))
    if isinstance(spec, str) and not spec.startswith("data:image"):
        return spec[7:] if spec.startswith("file://") else spec
    return None


def load_image_from_spec(spec: Any):
    from PIL import Image

    spec = parse_maybe_json(spec)
    if isinstance(spec, dict):
        spec = spec.get("image", spec.get("path", spec.get("image_path")))
    if hasattr(spec, "convert"):
        return spec.convert("RGB")
    if not isinstance(spec, str):
        raise ValueError(f"Unsupported image spec: {type(spec)!r}")
    if spec.startswith("file://"):
        return Image.open(spec[7:]).convert("RGB")
    if spec.startswith("data:image") and "base64," in spec:
        data = base64.b64decode(spec.split("base64,", 1)[1])
        return Image.open(BytesIO(data)).convert("RGB")
    if spec.startswith("http://") or spec.startswith("https://"):
        try:
            import requests
        except ImportError as exc:
            raise SystemExit("requests is required to load HTTP images for grounding scripts.") from exc
        response = requests.get(spec, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    if not os.path.exists(spec):
        raise FileNotFoundError(f"Image path not found: {spec}")
    return Image.open(spec).convert("RGB")


def _extract_json_payload(text: str) -> Any:
    text = str(text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _clean_descriptor_phrase(phrase: Any) -> Optional[str]:
    phrase = str(phrase or "").strip().strip('"').strip("'")
    if not phrase:
        return None
    phrase = re.sub(r"^[-*\d.\s]+", "", phrase).strip()
    phrase = re.sub(r"^(?:descriptor|descriptors|evidence|evidence descriptors?)\s*[:：-]\s*", "", phrase, flags=re.I)
    phrase = re.sub(r"^(?:visual evidence needed|visual evidence)\s*[:：-]\s*", "", phrase, flags=re.I)
    phrase = re.sub(r"^[-*\d.\s]+", "", phrase).strip()
    phrase = re.sub(r"^WHOLE[_\s-]*IMAGE\b[:：,.;\s-]*", "", phrase, flags=re.I).strip() or "WHOLE_IMAGE"
    phrase = re.sub(r"^(?:the )?(?:visual evidence needed to answer the question is)\s*", "", phrase, flags=re.I)
    phrase = re.sub(r"^(?:the )?image contains\s*", "", phrase, flags=re.I)
    phrase = re.sub(r"^(?:the )?(?:correct )?answer(?: is)?\s*[:：-]?\s*", "", phrase, flags=re.I)
    phrase = re.sub(r"^[A-F]\s*[\).:：-]\s*", "", phrase).strip()
    phrase = re.sub(r"\b(?:the )?(?:correct )?answer(?: is)?\s*[:：-]?\s*[A-F]\b\.?", "", phrase, flags=re.I).strip()
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:")
    if not phrase:
        return None
    upper = phrase.upper().replace(" ", "_")
    if upper == "WHOLE_IMAGE" or re.fullmatch(r"[A-F]WHOLE[_-]?IMAGE", upper):
        return "WHOLE_IMAGE"
    if re.fullmatch(r"(?:yes|no|true|false)", phrase.strip(), flags=re.I):
        return None
    if re.match(r"^(?:which|that|this|these)\b", phrase.strip(), flags=re.I):
        return None
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.I).strip()
    if re.fullmatch(r"[A-F]", phrase.strip(), flags=re.I):
        return None
    if re.fullmatch(r"[A-F]\s*(?:,?\s*[A-F]){1,5}", phrase.strip(), flags=re.I):
        return None
    if re.fullmatch(r"(?:option|choice)\s+[A-F]", phrase.strip(), flags=re.I):
        return None
    return phrase


def descriptor_mode(descriptor: str) -> str:
    payload = _extract_json_payload(str(descriptor or ""))
    if isinstance(payload, dict):
        mode = str(payload.get("mode") or "").strip().lower()
        if mode in {"whole_image", "whole-image", "global"}:
            return "whole_image"
        if mode in {"localized", "local", "region", "regions"}:
            return "localized"
    text = str(descriptor or "").strip()
    if text.upper() == "WHOLE_IMAGE":
        return "whole_image"
    return ""


def descriptor_phrases(descriptor: str) -> list[str]:
    descriptor = str(descriptor or "").strip()
    if not descriptor:
        return []
    payload = _extract_json_payload(descriptor)
    raw_phrases: list[Any] = []
    if isinstance(payload, dict):
        values = payload.get("descriptors", payload.get("evidence", payload.get("phrases", [])))
        if isinstance(values, str):
            raw_phrases = re.split(r"[,;\n]+|\s+-\s+", values)
        elif isinstance(values, list):
            raw_phrases = values
    elif isinstance(payload, list):
        raw_phrases = payload
    else:
        if descriptor.upper() == "WHOLE_IMAGE":
            return ["WHOLE_IMAGE"]
        cleaned_text = re.sub(r"```(?:json)?|```", "", descriptor, flags=re.I).strip()
        cleaned_text = re.sub(r"(?i)\b(?:evidence descriptors?|descriptors?)\b\s*[:：-]?", "\n", cleaned_text)
        raw_phrases = re.split(r"[,;\n]+|\s+-\s+", cleaned_text)

    phrases = []
    seen = set()
    for raw_phrase in raw_phrases:
        phrase = _clean_descriptor_phrase(raw_phrase)
        if phrase is None:
            continue
        key = phrase.lower()
        if key not in seen:
            phrases.append(phrase)
            seen.add(key)
    local_phrases = [phrase for phrase in phrases if phrase.upper() != "WHOLE_IMAGE"]
    return local_phrases if local_phrases else phrases


def is_whole_image_descriptor(descriptor: str) -> bool:
    mode = descriptor_mode(descriptor)
    phrases = descriptor_phrases(descriptor)
    local_phrases = [phrase for phrase in phrases if phrase.upper() != "WHOLE_IMAGE"]
    if local_phrases:
        return False
    if mode == "whole_image":
        return True
    return bool(phrases) and all(phrase.upper() == "WHOLE_IMAGE" for phrase in phrases)


def ocr_likely_needed(phrases: list[str]) -> bool:
    for phrase in phrases:
        compact = phrase.strip()
        lower = compact.lower()
        if re.fullmatch(r"[a-zA-Z0-9]", compact):
            return True
        if re.search(r"\b(label|letter|number|digit|text|word|value|angle|axis)\b", lower):
            return True
        if re.search(r"\b[A-Z]\b", compact):
            return True
    return False


def normalize_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box]
    x0, x1 = sorted((max(0.0, min(float(width), x0)), max(0.0, min(float(width), x1))))
    y0, y1 = sorted((max(0.0, min(float(height), y0)), max(0.0, min(float(height), y1))))
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0 / width, y0 / height, x1 / width, y1 / height]
