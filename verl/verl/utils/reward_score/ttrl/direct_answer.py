import json
import re
from collections import Counter


UNKNOWN_LABEL = "unknown"

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^0-9a-zA-Z]+")
_SPACE_RE = re.compile(r"\s+")


def _strip_special_tokens(text):
    text = str(text or "")
    text = re.sub(r"<\|[^>]*\|>", " ", text)
    text = re.sub(r"</?s>|<pad>|<image>|<IMG_CONTEXT>", " ", text, flags=re.IGNORECASE)
    return text.strip()


def extract_direct_answer(text):
    """Extract a short direct-answer string from a model response."""
    text = _strip_special_tokens(text)
    if not text:
        return UNKNOWN_LABEL

    patterns = [
        r"(?:^|[\r\n])\s*(?:final\s+answer|answer|short\s+answer)\s*[:：]\s*(.+)",
        r"\b(?:the\s+)?(?:final\s+)?answer\s+is\s*[:：]?\s*(.+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            candidate = matches[-1].group(1)
            candidate = re.split(r"[\r\n]", candidate.strip(), maxsplit=1)[0]
            candidate = re.split(r"(?<=[.!?])\s+", candidate.strip(), maxsplit=1)[0]
            return normalize_direct_answer(candidate)

    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    if lines:
        candidate = lines[-1] if len(lines[-1].split()) <= 12 else lines[0]
    else:
        candidate = text
    candidate = re.split(r"(?<=[.!?])\s+", candidate.strip(), maxsplit=1)[0]
    return normalize_direct_answer(candidate)


def normalize_direct_answer(answer):
    """VQA-style lightweight normalization for short free-form answers."""
    answer = _strip_special_tokens(answer)
    if not answer:
        return UNKNOWN_LABEL
    answer = answer.lower()
    answer = answer.replace("’", "'")
    answer = answer.replace("`", "'")
    answer = answer.replace(",", "")
    answer = _PUNCT_RE.sub(" ", answer)
    tokens = [token for token in _SPACE_RE.split(answer.strip()) if token and token not in _ARTICLES]
    if not tokens:
        return UNKNOWN_LABEL
    return " ".join(tokens)


def coerce_answer_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, dict):
        for key in ("direct_answers", "answers", "official_answers", "answer"):
            if key in value:
                return coerce_answer_list(value[key])
        return []
    text = str(value).strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            return coerce_answer_list(json.loads(text))
        except Exception:
            pass
    return [text]


def direct_answer_exact_score(prediction, references):
    pred = normalize_direct_answer(prediction)
    if pred == UNKNOWN_LABEL:
        return 0.0
    refs = [normalize_direct_answer(ref) for ref in coerce_answer_list(references)]
    return float(any(ref != UNKNOWN_LABEL and pred == ref for ref in refs))


def aokvqa_direct_answer_score(prediction, direct_answers):
    pred = normalize_direct_answer(prediction)
    if pred == UNKNOWN_LABEL:
        return 0.0
    refs = [normalize_direct_answer(ref) for ref in coerce_answer_list(direct_answers)]
    matches = sum(1 for ref in refs if ref != UNKNOWN_LABEL and ref == pred)
    return min(1.0, matches / 3.0)


def majority_answer(answers):
    normalized = [normalize_direct_answer(answer) for answer in answers]
    normalized = [answer for answer in normalized if answer != UNKNOWN_LABEL]
    if not normalized:
        return UNKNOWN_LABEL
    return Counter(normalized).most_common(1)[0][0]
