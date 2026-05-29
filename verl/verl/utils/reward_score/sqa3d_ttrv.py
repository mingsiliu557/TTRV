import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable


ARTICLES_AND_COPULAS = re.compile(r"\b(a|an|the|is|are)\b")
SQA3D_ANSWER_TYPES = ("what", "is", "how", "can", "which", "other")
OFFICIAL_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
    "23": "twenty-three",
}


def normalize_sqa3d_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = ARTICLES_AND_COPULAS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_sqa3d_answer_official(text: str | None) -> str:
    """SQA3D LLM eval answer cleanup, kept local to avoid importing that script."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[ ]+$", "", text)
    text = re.sub(r"^[ ]+", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\.[ ]{2,}", ". ", text)
    text = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", text)
    text = re.sub("ç", "c", text)
    text = re.sub("’", "'", text)
    replacements = {
        r"\bletf\b": "left",
        r"\blet\b": "left",
        r"\btehre\b": "there",
        r"\brigth\b": "right",
        r"\brght\b": "right",
        r"\bbehine\b": "behind",
        r"\btv\b": "TV",
        r"\bchai\b": "chair",
        r"\bwasing\b": "washing",
        r"\bwaslked\b": "walked",
        r"\boclock\b": "o'clock",
        r"\bo'[ ]+clock\b": "o'clock",
        r"\bbackwards\b": "backward",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    for digit, word in OFFICIAL_DIGIT_WORDS.items():
        text = re.sub(rf"\b{re.escape(digit)}\b", word, text)
    text = re.sub(r"\bnone\b", "zero", text)
    text = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", text)
    text = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", text)
    return text


@lru_cache(maxsize=8)
def _load_answer_dict(answer_dict_path: str | None):
    if not answer_dict_path:
        return None
    answer_dict_path = os.path.expanduser(answer_dict_path)
    with open(answer_dict_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entries = raw.items() if isinstance(raw, dict) else []
    if isinstance(raw, list):
        entries = []
        for item in raw:
            if isinstance(item, dict):
                entries.extend(item.items())

    answer_dict = {}
    for answer, idx in entries:
        try:
            answer_dict[normalize_sqa3d_answer(clean_sqa3d_answer_official(answer))] = int(idx)
        except (TypeError, ValueError):
            continue
    return answer_dict


@lru_cache(maxsize=8)
def _load_answer_prior(answer_dict_path: str | None):
    """Answer frequency prior from official SQA3D annotations.

    This is only used to choose between multiple official answer_dict phrases
    found in one generated sentence; exact answer_dict matches always win.
    Prefer train/val so evaluation labels are not used as a parsing prior.
    """
    answer_dict = _load_answer_dict(answer_dict_path)
    if not answer_dict_path or not answer_dict:
        return Counter()

    base_dir = Path(os.path.expanduser(answer_dict_path)).resolve().parent
    split_paths = [
        base_dir / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
        for split in ("train", "val")
    ]
    existing_paths = [path for path in split_paths if path.exists()]
    if not existing_paths:
        existing_paths = [
            path
            for path in [
                base_dir / "v1_balanced_sqa_annotations_test_scannetv2.json",
            ]
            if path.exists()
        ]

    prior = Counter()
    for path in existing_paths:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for annotation in payload.get("annotations", []):
            for answer in annotation.get("answers", []):
                norm = normalize_sqa3d_answer(clean_sqa3d_answer_official(answer.get("answer", "")))
                if norm in answer_dict:
                    prior[norm] += 1
    return prior


def _infer_question_type(question: str | None) -> str:
    first_token = str(question or "").strip().split()
    if not first_token:
        return "other"
    first = first_token[0].lower().strip("?:,.!")
    if first in SQA3D_ANSWER_TYPES:
        return first
    return "other"


def _annotation_paths(answer_dict_path: str | None) -> list[Path]:
    if not answer_dict_path:
        return []
    base_dir = Path(os.path.expanduser(answer_dict_path)).resolve().parent
    paths = [
        base_dir / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
        for split in ("train", "val")
    ]
    existing = [path for path in paths if path.exists()]
    if existing:
        return existing
    return [path for path in [base_dir / "v1_balanced_sqa_annotations_test_scannetv2.json"] if path.exists()]


def _question_paths(answer_dict_path: str | None) -> list[Path]:
    if not answer_dict_path:
        return []
    base_dir = Path(os.path.expanduser(answer_dict_path)).resolve().parent
    paths = [
        base_dir / f"v1_balanced_questions_{split}_scannetv2.json"
        for split in ("train", "val")
    ]
    existing = [path for path in paths if path.exists()]
    if existing:
        return existing
    return [path for path in [base_dir / "v1_balanced_questions_test_scannetv2.json"] if path.exists()]


@lru_cache(maxsize=8)
def _load_answer_type_prior(answer_dict_path: str | None):
    answer_dict = _load_answer_dict(answer_dict_path)
    if not answer_dict_path or not answer_dict:
        return {}

    qid_to_type = {}
    for path in _question_paths(answer_dict_path):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for question in payload.get("questions", []):
            qid_to_type[question.get("question_id")] = _infer_question_type(question.get("question", ""))

    typed_prior = defaultdict(Counter)
    for path in _annotation_paths(answer_dict_path):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for annotation in payload.get("annotations", []):
            answer_type = qid_to_type.get(annotation.get("question_id"), "other")
            for answer in annotation.get("answers", []):
                norm = normalize_sqa3d_answer(clean_sqa3d_answer_official(answer.get("answer", "")))
                if norm in answer_dict:
                    typed_prior[answer_type][norm] += 1
    return {key: Counter(value) for key, value in typed_prior.items()}


@lru_cache(maxsize=8)
def _answer_candidate_index(answer_keys: tuple[str, ...]):
    candidates = {candidate for candidate in answer_keys if candidate}
    metadata = {candidate: (len(candidate.split()), len(candidate)) for candidate in candidates}
    max_tokens = max((token_count for token_count, _ in metadata.values()), default=0)
    return frozenset(candidates), metadata, max_tokens


def _matched_answer_candidates(norm: str, answer_dict: dict[str, int]) -> dict[str, int]:
    """Return answer_dict candidates found as complete normalized token spans."""
    candidates, _metadata, max_tokens = _answer_candidate_index(tuple(answer_dict.keys()))
    if not candidates or max_tokens <= 0:
        return {}

    tokens = norm.split()
    if not tokens:
        return {}

    starts = []
    offset = 0
    for token in tokens:
        starts.append(offset)
        offset += len(token) + 1

    matches = {}
    max_tokens = min(max_tokens, len(tokens))
    for start_idx in range(len(tokens)):
        phrase = ""
        for end_idx in range(start_idx, min(len(tokens), start_idx + max_tokens)):
            phrase = tokens[end_idx] if end_idx == start_idx else f"{phrase} {tokens[end_idx]}"
            if phrase in candidates and phrase not in matches:
                matches[phrase] = starts[start_idx]
    return matches


def _answer_region(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    lower = text.lower()
    cut_points = []
    for marker in ("\n", "###", "##", " assistant:", " human:", " user:", " system:"):
        idx = lower.find(marker)
        if idx > 0:
            cut_points.append(idx)
    sentence = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if sentence:
        cut_points.append(sentence.end(1))
    if cut_points:
        text = text[: min(cut_points)]
    return text.strip()


def canonicalize_sqa3d_freeform(text: str | None) -> str:
    """Short text used for TTRV vote frequency, based on SQA3D LLM cleanup."""
    cleaned = clean_sqa3d_answer_official(_answer_region(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "unknown"


def evaluate_sqa3d_official_freeform(prediction: str, ground_truth: str | None) -> bool:
    """Official SQA3D LLM free-form metric from SQA3D/LLM/eval_sqa3d_llm.py.

    The official script compares a cleaned prediction against the primary GT
    answer and accepts exact match, substring match, whitespace-insensitive
    substring match, or any token overlap.
    """
    pred = clean_sqa3d_answer_official(prediction)
    answer = clean_sqa3d_answer_official(ground_truth)
    if not pred or not answer:
        return False
    if pred == answer:
        return True
    if pred in answer:
        return True
    if "".join(pred.split()) in "".join(answer.split()):
        return True
    return len(set(pred.split()).intersection(answer.split())) > 0


def evaluate_sqa3d_strict_freeform(prediction: str, ground_truth: str | None) -> bool:
    pred = canonicalize_sqa3d_freeform(prediction)
    answer = clean_sqa3d_answer_official(ground_truth)
    answer = re.sub(r"\s+", " ", answer).strip()
    return bool(pred and answer and pred == answer)


def map_to_sqa3d_answer(
    text: str | None,
    answer_dict: dict[str, int] | None = None,
    answer_prior: Counter | None = None,
    answer_type_prior: dict[str, Counter] | None = None,
    preferred_family: str | None = None,
    question_type: str | None = None,
) -> str:
    del preferred_family
    norm = normalize_sqa3d_answer(clean_sqa3d_answer_official(_answer_region(text)))
    if not norm:
        return "unknown"
    if not answer_dict:
        return norm
    if norm in answer_dict:
        return norm

    matches = []
    candidate_starts = _matched_answer_candidates(norm, answer_dict)
    if candidate_starts:
        _candidates, metadata, _max_tokens = _answer_candidate_index(tuple(answer_dict.keys()))
        normalized_type = normalize_sqa3d_answer_type(question_type)
        typed_counts = (answer_type_prior or {}).get(normalized_type, Counter()) if answer_type_prior else Counter()
        for candidate, start in candidate_starts.items():
            typed_prior = int(typed_counts.get(candidate, 0))
            token_count, char_count = metadata[candidate]
            prior = int(answer_prior.get(candidate, 0)) if answer_prior else 0
            matches.append((typed_prior, prior, token_count, char_count, -start, candidate))
    if matches:
        if " or " in f" {norm} " and len({match[-1] for match in matches}) > 1:
            return "unknown"
        if answer_type_prior and any(match[0] > 0 for match in matches):
            matches = [match for match in matches if match[0] > 0]
        elif answer_prior and any(match[1] > 0 for match in matches):
            matches = [match for match in matches if match[1] > 0]
        return max(matches)[5]
    return "unknown"


def _entropy(mapped: Iterable[str]) -> float:
    mapped = list(mapped)
    if not mapped:
        return 0.0
    total = len(mapped)
    counts = Counter(mapped)
    return -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())


def _normalized_entropy(mapped: Iterable[str]) -> float:
    mapped = list(mapped)
    if len(mapped) <= 1:
        return 0.0
    entropy = _entropy(mapped)
    max_entropy = math.log(len(Counter(mapped)))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def evaluate_sqa3d_em(
    prediction: str,
    gt_answers: list[str],
    answer_dict: dict[str, int] | None = None,
    answer_prior: Counter | None = None,
    answer_type_prior: dict[str, Counter] | None = None,
    question_type: str | None = None,
) -> bool:
    mapped_pred = map_to_sqa3d_answer(
        prediction,
        answer_dict,
        answer_prior=answer_prior,
        answer_type_prior=answer_type_prior,
        question_type=question_type,
    )
    if mapped_pred == "unknown":
        return False
    mapped_gts = [
        map_to_sqa3d_answer(
            gt,
            answer_dict,
            answer_prior=answer_prior,
            answer_type_prior=answer_type_prior,
            question_type=question_type,
        )
        for gt in gt_answers
    ]
    if mapped_pred in mapped_gts:
        return True
    if answer_dict and mapped_pred in answer_dict:
        pred_class = answer_dict[mapped_pred]
        return any(mapped_gt in answer_dict and answer_dict[mapped_gt] == pred_class for mapped_gt in mapped_gts)
    pred_norm = normalize_sqa3d_answer(clean_sqa3d_answer_official(prediction))
    gt_norms = [normalize_sqa3d_answer(clean_sqa3d_answer_official(answer)) for answer in gt_answers]
    if mapped_gts and all(mapped_gt == "unknown" for mapped_gt in mapped_gts):
        return pred_norm in gt_norms
    return False


def evaluate_sqa3d_closed_set_em(
    prediction: str,
    gt_answers: list[str],
    answer_dict: dict[str, int] | None = None,
    answer_prior: Counter | None = None,
    answer_type_prior: dict[str, Counter] | None = None,
    question_type: str | None = None,
) -> bool:
    return evaluate_sqa3d_em(
        prediction,
        gt_answers,
        answer_dict=answer_dict,
        answer_prior=answer_prior,
        answer_type_prior=answer_type_prior,
        question_type=question_type,
    )


def normalize_sqa3d_answer_type(answer_type: str | None) -> str:
    norm = str(answer_type or "").strip().lower()
    if norm in SQA3D_ANSWER_TYPES:
        return norm
    return "other"


def _majority_metric_value(preds: list[str], values: list[float]) -> float:
    if not preds or not values:
        return 0.0
    winner = Counter(preds).most_common(1)[0][0]
    for pred, value in zip(preds, values):
        if pred == winner:
            return float(value)
    return 0.0


def summarize_sqa3d_validation(
    data_sources: list[str],
    group_keys: list[str],
    infos_dict: dict[str, list],
) -> dict[str, float]:
    """Build SQA3D-specific validation metrics.

    Returns overall and per-answer-type EM/unknown metrics for:
    - baseline: first rollout per sample
    - sample-level: best rollout EM within the sample
    - dataset-level: majority-vote prediction within the sample
    """
    required = {"acc", "pred", "answer_type"}
    if not required.issubset(infos_dict) or len(group_keys) != len(data_sources):
        return {}

    accs = infos_dict["acc"]
    preds = infos_dict["pred"]
    answer_types = infos_dict["answer_type"]
    unknowns = infos_dict.get("unknown")
    strict_accs = infos_dict.get("strict_acc")
    closed_accs = infos_dict.get("closed_acc")
    closed_unknowns = infos_dict.get("closed_unknown")
    blanks = infos_dict.get("blank")
    if not (len(accs) == len(preds) == len(answer_types) == len(group_keys) == len(data_sources)):
        return {}
    if unknowns is not None and len(unknowns) != len(accs):
        return {}
    if strict_accs is not None and len(strict_accs) != len(accs):
        return {}
    if closed_accs is not None and len(closed_accs) != len(accs):
        return {}
    if closed_unknowns is not None and len(closed_unknowns) != len(accs):
        return {}
    if blanks is not None and len(blanks) != len(accs):
        return {}

    grouped = defaultdict(list)
    for idx, data_source in enumerate(data_sources):
        if str(data_source) != "sqa3d":
            continue
        grouped[str(group_keys[idx])].append(
            {
                "acc": float(accs[idx]),
                "pred": str(preds[idx]),
                "answer_type": normalize_sqa3d_answer_type(answer_types[idx]),
                "unknown": float(unknowns[idx]) if unknowns is not None else float(str(preds[idx]) == "unknown"),
                "strict_acc": float(strict_accs[idx]) if strict_accs is not None else None,
                "closed_acc": float(closed_accs[idx]) if closed_accs is not None else None,
                "closed_unknown": float(closed_unknowns[idx]) if closed_unknowns is not None else None,
                "blank": float(blanks[idx]) if blanks is not None else 0.0,
            }
        )

    if not grouped:
        return {}

    aggregate = defaultdict(list)
    for records in grouped.values():
        record_answer_type = records[0]["answer_type"]
        record_accs = [record["acc"] for record in records]
        record_preds = [record["pred"] for record in records]
        record_unknowns = [record["unknown"] for record in records]
        record_blanks = [record["blank"] for record in records]
        optional_metrics = {}
        for source_name, output_name in (
            ("strict_acc", "strict_em"),
            ("closed_acc", "closed_em"),
            ("closed_unknown", "closed_unknown"),
        ):
            values = [record[source_name] for record in records]
            if all(value is not None for value in values):
                optional_metrics[output_name] = values
        views = {
            "baseline": {
                "em": record_accs[0],
                "unknown": record_unknowns[0],
                "blank": record_blanks[0],
            },
            "sample-level": {
                "em": max(record_accs),
                "unknown": float(all(value >= 0.5 for value in record_unknowns)),
                "blank": float(all(value >= 0.5 for value in record_blanks)),
            },
            "dataset-level": {
                "em": _majority_metric_value(record_preds, record_accs),
                "unknown": _majority_metric_value(record_preds, record_unknowns),
                "blank": _majority_metric_value(record_preds, record_blanks),
            },
        }
        for output_name, values in optional_metrics.items():
            views["baseline"][output_name] = values[0]
            views["sample-level"][output_name] = max(values)
            views["dataset-level"][output_name] = _majority_metric_value(record_preds, values)
        for view_name, metrics in views.items():
            for metric_name, metric_value in metrics.items():
                aggregate[(metric_name, "overall", view_name)].append(metric_value)
                aggregate[(metric_name, record_answer_type, view_name)].append(metric_value)

    summary = {}
    for (metric_name, answer_type, view_name), values in aggregate.items():
        summary[f"val-sqa3d/sqa3d/{metric_name}/{answer_type}/{view_name}"] = float(sum(values) / len(values))
    return summary


def ttrv_score_sqa3d(
    group_responses: list[str],
    ground_truth: str | None = None,
    alpha: float = 0.5,
    unknown_reward: float = -1.0,
    answer_dict_path: str | None = None,
) -> list[float]:
    answer_dict = _load_answer_dict(answer_dict_path)
    answer_prior = _load_answer_prior(answer_dict_path)
    answer_type_prior = _load_answer_type_prior(answer_dict_path)
    mapped = [
        map_to_sqa3d_answer(
            response,
            answer_dict,
            answer_prior=answer_prior,
            answer_type_prior=answer_type_prior,
        )
        for response in group_responses
    ]
    total = len(mapped)
    counts = Counter(mapped)
    entropy = _normalized_entropy(mapped)
    return [
        unknown_reward if pred == "unknown" else (counts[pred] / total) - alpha * entropy
        for pred in mapped
    ]


def _ground_truth_answers(ground_truth) -> list[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, list):
        return [str(item) for item in ground_truth]
    return [str(ground_truth)]


def _log_safe_text(text: str, max_chars: int) -> str:
    max_chars = int(max_chars)
    text = str(text).replace("\n", "\\n").replace("\r", "\\r")
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    alpha: float = 0.5,
    unknown_reward: float = -1.0,
    answer_dict_path: str | None = None,
    train_group_size: int | None = None,
    reward_mode: str = "train",
    reward_strategy: str = "ttrv_answer_space",
    em_bonus: float = 1.0,
    frequency_beta: float = 1.0,
    log_train_groups: bool = False,
    log_train_groups_limit: int = 0,
    log_train_response_chars: int = 160,
    **_kwargs,
):
    """Batch reward for SQA3D TTRV.

    Training groups repeated rollouts by extra_info["index"] and returns
    frequency - alpha * normalized_entropy over answers extracted into the
    official SQA3D answer_dict space by default. Validation returns the
    official SQA3D LLM free-form score; closed-set diagnostics are logged
    separately from the primary official metric.
    """
    answer_dict = _load_answer_dict(answer_dict_path)
    answer_prior = _load_answer_prior(answer_dict_path)
    answer_type_prior = _load_answer_type_prior(answer_dict_path)
    outputs = [None] * len(solution_strs)
    groups = defaultdict(list)
    for i, extra in enumerate(extra_infos):
        extra = extra or {}
        groups[extra.get("index", i)].append(i)

    alpha = float(alpha)
    unknown_reward = float(unknown_reward)
    train_group_size = int(train_group_size) if train_group_size is not None else None
    em_bonus = float(em_bonus)
    frequency_beta = float(frequency_beta)
    log_train_groups = _as_bool(log_train_groups)
    log_train_groups_limit = int(log_train_groups_limit)
    log_train_response_chars = int(log_train_response_chars)
    reward_strategy = str(reward_strategy or "ttrv_answer_space").lower()
    logged_groups = 0
    for group_key, indices in groups.items():
        responses = [solution_strs[i] for i in indices]
        use_freeform_reward = reward_strategy in {
            "freeform",
            "freeform_frequency",
            "ttrv_freeform",
            "ttrv_freeform_frequency",
        }
        if use_freeform_reward:
            mapped = [canonicalize_sqa3d_freeform(response) for response in responses]
        else:
            mapped = [
                map_to_sqa3d_answer(
                    response,
                    answer_dict,
                    answer_prior=answer_prior,
                    answer_type_prior=answer_type_prior,
                    question_type=(extra_infos[i] or {}).get(
                        "question_type", (extra_infos[i] or {}).get("answer_type")
                    ),
                )
                for response, i in zip(responses, indices)
            ]
        counts = Counter(mapped)
        raw_entropy = _entropy(mapped)
        entropy = _normalized_entropy(mapped)
        group_size = len(indices)
        is_eval = reward_mode == "eval" or (group_size == 1 and (train_group_size is None or train_group_size != 1))
        unique_pred_count = len(counts)
        unique_pred_ratio = unique_pred_count / max(group_size, 1)
        group_unknown_rate = sum(pred == "unknown" for pred in mapped) / max(group_size, 1)
        group_blank_rate = sum(not str(response).strip() for response in responses) / max(group_size, 1)
        group_records = []

        for local_i, global_i in enumerate(indices):
            pred = mapped[local_i]
            gt_answers = _ground_truth_answers(ground_truths[global_i])
            primary_gt = gt_answers[0] if gt_answers else None
            question_type = (extra_infos[global_i] or {}).get(
                "question_type", (extra_infos[global_i] or {}).get("answer_type")
            )
            official_acc = 1.0 if evaluate_sqa3d_official_freeform(solution_strs[global_i], primary_gt) else 0.0
            strict_acc = 1.0 if evaluate_sqa3d_strict_freeform(solution_strs[global_i], primary_gt) else 0.0
            closed_pred = map_to_sqa3d_answer(
                solution_strs[global_i],
                answer_dict,
                answer_prior=answer_prior,
                answer_type_prior=answer_type_prior,
                question_type=question_type,
            )
            closed_acc = (
                1.0
                if evaluate_sqa3d_closed_set_em(
                    solution_strs[global_i],
                    gt_answers,
                    answer_dict,
                    answer_prior=answer_prior,
                    answer_type_prior=answer_type_prior,
                    question_type=question_type,
                )
                else 0.0
            )
            acc = official_acc
            freq = 0.0 if is_eval else counts[pred] / group_size
            record_entropy = 0.0 if is_eval else entropy
            record_raw_entropy = 0.0 if is_eval else raw_entropy
            if is_eval:
                reward = acc
            elif pred == "unknown":
                reward = unknown_reward
            elif reward_strategy in {"hybrid", "em_hybrid", "official_hybrid"}:
                reward = em_bonus * acc + frequency_beta * freq - alpha * entropy
            else:
                reward = frequency_beta * freq - alpha * entropy
            group_records.append(
                {
                    "local_i": local_i,
                    "global_i": global_i,
                    "pred": pred,
                    "closed_pred": closed_pred,
                    "acc": acc,
                    "strict_acc": strict_acc,
                    "closed_acc": closed_acc,
                    "freeform_pred": canonicalize_sqa3d_freeform(solution_strs[global_i]),
                    "freq": freq,
                    "entropy": record_entropy,
                    "raw_entropy": record_raw_entropy,
                    "reward": reward,
                    "response": solution_strs[global_i],
                }
            )

        group_em_mean = sum(record["acc"] for record in group_records) / max(group_size, 1)
        group_pass = 1.0 if any(record["acc"] >= 0.5 for record in group_records) else 0.0

        if (
            log_train_groups
            and not is_eval
            and (log_train_groups_limit <= 0 or logged_groups < log_train_groups_limit)
        ):
            extra0 = extra_infos[indices[0]] or {}
            print(
                "[sqa3d_group] "
                f"index={group_key} question_id={extra0.get('question_id')} size={group_size} "
                f"unique={unique_pred_count} unique_ratio={unique_pred_ratio:.4f} "
                f"unknown_rate={group_unknown_rate:.4f} blank_rate={group_blank_rate:.4f} "
                f"em_mean={group_em_mean:.4f} pass@{group_size}={group_pass:.4f} "
                f"entropy={entropy:.4f} raw_entropy={raw_entropy:.4f} strategy={reward_strategy}"
            )
            for record in group_records:
                print(
                    "[sqa3d_vote] "
                    f"index={group_key} vote={record['local_i']} pred={record['pred']} "
                    f"freq={record['freq']:.4f} entropy={record['entropy']:.4f} "
                    f"reward={record['reward']:.4f} em={record['acc']:.1f} "
                    f"response={_log_safe_text(record['response'], log_train_response_chars)}"
                )
            logged_groups += 1

        for record in group_records:
            global_i = record["global_i"]
            extra = extra_infos[global_i] or {}
            outputs[global_i] = {
                "score": float(record["reward"]),
                "acc": float(record["acc"]),
                "official_acc": float(record["acc"]),
                "strict_acc": float(record["strict_acc"]),
                "closed_acc": float(record["closed_acc"]),
                "pred": record["pred"],
                "freeform_pred": record["freeform_pred"],
                "closed_pred": record["closed_pred"],
                "unknown": 1.0 if record["pred"] == "unknown" else 0.0,
                "closed_unknown": 1.0 if record["closed_pred"] == "unknown" else 0.0,
                "blank": 1.0 if not str(solution_strs[global_i]).strip() else 0.0,
                "freq": float(record["freq"]),
                "entropy": float(record["entropy"]),
                "raw_entropy": float(record["raw_entropy"]),
                "group_size": float(group_size),
                "unique_pred_count": float(unique_pred_count),
                "unique_pred_ratio": float(unique_pred_ratio),
                "group_unknown_rate": float(group_unknown_rate),
                "group_blank_rate": float(group_blank_rate),
                "group_em_mean": float(group_em_mean),
                "group_pass": float(group_pass),
                "answer_type": str(extra.get("question_type", extra.get("answer_type", "unknown"))),
            }

    return outputs
