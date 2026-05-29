import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CHOICES = {"A", "B", "C", "D"}
UNKNOWN = "unknown"
OPTION_TEXT_SKIP = {"a", "an", "the", "and", "or", "of", "with", "other", "object", "part"}
JOINT_OPTION_AMBIGUITY_GROUPS = (
    ("hinge", "revolute"),
    ("slider", "prismatic"),
    ("fixed", "rigid"),
)


def _normalize_option_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _option_tokens(text: str) -> list[str]:
    return [token for token in _normalize_option_text(text).split() if token not in OPTION_TEXT_SKIP]


def _option_phrase_variants(text: Any) -> list[str]:
    variants = [_normalize_option_text(str(text))]
    without_parenthetical = re.sub(r"\([^)]*\)", " ", str(text))
    variants.append(_normalize_option_text(without_parenthetical))
    seen = set()
    out = []
    for variant in variants:
        if len(variant) < 3 or variant in OPTION_TEXT_SKIP or variant in seen:
            continue
        seen.add(variant)
        out.append(variant)
    return out


def _extract_prompt_options(prompt: Any) -> dict[str, str]:
    if not prompt:
        return {}
    matches = re.findall(
        r"(?:^|\n)\s*([ABCD])\.\s*(.*?)(?=(?:\n\s*[ABCD]\.)|\s+ASSISTANT:|$)",
        str(prompt),
        flags=re.S,
    )
    return {choice.upper(): text.strip() for choice, text in matches}


def _match_option_text(raw: str, prompt: Any) -> str | None:
    options = _extract_prompt_options(prompt)
    if not options:
        return None
    raw_norm = _normalize_option_text(raw)
    matches: list[tuple[int, str]] = []
    for choice, option_text in options.items():
        for option_norm in _option_phrase_variants(option_text):
            if option_norm in raw_norm:
                matches.append((len(option_norm), choice))
    if not matches:
        raw_tokens = set(_option_tokens(raw))
        token_matches: list[tuple[int, float, int, str]] = []
        for choice, option_text in options.items():
            option_tokens = _option_tokens(option_text)
            if not option_tokens:
                continue
            overlap = sum(1 for token in set(option_tokens) if token in raw_tokens)
            if not overlap:
                continue
            ratio = overlap / len(set(option_tokens))
            if overlap == len(set(option_tokens)) or overlap >= 2:
                token_matches.append((overlap, ratio, len(set(option_tokens)), choice))
        if not token_matches:
            return None
        best = max(token_matches, key=lambda item: (item[0], item[1], item[2]))
        best_choices = [
            choice
            for overlap, ratio, token_count, choice in token_matches
            if (overlap, ratio, token_count) == best[:3]
        ]
        return best_choices[0] if len(best_choices) == 1 else None
    max_len = max(length for length, _choice in matches)
    choices = {choice for length, choice in matches if length == max_len}
    return next(iter(choices)) if len(choices) == 1 else None


def parse_prediction(raw: Any, prompt: Any = None) -> str:
    """Parse a free-form MCQ response into A/B/C/D/unknown."""
    if raw is None:
        return UNKNOWN
    text = str(raw).strip().replace("</s>", " ").replace("<|im_end|>", " ")
    first = re.match(r"^\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\s*$", text, flags=re.I)
    if first:
        return first.group(1).upper()
    first = re.match(r"^\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;-]", text, flags=re.I)
    if first:
        return first.group(1).upper()
    guided = re.search(
        r"(?:answer|option|choice|letter|choose|select|pick)\s*(?:is|:|-)?\s*[\(\[]?\s*['\"]?([ABCD])\b",
        text,
        flags=re.I,
    )
    if guided:
        return guided.group(1).upper()
    standalone = re.findall(r"\b([ABCD])\b", text)
    if standalone:
        # Only accept uppercase standalone letters here. A case-insensitive
        # fallback incorrectly maps the English article "a" to option A.
        return standalone[-1].upper()
    option_match = _match_option_text(text, prompt)
    if option_match:
        return option_match
    return UNKNOWN


def _has_ambiguous_joint_options(prompt: Any) -> bool:
    options = _extract_prompt_options(prompt)
    if len(options) < 2:
        return False
    normalized_options = {
        choice: f" {_normalize_option_text(option_text)} "
        for choice, option_text in options.items()
    }
    for terms in JOINT_OPTION_AMBIGUITY_GROUPS:
        term_choices: dict[str, set[str]] = defaultdict(set)
        for choice, option_text in normalized_options.items():
            for term in terms:
                if f" {term} " in option_text:
                    term_choices[term].add(choice)
        matched_terms = [term for term, choices in term_choices.items() if choices]
        matched_choices = set().union(*(term_choices[term] for term in matched_terms)) if matched_terms else set()
        if len(matched_terms) >= 2 and len(matched_choices) >= 2:
            return True
    return False


def _entropy(mapped: Iterable[str]) -> float:
    mapped = list(mapped)
    total = len(mapped)
    if total == 0:
        return 0.0
    counts = Counter(mapped)
    return -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())


def normalized_entropy(mapped: Iterable[str]) -> float:
    mapped = list(mapped)
    if len(mapped) <= 1:
        return 0.0
    entropy = _entropy(mapped)
    max_entropy = math.log(len(Counter(mapped)))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _ttrl_majority_label(mapped: Iterable[str]) -> tuple[str | None, int, float, bool]:
    """Return a unique known parsed majority label, if one exists."""
    mapped = list(mapped)
    total = len(mapped)
    if total == 0:
        return None, 0, 0.0, False
    counts = Counter(mapped)
    majority_count = max(counts.values())
    majority_values = [pred for pred, count in counts.items() if count == majority_count]
    majority_tie = len(majority_values) != 1
    majority_label = majority_values[0] if not majority_tie and majority_values[0] in CHOICES else None
    return majority_label, majority_count, majority_count / total, majority_tie


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_csv_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _geo_view_ids(group_size: int, geo_num_views: int, geo_samples_per_view: int | None) -> list[int]:
    if group_size <= 0:
        return []
    geo_num_views = max(1, int(geo_num_views))
    if geo_samples_per_view is None or geo_samples_per_view <= 0:
        geo_samples_per_view = max(1, math.ceil(group_size / geo_num_views))
    view_ids = []
    for local_i in range(group_size):
        view_ids.append(min(local_i // geo_samples_per_view, geo_num_views - 1))
    return view_ids


def _geo_harmony_label(
    parsed: list[str],
    view_ids: list[int],
    geo_num_views: int,
    min_view_support: int,
    min_hm: float,
    min_view_prob: float = 0.0,
    min_score_margin: float = 0.0,
    eps: float = 1e-6,
) -> tuple[str | None, float, int, bool, dict[str, float]]:
    """Select a pseudo-label by harmonic mean of per-view parsed-answer frequencies."""
    if not parsed or len(parsed) != len(view_ids):
        return None, 0.0, 0, False, {}
    geo_num_views = max(1, int(geo_num_views))
    min_view_support = max(1, int(min_view_support))
    view_counts: list[Counter[str]] = [Counter() for _ in range(geo_num_views)]
    view_totals = [0 for _ in range(geo_num_views)]
    for pred, view_id in zip(parsed, view_ids):
        if 0 <= int(view_id) < geo_num_views:
            view_counts[int(view_id)][pred] += 1
            view_totals[int(view_id)] += 1

    scores: dict[str, float] = {}
    support_counts: dict[str, int] = {}
    for choice in sorted(CHOICES):
        probs = []
        support = 0
        for view_id in range(geo_num_views):
            total = view_totals[view_id]
            prob = (view_counts[view_id].get(choice, 0) / total) if total else 0.0
            support += int(prob > 0)
            probs.append(prob)
        support_counts[choice] = support
        if support < min_view_support:
            scores[choice] = 0.0
            continue
        if min_view_prob > 0.0 and min(probs) < min_view_prob:
            scores[choice] = 0.0
            continue
        scores[choice] = geo_num_views / sum(1.0 / (prob + eps) for prob in probs)

    if not scores:
        return None, 0.0, 0, False, {}
    best_score = max(scores.values())
    if best_score < min_hm:
        return None, float(best_score), 0, False, scores
    if min_score_margin > 0.0:
        second_best = max((score for score in scores.values() if score < best_score), default=0.0)
        if best_score - second_best < min_score_margin:
            return None, float(best_score), 0, False, scores
    best_choices = [choice for choice, score in scores.items() if abs(score - best_score) <= 1e-12]
    tie = len(best_choices) != 1
    if tie:
        return None, float(best_score), max(support_counts.get(choice, 0) for choice in best_choices), True, scores
    best = best_choices[0]
    return best, float(best_score), support_counts.get(best, 0), False, scores


def _geo_harmony_soft_distribution(
    parsed: list[str],
    view_ids: list[int],
    geo_num_views: int,
    min_view_support: int,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> tuple[dict[str, float], dict[str, float], str | None, float, int, int, bool]:
    """Build a Geo-Harmony soft pseudo-label distribution over known MCQ choices.

    Per-view probabilities use the full number of samples in each view as the
    denominator, so unknown predictions reduce all known-choice probabilities
    for that view. Unknown is kept out of the normalized A/B/C/D soft label
    distribution.
    """
    zero_scores = {choice: 0.0 for choice in sorted(CHOICES)}
    if not parsed or len(parsed) != len(view_ids):
        return dict(zero_scores), dict(zero_scores), None, 0.0, 0, 0, False

    geo_num_views = max(1, int(geo_num_views))
    min_view_support = max(1, int(min_view_support))
    gamma = max(0.0, float(gamma))
    view_counts: list[Counter[str]] = [Counter() for _ in range(geo_num_views)]
    view_totals = [0 for _ in range(geo_num_views)]
    known_count = 0
    for pred, view_id in zip(parsed, view_ids):
        if not (0 <= int(view_id) < geo_num_views):
            continue
        view_id = int(view_id)
        view_totals[view_id] += 1
        if pred in CHOICES:
            view_counts[view_id][pred] += 1
            known_count += 1

    if known_count <= 0:
        return dict(zero_scores), dict(zero_scores), None, 0.0, 0, 0, False

    scores: dict[str, float] = {}
    support_counts: dict[str, int] = {}
    for choice in sorted(CHOICES):
        probs = []
        support = 0
        for view_id in range(geo_num_views):
            total = view_totals[view_id]
            prob = (view_counts[view_id].get(choice, 0) / total) if total else 0.0
            support += int(prob > 0.0)
            probs.append(prob)
        support_counts[choice] = support
        if support < min_view_support:
            scores[choice] = 0.0
            continue
        scores[choice] = geo_num_views / sum(1.0 / (prob + eps) for prob in probs)

    weights = {choice: (max(score, 0.0) ** gamma) for choice, score in scores.items()}
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        return dict(zero_scores), scores, None, 0.0, known_count, 0, False

    distribution = {choice: weights[choice] / weight_sum for choice in sorted(CHOICES)}
    top_prob = max(distribution.values())
    top_choices = [choice for choice, prob in distribution.items() if abs(prob - top_prob) <= 1e-12]
    tie = len(top_choices) != 1
    label = None if tie or top_prob <= 0.0 else top_choices[0]
    view_support = max((support_counts.get(choice, 0) for choice in top_choices), default=0)
    return distribution, scores, label, float(top_prob), known_count, view_support, tie


def _view_majority_label(parsed: list[str], view_ids: list[int], target_view_id: int = 0) -> tuple[str | None, int, float, bool]:
    view_preds = [pred for pred, view_id in zip(parsed, view_ids) if int(view_id) == int(target_view_id)]
    if not view_preds:
        return None, 0, 0.0, False
    counts = Counter(view_preds)
    majority_count = max(counts.values())
    majority_values = [pred for pred, count in counts.items() if count == majority_count]
    tie = len(majority_values) != 1
    label = majority_values[0] if not tie and majority_values[0] in CHOICES else None
    return label, majority_count, majority_count / len(view_preds), tie


def _answer_letter(ground_truth: Any, extra: dict[str, Any]) -> str | None:
    answer = extra.get("answer", ground_truth)
    if answer is None:
        return None
    answer = str(answer).strip().upper()
    return answer if answer in CHOICES else None


def _group_key(extra: dict[str, Any], index: int) -> Any:
    return extra.get("index", extra.get("id", index))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _log_safe_text(text: str, max_chars: int) -> str:
    text = str(text).replace("\n", "\\n").replace("\r", "\\r")
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def _prompt_for_index(extra_infos: list[Any], prompt_strs: list[Any] | None, index: int) -> Any:
    extra = extra_infos[index] or {}
    if isinstance(extra, dict):
        for key in ("prompt", "prompt_text", "raw_prompt"):
            if extra.get(key):
                return extra.get(key)
    if prompt_strs is not None and index < len(prompt_strs):
        return prompt_strs[index]
    return None


def _compute_ttrv_reward(
    pred: str,
    freq: float,
    entropy: float,
    alpha: float,
    frequency_beta: float,
    unknown_reward: float,
    reward_variant: str,
    ttrl_pseudo_label: str | None = None,
    ttrl_majority_tie: bool = False,
    geo_soft_distribution: dict[str, float] | None = None,
    geo_soft_selected: bool = False,
) -> float:
    if reward_variant == "geo_harmony_soft":
        if not geo_soft_selected:
            return 0.0
        if pred == UNKNOWN:
            return unknown_reward
        return float((geo_soft_distribution or {}).get(pred, 0.0))
    if pred == UNKNOWN:
        return unknown_reward
    if reward_variant == "ttrl_majority_vote":
        if ttrl_majority_tie or ttrl_pseudo_label is None:
            return 0.0
        return 1.0 if pred == ttrl_pseudo_label else 0.0
    if reward_variant == "geo_harmony":
        if ttrl_majority_tie or ttrl_pseudo_label is None:
            return 0.0
        return 1.0 if pred == ttrl_pseudo_label else 0.0
    if reward_variant == "freq_entropy_product":
        return freq * (frequency_beta - alpha * entropy)
    if reward_variant == "entropy_weighted_frequency":
        return frequency_beta * freq * max(0.0, 1.0 - alpha * entropy)
    return frequency_beta * freq - alpha * entropy


def compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    prompt_strs: list[Any] | None = None,
    alpha: float = 0.75,
    frequency_beta: float = 1.0,
    unknown_reward: float = -1.0,
    train_group_size: int | None = None,
    reward_mode: str = "train",
    log_train_groups: bool = False,
    log_train_groups_limit: int = 0,
    log_train_response_chars: int = 160,
    reward_variant: str = "standard",
    ttrl_min_majority_ratio: float = 0.0,
    ttrl_max_majority_ratio: float = 1.0,
    geo_num_views: int = 1,
    geo_samples_per_view: int | None = None,
    geo_min_view_support: int = 0,
    geo_min_hm: float = 0.0,
    geo_min_view_prob: float = 0.0,
    geo_min_score_margin: float = 0.0,
    geo_skip_question_types: str = "",
    geo_require_original_majority: bool = False,
    geo_min_original_majority_ratio: float = 0.0,
    geo_skip_ambiguous_joint_options: bool = False,
    geo_soft_gamma: float = 2.0,
    geo_soft_min_max_prob: float = 0.0,
    geo_soft_min_known_count: int = 0,
    **_kwargs,
):
    """Batch reward for PhysX-MCQ TTRV.

    Raw responses are always parsed to A/B/C/D/unknown first. Training reward is
    pure TTRV self-consistency over parsed predictions:

        frequency_beta * frequency - alpha * normalized_entropy

    Ground-truth answers are used only to report accuracy. Validation returns
    accuracy as score while still logging the TTRV reward components.
    """
    alpha = float(alpha)
    frequency_beta = float(frequency_beta)
    unknown_reward = float(unknown_reward)
    reward_mode = str(reward_mode or "train").lower()
    reward_variant = str(reward_variant or "standard").strip().lower()
    is_geo_variant = reward_variant in {"geo_harmony", "geo_harmony_soft"}
    ttrl_min_majority_ratio = float(ttrl_min_majority_ratio)
    ttrl_max_majority_ratio = float(ttrl_max_majority_ratio)
    geo_num_views = max(1, _safe_int(geo_num_views, 1))
    geo_samples_per_view = _safe_int(geo_samples_per_view, 0) or None
    geo_min_view_support = _safe_int(geo_min_view_support, 0)
    if geo_min_view_support <= 0:
        geo_min_view_support = geo_num_views if is_geo_variant else 1
    geo_min_hm = _safe_float(geo_min_hm, 0.0)
    geo_min_view_prob = _safe_float(geo_min_view_prob, 0.0)
    geo_min_score_margin = _safe_float(geo_min_score_margin, 0.0)
    geo_skip_question_types_set = _split_csv_set(geo_skip_question_types)
    geo_require_original_majority = _as_bool(geo_require_original_majority)
    geo_min_original_majority_ratio = _safe_float(geo_min_original_majority_ratio, 0.0)
    geo_skip_ambiguous_joint_options = _as_bool(geo_skip_ambiguous_joint_options)
    geo_soft_gamma = _safe_float(geo_soft_gamma, 2.0)
    geo_soft_min_max_prob = _safe_float(geo_soft_min_max_prob, 0.0)
    geo_soft_min_known_count = _safe_int(geo_soft_min_known_count, 0)
    train_group_size = int(train_group_size) if train_group_size is not None else None
    log_train_groups = _as_bool(log_train_groups)
    log_train_groups_limit = int(log_train_groups_limit)
    log_train_response_chars = int(log_train_response_chars)

    outputs = [None] * len(solution_strs)
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, extra in enumerate(extra_infos):
        groups[_group_key(extra or {}, i)].append(i)

    logged_groups = 0
    for group_key, indices in groups.items():
        responses = [solution_strs[i] for i in indices]
        parsed = [
            parse_prediction(response, _prompt_for_index(extra_infos, prompt_strs, i))
            for i, response in zip(indices, responses)
        ]
        counts = Counter(parsed)
        group_size = len(indices)
        raw_entropy = _entropy(parsed)
        entropy = normalized_entropy(parsed)
        ttrl_pseudo_label, ttrl_majority_count, ttrl_majority_ratio, ttrl_majority_tie = _ttrl_majority_label(parsed)
        if ttrl_pseudo_label is not None and ttrl_majority_ratio < ttrl_min_majority_ratio:
            ttrl_pseudo_label = None
        if ttrl_pseudo_label is not None and ttrl_majority_ratio > ttrl_max_majority_ratio:
            ttrl_pseudo_label = None
        geo_view_ids = _geo_view_ids(group_size, geo_num_views, geo_samples_per_view)
        geo_label = None
        geo_hm = 0.0
        geo_view_support = 0
        geo_tie = False
        geo_scores = {}
        geo_skip_reason = ""
        geo_original_label = None
        geo_original_count = 0
        geo_original_ratio = 0.0
        geo_original_tie = False
        geo_soft_distribution = {choice: 0.0 for choice in sorted(CHOICES)}
        geo_soft_scores = {choice: 0.0 for choice in sorted(CHOICES)}
        geo_soft_label = None
        geo_soft_top_prob = 0.0
        geo_soft_known_count = 0
        geo_soft_view_support = 0
        geo_soft_tie = False
        geo_soft_selected = False
        question_type0 = str((extra_infos[indices[0]] or {}).get("question_type", "unknown")) if indices else "unknown"
        geo_skipped = is_geo_variant and question_type0 in geo_skip_question_types_set
        if geo_skipped:
            geo_skip_reason = f"question_type:{question_type0}"
        if (
            is_geo_variant
            and not geo_skipped
            and geo_skip_ambiguous_joint_options
            and question_type0 == "joint_type"
            and _has_ambiguous_joint_options(_prompt_for_index(extra_infos, prompt_strs, indices[0]))
        ):
            geo_skipped = True
            geo_skip_reason = "ambiguous_joint_options"
        if reward_variant == "geo_harmony" and not geo_skipped:
            geo_label, geo_hm, geo_view_support, geo_tie, geo_scores = _geo_harmony_label(
                parsed=parsed,
                view_ids=geo_view_ids,
                geo_num_views=geo_num_views,
                min_view_support=geo_min_view_support,
                min_hm=geo_min_hm,
                min_view_prob=geo_min_view_prob,
                min_score_margin=geo_min_score_margin,
            )
            geo_original_label, geo_original_count, geo_original_ratio, geo_original_tie = _view_majority_label(
                parsed=parsed,
                view_ids=geo_view_ids,
                target_view_id=0,
            )
            if geo_label is not None and geo_require_original_majority:
                if geo_original_tie or geo_original_label != geo_label:
                    geo_label = None
                elif geo_original_ratio < geo_min_original_majority_ratio:
                    geo_label = None
            ttrl_pseudo_label = geo_label
            ttrl_majority_count = counts.get(geo_label, 0) if geo_label else 0
            ttrl_majority_ratio = ttrl_majority_count / max(group_size, 1)
            ttrl_majority_tie = geo_tie
            if geo_label is not None and ttrl_majority_ratio < ttrl_min_majority_ratio:
                ttrl_pseudo_label = None
            if geo_label is not None and ttrl_majority_ratio > ttrl_max_majority_ratio:
                ttrl_pseudo_label = None
        elif reward_variant == "geo_harmony_soft" and not geo_skipped:
            (
                geo_soft_distribution,
                geo_soft_scores,
                geo_soft_label,
                geo_soft_top_prob,
                geo_soft_known_count,
                geo_soft_view_support,
                geo_soft_tie,
            ) = _geo_harmony_soft_distribution(
                parsed=parsed,
                view_ids=geo_view_ids,
                geo_num_views=geo_num_views,
                min_view_support=geo_min_view_support,
                gamma=geo_soft_gamma,
            )
            geo_label = geo_soft_label
            geo_hm = max(geo_soft_scores.values(), default=0.0)
            geo_view_support = geo_soft_view_support
            geo_tie = geo_soft_tie
            geo_scores = geo_soft_scores
            geo_original_label, geo_original_count, geo_original_ratio, geo_original_tie = _view_majority_label(
                parsed=parsed,
                view_ids=geo_view_ids,
                target_view_id=0,
            )
            if geo_soft_label is not None:
                geo_soft_selected = True
                if geo_min_hm > 0.0 and geo_hm < geo_min_hm:
                    geo_soft_selected = False
                if geo_soft_min_known_count > 0 and geo_soft_known_count < geo_soft_min_known_count:
                    geo_soft_selected = False
                if geo_soft_min_max_prob > 0.0 and geo_soft_top_prob < geo_soft_min_max_prob:
                    geo_soft_selected = False
            ttrl_pseudo_label = geo_soft_label if geo_soft_selected else None
            ttrl_majority_count = counts.get(geo_soft_label, 0) if geo_soft_label else 0
            ttrl_majority_ratio = ttrl_majority_count / max(group_size, 1)
            ttrl_majority_tie = geo_soft_tie
        if is_geo_variant and geo_skipped:
            ttrl_pseudo_label = None
            ttrl_majority_count = 0
            ttrl_majority_ratio = 0.0
            ttrl_majority_tie = False
            geo_soft_selected = False
        ttrl_pseudo_label_for_log = ttrl_pseudo_label or "none"
        is_eval = reward_mode == "eval"
        unique_pred_count = len(counts)
        unique_pred_ratio = unique_pred_count / max(group_size, 1)
        group_unknown_rate = sum(pred == UNKNOWN for pred in parsed) / max(group_size, 1)
        group_records = []

        for local_i, global_i in enumerate(indices):
            extra = extra_infos[global_i] or {}
            pred = parsed[local_i]
            answer = _answer_letter(ground_truths[global_i], extra)
            acc = 1.0 if answer is not None and pred == answer else 0.0
            freq = counts[pred] / max(group_size, 1)
            ttrv_reward = _compute_ttrv_reward(
                pred=pred,
                freq=freq,
                entropy=entropy,
                alpha=alpha,
                frequency_beta=frequency_beta,
                unknown_reward=unknown_reward,
                reward_variant=reward_variant,
                ttrl_pseudo_label=ttrl_pseudo_label,
                ttrl_majority_tie=ttrl_majority_tie,
                geo_soft_distribution=geo_soft_distribution,
                geo_soft_selected=geo_soft_selected,
            )
            if geo_skipped:
                ttrv_reward = 0.0
            score = acc if is_eval else ttrv_reward
            geo_soft_gt_mass = float(geo_soft_distribution.get(answer, 0.0)) if answer in CHOICES else 0.0
            record = {
                "local_i": local_i,
                "global_i": global_i,
                "id": extra.get("id", group_key),
                "question_type": str(extra.get("question_type", "unknown")),
                "answer": answer,
                "prediction_raw": solution_strs[global_i],
                "prediction": pred,
                "frequency": float(freq),
                "normalized_entropy": float(entropy),
                "raw_entropy": float(raw_entropy),
                "ttrv_reward": float(ttrv_reward),
                "reward_variant": reward_variant,
                "ttrl_pseudo_label": ttrl_pseudo_label_for_log,
                "ttrl_majority_count": float(ttrl_majority_count),
                "ttrl_majority_ratio": float(ttrl_majority_ratio),
                "ttrl_majority_tie": bool(ttrl_majority_tie),
                "ttrl_min_majority_ratio": float(ttrl_min_majority_ratio),
                "ttrl_max_majority_ratio": float(ttrl_max_majority_ratio),
                "geo_harmony_label": geo_label or "none",
                "geo_harmony_hm": float(geo_hm),
                "geo_harmony_scores": dict(sorted((geo_scores or {}).items())),
                "geo_harmony_score_A": float((geo_scores or {}).get("A", 0.0)),
                "geo_harmony_score_B": float((geo_scores or {}).get("B", 0.0)),
                "geo_harmony_score_C": float((geo_scores or {}).get("C", 0.0)),
                "geo_harmony_score_D": float((geo_scores or {}).get("D", 0.0)),
                "geo_original_majority_label": geo_original_label or "none",
                "geo_original_majority_count": float(geo_original_count),
                "geo_original_majority_ratio": float(geo_original_ratio),
                "geo_original_majority_tie": bool(geo_original_tie),
                "geo_require_original_majority": bool(geo_require_original_majority),
                "geo_min_original_majority_ratio": float(geo_min_original_majority_ratio),
                "geo_skip_ambiguous_joint_options": bool(geo_skip_ambiguous_joint_options),
                "geo_view_id": int(geo_view_ids[local_i]) if local_i < len(geo_view_ids) else 0,
                "geo_view_support": float(geo_view_support),
                "geo_num_views": float(geo_num_views),
                "geo_samples_per_view": float(geo_samples_per_view or 0),
                "geo_min_view_support": float(geo_min_view_support),
                "geo_min_hm": float(geo_min_hm),
                "geo_min_view_prob": float(geo_min_view_prob),
                "geo_min_score_margin": float(geo_min_score_margin),
                "geo_skipped": bool(geo_skipped),
                "geo_skip_reason": geo_skip_reason,
                "geo_soft_label": geo_soft_label or "none",
                "geo_soft_top_prob": float(geo_soft_top_prob),
                "geo_soft_known_count": float(geo_soft_known_count),
                "geo_soft_selected": bool(geo_soft_selected),
                "geo_soft_tie": bool(geo_soft_tie),
                "geo_soft_view_support": float(geo_soft_view_support),
                "geo_soft_gamma": float(geo_soft_gamma),
                "geo_soft_min_max_prob": float(geo_soft_min_max_prob),
                "geo_soft_min_known_count": float(geo_soft_min_known_count),
                "geo_soft_distribution": dict(sorted(geo_soft_distribution.items())),
                "geo_soft_scores": dict(sorted(geo_soft_scores.items())),
                "geo_soft_prob_A": float(geo_soft_distribution.get("A", 0.0)),
                "geo_soft_prob_B": float(geo_soft_distribution.get("B", 0.0)),
                "geo_soft_prob_C": float(geo_soft_distribution.get("C", 0.0)),
                "geo_soft_prob_D": float(geo_soft_distribution.get("D", 0.0)),
                "geo_soft_score_A": float(geo_soft_scores.get("A", 0.0)),
                "geo_soft_score_B": float(geo_soft_scores.get("B", 0.0)),
                "geo_soft_score_C": float(geo_soft_scores.get("C", 0.0)),
                "geo_soft_score_D": float(geo_soft_scores.get("D", 0.0)),
                "geo_soft_gt_mass": float(geo_soft_gt_mass),
                "score": float(score),
                "acc": float(acc),
                "correct": bool(acc),
                "invalid": pred == UNKNOWN,
            }
            group_records.append(record)

        group_acc_values = [float(record["acc"]) for record in group_records]
        group_acc_mean = sum(group_acc_values) / max(group_size, 1)
        group_pass = max(group_acc_values) if group_acc_values else 0.0
        group_best_acc = group_pass
        group_worst_acc = min(group_acc_values) if group_acc_values else 0.0
        majority_pred = counts.most_common(1)[0][0] if counts else UNKNOWN
        majority_records = [record for record in group_records if record["prediction"] == majority_pred]
        group_majority_acc = float(majority_records[0]["acc"]) if majority_records else 0.0
        if log_train_groups and not is_eval and (log_train_groups_limit <= 0 or logged_groups < log_train_groups_limit):
            extra0 = extra_infos[indices[0]] or {}
            print(
                "[physx_mcq_group] "
                f"index={group_key} id={extra0.get('id')} question_type={extra0.get('question_type')} "
                f"size={group_size} unique={unique_pred_count} unique_ratio={unique_pred_ratio:.4f} "
                f"unknown_rate={group_unknown_rate:.4f} acc_mean={group_acc_mean:.4f} "
                f"pass={group_pass:.4f} maj_pred={majority_pred} maj_acc={group_majority_acc:.4f} "
                f"ttrl_label={ttrl_pseudo_label} ttrl_tie={ttrl_majority_tie} "
                f"entropy={entropy:.4f} raw_entropy={raw_entropy:.4f}"
            )
            for record in group_records:
                print(
                    "[physx_mcq_vote] "
                f"index={group_key} vote={record['local_i']} pred={record['prediction']} "
                f"freq={record['frequency']:.4f} entropy={record['normalized_entropy']:.4f} "
                f"reward={record['ttrv_reward']:.4f} variant={reward_variant} "
                f"pseudo={record['ttrl_pseudo_label']} tie={record['ttrl_majority_tie']} "
                f"geo_view={record['geo_view_id']} geo_hm={record['geo_harmony_hm']:.4f} "
                f"acc={record['acc']:.1f} "
                f"response={_log_safe_text(record['prediction_raw'], log_train_response_chars)}"
            )
            logged_groups += 1

        for record in group_records:
            global_i = record["global_i"]
            extra = extra_infos[global_i] or {}
            outputs[global_i] = {
                "score": float(record["score"]),
                "acc": float(record["acc"]),
                "official_acc": float(record["acc"]),
                "pred": record["prediction"],
                "prediction": record["prediction"],
                "prediction_raw": record["prediction_raw"],
                "answer": record["answer"],
                "correct": float(record["acc"]),
                "unknown": 1.0 if record["prediction"] == UNKNOWN else 0.0,
                "invalid": 1.0 if record["prediction"] == UNKNOWN else 0.0,
                "blank": 1.0 if not str(record["prediction_raw"]).strip() else 0.0,
                "freq": float(record["frequency"]),
                "frequency": float(record["frequency"]),
                "entropy": float(record["normalized_entropy"]),
                "normalized_entropy": float(record["normalized_entropy"]),
                "raw_entropy": float(record["raw_entropy"]),
                "ttrv_reward": float(record["ttrv_reward"]),
                "reward_variant": reward_variant,
                "ttrl_pseudo_label": record["ttrl_pseudo_label"],
                "ttrl_majority_count": float(record["ttrl_majority_count"]),
                "ttrl_majority_ratio": float(record["ttrl_majority_ratio"]),
                "ttrl_majority_tie": float(record["ttrl_majority_tie"]),
                "ttrl_min_majority_ratio": float(record["ttrl_min_majority_ratio"]),
                "ttrl_max_majority_ratio": float(record["ttrl_max_majority_ratio"]),
                "geo_harmony_label": record["geo_harmony_label"],
                "geo_harmony_hm": float(record["geo_harmony_hm"]),
                "geo_harmony_scores_json": json.dumps(record["geo_harmony_scores"], sort_keys=True),
                "geo_harmony_score_A": float(record["geo_harmony_score_A"]),
                "geo_harmony_score_B": float(record["geo_harmony_score_B"]),
                "geo_harmony_score_C": float(record["geo_harmony_score_C"]),
                "geo_harmony_score_D": float(record["geo_harmony_score_D"]),
                "geo_original_majority_label": record["geo_original_majority_label"],
                "geo_original_majority_count": float(record["geo_original_majority_count"]),
                "geo_original_majority_ratio": float(record["geo_original_majority_ratio"]),
                "geo_original_majority_tie": float(record["geo_original_majority_tie"]),
                "geo_require_original_majority": 1.0 if record["geo_require_original_majority"] else 0.0,
                "geo_min_original_majority_ratio": float(record["geo_min_original_majority_ratio"]),
                "geo_skip_ambiguous_joint_options": (
                    1.0 if record["geo_skip_ambiguous_joint_options"] else 0.0
                ),
                "geo_view_id": float(record["geo_view_id"]),
                "geo_view_support": float(record["geo_view_support"]),
                "geo_num_views": float(record["geo_num_views"]),
                "geo_samples_per_view": float(record["geo_samples_per_view"]),
                "geo_min_view_support": float(record["geo_min_view_support"]),
                "geo_min_hm": float(record["geo_min_hm"]),
                "geo_min_view_prob": float(record["geo_min_view_prob"]),
                "geo_min_score_margin": float(record["geo_min_score_margin"]),
                "geo_skipped": 1.0 if record["geo_skipped"] else 0.0,
                "geo_skip_reason": record["geo_skip_reason"],
                "geo_soft_label": record["geo_soft_label"],
                "geo_soft_top_prob": float(record["geo_soft_top_prob"]),
                "geo_soft_known_count": float(record["geo_soft_known_count"]),
                "geo_soft_selected": 1.0 if record["geo_soft_selected"] else 0.0,
                "geo_soft_tie": 1.0 if record["geo_soft_tie"] else 0.0,
                "geo_soft_view_support": float(record["geo_soft_view_support"]),
                "geo_soft_gamma": float(record["geo_soft_gamma"]),
                "geo_soft_min_max_prob": float(record["geo_soft_min_max_prob"]),
                "geo_soft_min_known_count": float(record["geo_soft_min_known_count"]),
                "geo_soft_distribution_json": json.dumps(record["geo_soft_distribution"], sort_keys=True),
                "geo_soft_scores_json": json.dumps(record["geo_soft_scores"], sort_keys=True),
                "geo_soft_prob_A": float(record["geo_soft_prob_A"]),
                "geo_soft_prob_B": float(record["geo_soft_prob_B"]),
                "geo_soft_prob_C": float(record["geo_soft_prob_C"]),
                "geo_soft_prob_D": float(record["geo_soft_prob_D"]),
                "geo_soft_score_A": float(record["geo_soft_score_A"]),
                "geo_soft_score_B": float(record["geo_soft_score_B"]),
                "geo_soft_score_C": float(record["geo_soft_score_C"]),
                "geo_soft_score_D": float(record["geo_soft_score_D"]),
                "geo_soft_gt_mass": float(record["geo_soft_gt_mass"]),
                "group_size": float(group_size),
                "unique_pred_count": float(unique_pred_count),
                "unique_pred_ratio": float(unique_pred_ratio),
                "group_unknown_rate": float(group_unknown_rate),
                "group_acc_mean": float(group_acc_mean),
                "group_pass": float(group_pass),
                "group_best_acc": float(group_best_acc),
                "group_worst_acc": float(group_worst_acc),
                "group_majority_prediction": majority_pred,
                "group_majority_acc": float(group_majority_acc),
                "question_type": str(extra.get("question_type", "unknown")),
                "answer_type": str(extra.get("question_type", "unknown")),
                "task": "physx_mcq",
            }

    return outputs


def summarize_physx_validation(data_sources, group_keys, infos_dict: dict[str, list[Any]]) -> dict[str, float]:
    preds = infos_dict.get("prediction") or infos_dict.get("pred") or []
    accs = infos_dict.get("acc") or []
    qtypes = infos_dict.get("question_type") or infos_dict.get("answer_type") or []
    unknowns = infos_dict.get("unknown") or infos_dict.get("invalid") or []
    if not accs or not qtypes:
        return {}

    rows = []
    for i, acc in enumerate(accs):
        data_source = str(data_sources[i]) if i < len(data_sources) else ""
        if data_source != "physx_mcq":
            continue
        pred = preds[i] if i < len(preds) else None
        unknown = unknowns[i] if i < len(unknowns) else (1.0 if pred == UNKNOWN else 0.0)
        rows.append(
            {
                "acc": float(acc),
                "question_type": str(qtypes[i] if i < len(qtypes) else "unknown"),
                "unknown": float(unknown),
            }
        )
    if not rows:
        return {}

    def add_metrics(prefix: str, subset: list[dict[str, Any]], out: dict[str, float]) -> None:
        n = len(subset)
        out[f"{prefix}/num_examples"] = float(n)
        out[f"{prefix}/accuracy"] = float(sum(row["acc"] for row in subset) / n) if n else 0.0
        out[f"{prefix}/invalid_rate"] = float(sum(row["unknown"] for row in subset) / n) if n else 0.0

    metrics: dict[str, float] = {}
    add_metrics("val-physx/overall", rows, metrics)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    for question_type, subset in sorted(by_type.items()):
        safe_type = re.sub(r"[^A-Za-z0-9_.-]+", "_", question_type)
        add_metrics(f"val-physx/question_type/{safe_type}", subset, metrics)
    return metrics



def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _record_group_key(record: dict[str, Any], fallback: int) -> str:
    return str(
        record.get("question_id")
        or record.get("index")
        or record.get("id")
        or record.get("sample_index")
        or fallback
    )


def summarize_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"num_evaluated": 0, "num_correct": 0, "invalid_outputs": 0})
    num_correct = 0
    invalid = 0
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    numeric_keys = [
        "frequency",
        "normalized_entropy",
        "raw_entropy",
        "ttrv_reward",
        "reward",
        "unique_pred_ratio",
        "group_unknown_rate",
        "response_token_len",
        "ttrl_majority_count",
        "ttrl_majority_ratio",
        "ttrl_majority_tie",
        "ttrl_min_majority_ratio",
        "ttrl_max_majority_ratio",
        "geo_harmony_hm",
        "geo_original_majority_count",
        "geo_original_majority_ratio",
        "geo_original_majority_tie",
        "geo_require_original_majority",
        "geo_min_original_majority_ratio",
        "geo_view_id",
        "geo_view_support",
        "geo_num_views",
        "geo_samples_per_view",
        "geo_min_view_support",
        "geo_min_hm",
        "geo_min_view_prob",
        "geo_min_score_margin",
        "geo_skipped",
        "geo_soft_top_prob",
        "geo_soft_known_count",
        "geo_soft_selected",
        "geo_soft_tie",
        "geo_soft_view_support",
        "geo_soft_gamma",
        "geo_soft_min_max_prob",
        "geo_soft_min_known_count",
        "geo_soft_prob_A",
        "geo_soft_prob_B",
        "geo_soft_prob_C",
        "geo_soft_prob_D",
        "geo_soft_score_A",
        "geo_soft_score_B",
        "geo_soft_score_C",
        "geo_soft_score_D",
        "geo_soft_gt_mass",
    ]
    numeric_values: dict[str, list[float]] = {key: [] for key in numeric_keys}
    prediction_distribution: Counter[str] = Counter()
    pseudo_label_distribution: Counter[str] = Counter()
    hit_max_count = 0

    for i, record in enumerate(records):
        pred = record.get("prediction", record.get("pred"))
        if pred is None and record.get("response") is not None:
            pred = parse_prediction(record.get("response"), record.get("prompt"))
        pred = str(pred or UNKNOWN)
        prediction_distribution[pred] += 1
        pseudo_label = record.get("ttrl_pseudo_label")
        pseudo_label_distribution[str(pseudo_label or "none")] += 1
        qtype = str(record.get("question_type") or "unknown")
        acc_val = record.get("acc", record.get("correct", 0.0))
        correct = bool(_to_float(acc_val)) if not isinstance(acc_val, bool) else acc_val
        is_invalid = pred == UNKNOWN or bool(_to_float(record.get("invalid", 0.0)))
        hit_max_count += int(bool(record.get("hit_max_response_length")))
        by_type[qtype]["num_evaluated"] += 1
        by_type[qtype]["num_correct"] += int(correct)
        by_type[qtype]["invalid_outputs"] += int(is_invalid)
        num_correct += int(correct)
        invalid += int(is_invalid)
        groups[_record_group_key(record, i)].append({**record, "_pred": pred, "_acc": float(correct), "_invalid": float(is_invalid), "_qtype": qtype})
        for key in numeric_keys:
            if record.get(key) is not None:
                numeric_values[key].append(_to_float(record.get(key)))

    group_rows = []
    for group_key, group_records in groups.items():
        preds = [str(row.get("_pred") or UNKNOWN) for row in group_records]
        accs = [_to_float(row.get("_acc")) for row in group_records]
        invalids = [_to_float(row.get("_invalid")) for row in group_records]
        counts = Counter(preds)
        majority_pred = counts.most_common(1)[0][0] if counts else UNKNOWN
        majority_acc = 0.0
        for row in group_records:
            if str(row.get("_pred") or UNKNOWN) == majority_pred:
                majority_acc = _to_float(row.get("_acc"))
                break
        group_rows.append(
            {
                "group_key": group_key,
                "question_type": str(group_records[0].get("_qtype", "unknown")),
                "num_votes": len(group_records),
                "response_accuracy": _mean(accs),
                "majority_prediction": majority_pred,
                "majority_accuracy": majority_acc,
                "pass_at_votes": max(accs) if accs else 0.0,
                "best_accuracy_at_votes": max(accs) if accs else 0.0,
                "worst_accuracy_at_votes": min(accs) if accs else 0.0,
                "invalid_rate": _mean(invalids),
            }
        )

    num_evaluated = len(records)
    accuracy_by_type = {}
    for qtype, row in by_type.items():
        den = row["num_evaluated"]
        accuracy_by_type[qtype] = {**row, "accuracy": row["num_correct"] / den if den else 0.0}

    group_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        group_by_type[row["question_type"]].append(row)
    vote_by_question_type = {}
    for qtype, rows in group_by_type.items():
        mean_votes = _mean([_to_float(row["num_votes"]) for row in rows])
        response_accuracy = _mean([_to_float(row["response_accuracy"]) for row in rows])
        vote_by_question_type[qtype] = {
            "num_groups": len(rows),
            "mean_votes_per_group": mean_votes,
            "response_accuracy": response_accuracy,
            "acc_at_1": response_accuracy if abs(mean_votes - 1.0) < 1e-9 else None,
            "majority_accuracy": _mean([_to_float(row["majority_accuracy"]) for row in rows]),
            "pass_at_votes": _mean([_to_float(row["pass_at_votes"]) for row in rows]),
            "best_accuracy_at_votes": _mean([_to_float(row["best_accuracy_at_votes"]) for row in rows]),
            "worst_accuracy_at_votes": _mean([_to_float(row["worst_accuracy_at_votes"]) for row in rows]),
            "invalid_rate": _mean([_to_float(row["invalid_rate"]) for row in rows]),
        }

    mean_votes_per_group = _mean([_to_float(row["num_votes"]) for row in group_rows])
    response_accuracy = num_correct / num_evaluated if num_evaluated else 0.0
    ttrv_metrics = {
        "num_groups": len(group_rows),
        "mean_votes_per_group": mean_votes_per_group,
        "response_accuracy": response_accuracy,
        "acc_at_1": response_accuracy if abs(mean_votes_per_group - 1.0) < 1e-9 else None,
        "majority_accuracy": _mean([_to_float(row["majority_accuracy"]) for row in group_rows]),
        "pass_at_votes": _mean([_to_float(row["pass_at_votes"]) for row in group_rows]),
        "best_accuracy_at_votes": _mean([_to_float(row["best_accuracy_at_votes"]) for row in group_rows]),
        "worst_accuracy_at_votes": _mean([_to_float(row["worst_accuracy_at_votes"]) for row in group_rows]),
        "vote_invalid_rate": invalid / num_evaluated if num_evaluated else 0.0,
        "group_invalid_rate": _mean([_to_float(row["invalid_rate"]) for row in group_rows]),
        "frequency_mean": _mean(numeric_values["frequency"]),
        "normalized_entropy_mean": _mean(numeric_values["normalized_entropy"]),
        "raw_entropy_mean": _mean(numeric_values["raw_entropy"]),
        "ttrv_reward_mean": _mean(numeric_values["ttrv_reward"]),
        "reward_mean": _mean(numeric_values["reward"]),
        "unique_pred_ratio_mean": _mean(numeric_values["unique_pred_ratio"]),
        "group_unknown_rate_mean": _mean(numeric_values["group_unknown_rate"]),
        "ttrl_majority_count_mean": _mean(numeric_values["ttrl_majority_count"]),
        "ttrl_majority_ratio_mean": _mean(numeric_values["ttrl_majority_ratio"]),
        "ttrl_majority_tie_mean": _mean(numeric_values["ttrl_majority_tie"]),
        "ttrl_max_majority_ratio_mean": _mean(numeric_values["ttrl_max_majority_ratio"]),
        "geo_harmony_hm_mean": _mean(numeric_values["geo_harmony_hm"]),
        "geo_original_majority_ratio_mean": _mean(numeric_values["geo_original_majority_ratio"]),
        "geo_original_majority_tie_mean": _mean(numeric_values["geo_original_majority_tie"]),
        "geo_view_support_mean": _mean(numeric_values["geo_view_support"]),
        "geo_min_view_prob_mean": _mean(numeric_values["geo_min_view_prob"]),
        "geo_min_score_margin_mean": _mean(numeric_values["geo_min_score_margin"]),
        "geo_skipped_rate": _mean(numeric_values["geo_skipped"]),
        "geo_soft_top_prob_mean": _mean(numeric_values["geo_soft_top_prob"]),
        "geo_soft_known_count_mean": _mean(numeric_values["geo_soft_known_count"]),
        "geo_soft_selected_rate": _mean(numeric_values["geo_soft_selected"]),
        "geo_soft_tie_rate": _mean(numeric_values["geo_soft_tie"]),
        "geo_soft_view_support_mean": _mean(numeric_values["geo_soft_view_support"]),
        "geo_soft_gt_mass_mean": _mean(numeric_values["geo_soft_gt_mass"]),
        "geo_soft_prob_A_mean": _mean(numeric_values["geo_soft_prob_A"]),
        "geo_soft_prob_B_mean": _mean(numeric_values["geo_soft_prob_B"]),
        "geo_soft_prob_C_mean": _mean(numeric_values["geo_soft_prob_C"]),
        "geo_soft_prob_D_mean": _mean(numeric_values["geo_soft_prob_D"]),
        "geo_soft_score_A_mean": _mean(numeric_values["geo_soft_score_A"]),
        "geo_soft_score_B_mean": _mean(numeric_values["geo_soft_score_B"]),
        "geo_soft_score_C_mean": _mean(numeric_values["geo_soft_score_C"]),
        "geo_soft_score_D_mean": _mean(numeric_values["geo_soft_score_D"]),
    }

    return {
        "num_evaluated": num_evaluated,
        "num_correct": num_correct,
        "accuracy": num_correct / num_evaluated if num_evaluated else 0.0,
        "invalid_outputs": invalid,
        "hit_max_response_length_outputs": hit_max_count,
        "hit_max_response_length_rate": hit_max_count / num_evaluated if num_evaluated else 0.0,
        "response_token_len_mean": _mean(numeric_values["response_token_len"]),
        "response_token_len_max": max(numeric_values["response_token_len"]) if numeric_values["response_token_len"] else 0.0,
        "prediction_distribution": dict(sorted(prediction_distribution.items())),
        "ttrl_pseudo_label_distribution": dict(sorted(pseudo_label_distribution.items())),
        "accuracy_by_question_type": dict(sorted(accuracy_by_type.items())),
        "ttrv_metrics": ttrv_metrics,
        "ttrv_metrics_by_question_type": dict(sorted(vote_by_question_type.items())),
    }



def render_report(metrics: dict[str, Any], title: str) -> str:
    ttrv = metrics.get("ttrv_metrics", {})
    lines = [
        f"# {title}",
        "",
        "- Vote records: {}".format(metrics["num_evaluated"]),
        "- Correct vote records: {}".format(metrics["num_correct"]),
        "- Vote accuracy: {:.6f}".format(metrics["accuracy"]),
        "- Invalid vote outputs: {}".format(metrics["invalid_outputs"]),
        "- Hit max response length outputs: {}".format(metrics.get("hit_max_response_length_outputs", 0)),
        "- Mean response token length: {:.6f}".format(float(metrics.get("response_token_len_mean", 0.0))),
        "- Prompt groups: {}".format(int(ttrv.get("num_groups", 0))),
        "- Mean votes per group: {:.6f}".format(float(ttrv.get("mean_votes_per_group", 0.0))),
        "",
        "## TTRV-Style Vote Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in [
        "response_accuracy",
        "acc_at_1",
        "majority_accuracy",
        "pass_at_votes",
        "best_accuracy_at_votes",
        "worst_accuracy_at_votes",
        "vote_invalid_rate",
        "group_invalid_rate",
        "frequency_mean",
        "normalized_entropy_mean",
        "raw_entropy_mean",
        "ttrv_reward_mean",
        "reward_mean",
        "unique_pred_ratio_mean",
        "group_unknown_rate_mean",
        "ttrl_majority_count_mean",
        "ttrl_majority_ratio_mean",
        "ttrl_majority_tie_mean",
        "ttrl_max_majority_ratio_mean",
        "geo_harmony_hm_mean",
        "geo_original_majority_ratio_mean",
        "geo_original_majority_tie_mean",
        "geo_view_support_mean",
        "geo_skipped_rate",
        "geo_soft_selected_rate",
        "geo_soft_top_prob_mean",
        "geo_soft_known_count_mean",
        "geo_soft_gt_mass_mean",
        "geo_soft_view_support_mean",
    ]:
        if key in ttrv:
            value = ttrv[key]
            value_str = "n/a" if value is None else f"{float(value):.6f}"
            lines.append(f"| {key} | {value_str} |")

    lines.extend([
        "",
        "## Response Stats",
        "",
        "| metric | value |",
        "| --- | ---: |",
        "| hit_max_response_length_rate | {:.6f} |".format(float(metrics.get("hit_max_response_length_rate", 0.0))),
        "| response_token_len_mean | {:.6f} |".format(float(metrics.get("response_token_len_mean", 0.0))),
        "| response_token_len_max | {:.6f} |".format(float(metrics.get("response_token_len_max", 0.0))),
        "",
        "## Prediction Distribution",
        "",
        "| prediction | count |",
        "| --- | ---: |",
    ])
    for pred, count in metrics.get("prediction_distribution", {}).items():
        lines.append(f"| {pred} | {count} |")

    if metrics.get("ttrl_pseudo_label_distribution"):
        lines.extend([
            "",
            "## TTRL Pseudo Label Distribution",
            "",
            "| pseudo_label | count |",
            "| --- | ---: |",
        ])
        for label, count in metrics.get("ttrl_pseudo_label_distribution", {}).items():
            lines.append(f"| {label} | {count} |")

    lines.extend([
        "",
        "## Accuracy By Question Type",
        "",
        "| question_type | vote_records | correct | vote_accuracy | invalid_outputs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for qtype, row in metrics["accuracy_by_question_type"].items():
        lines.append(
            "| {} | {} | {} | {:.6f} | {} |".format(
                qtype,
                row["num_evaluated"],
                row["num_correct"],
                row["accuracy"],
                row["invalid_outputs"],
            )
        )

    vote_by_type = metrics.get("ttrv_metrics_by_question_type", {})
    if vote_by_type:
        lines.extend([
            "",
            "## TTRV Metrics By Question Type",
            "",
            "| question_type | groups | votes/group | response_acc | acc@1 | majority_acc | pass@votes | invalid_rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for qtype, row in vote_by_type.items():
            acc_at_1 = row.get("acc_at_1")
            acc_at_1_str = "n/a" if acc_at_1 is None else f"{float(acc_at_1):.6f}"
            lines.append(
                "| {} | {} | {:.6f} | {:.6f} | {} | {:.6f} | {:.6f} | {:.6f} |".format(
                    qtype,
                    row["num_groups"],
                    row["mean_votes_per_group"],
                    row["response_accuracy"],
                    acc_at_1_str,
                    row["majority_accuracy"],
                    row["pass_at_votes"],
                    row["invalid_rate"],
                )
            )
    lines.append("")
    return "\n".join(lines)

def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PhysX-MCQ validation prediction JSONL files.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--title", default="PhysX-MCQ TTRV Evaluation")
    args = parser.parse_args()

    records = []
    with args.predictions_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    metrics = summarize_prediction_records(records)
    metrics["predictions_jsonl"] = str(args.predictions_jsonl)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_report(metrics, args.title), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
