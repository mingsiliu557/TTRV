from collections import Counter
from typing import List
import math
import re
from decimal import Decimal, InvalidOperation
import numpy as np
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.direct_answer import extract_direct_answer
from verl.utils.reward_score.ttrl.auto_verify import auto_verify

CHOICE_LABELS = ("A", "B", "C", "D")
UNKNOWN_LABEL = "unknown"
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
DENSITY_REWARD_STYLES = {
    "density_peak_hard",
    "density_peak_soft",
    "density_peak_answer_entropy",
    "density_peak_density_entropy",
    "density_cluster_soft",
    "density_cluster_answer_entropy",
    "density_cluster_density_entropy",
}
NUMERIC_REWARD_STYLES = {
    "numeric_kernel_density",
    "numeric_trimmed_mean",
}


def _parse_choice_labels(choice_labels=None):
    if choice_labels is None:
        return CHOICE_LABELS
    if isinstance(choice_labels, str):
        text = choice_labels.strip().upper()
        if text in {"A-D", "ABCD"}:
            return CHOICE_LABELS
        if text in {"A-F", "ABCDEF"}:
            return ("A", "B", "C", "D", "E", "F")
        if "," not in text and re.fullmatch(r"[A-Z]+", text):
            return tuple(text)
        labels = tuple(label.strip().upper() for label in text.split(",") if label.strip())
    else:
        labels = tuple(str(label).strip().upper() for label in choice_labels if str(label).strip())
    if not labels:
        return CHOICE_LABELS
    if any(not re.fullmatch(r"[A-Z]", label) for label in labels):
        raise ValueError(f"Invalid answer_choice_labels={choice_labels}")
    return labels


def _choice_char_class(choice_labels):
    return "".join(re.escape(label) for label in _parse_choice_labels(choice_labels))


def _is_choice_group(task, ground_truth, choice_labels=None):
    if task in {"vqa_da", "ocr"}:
        return False
    if task == "gpqa":
        return True
    labels = _parse_choice_labels(choice_labels)
    return str(ground_truth).strip().upper() in labels


def _normalize_choice_answer(answer, choice_labels=None):
    labels = _parse_choice_labels(choice_labels)
    if answer is None:
        return UNKNOWN_LABEL

    answer = str(answer).strip()
    if not answer:
        return UNKNOWN_LABEL

    upper_answer = answer.upper()
    if upper_answer in labels:
        return upper_answer

    matches = re.findall(rf"\b([{_choice_char_class(labels)}])\b", upper_answer)
    if matches:
        return matches[-1]
    return UNKNOWN_LABEL


def _normalize_freeform_answer(answer):
    if answer is None:
        return UNKNOWN_LABEL
    answer = str(answer).strip()
    return answer if answer else UNKNOWN_LABEL


def _normalize_numeric_answer(answer):
    text = str(answer or "").replace(",", "").strip()
    match = NUMBER_RE.search(text)
    if not match:
        return UNKNOWN_LABEL
    try:
        value = Decimal(match.group(0))
    except InvalidOperation:
        return UNKNOWN_LABEL
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        normalized = "0"
    return normalized


def _numeric_value(answer):
    normalized = _normalize_numeric_answer(answer)
    if normalized == UNKNOWN_LABEL:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _numeric_smape_score_value(prediction, target):
    if prediction is None or target is None:
        return 0.0
    denom = abs(float(prediction)) + abs(float(target))
    if denom == 0:
        return 1.0 if float(prediction) == float(target) else 0.0
    return max(0.0, 1.0 - abs(float(prediction) - float(target)) / denom)


def _numeric_distance(left, right):
    denom = abs(float(left)) + abs(float(right))
    if denom == 0:
        return 0.0 if float(left) == float(right) else 1.0
    return abs(float(left) - float(right)) / denom


def _format_numeric_label(value):
    if value is None:
        return UNKNOWN_LABEL
    value = float(value)
    if math.isfinite(value) and abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.6g}"


def _metric_from_extra_info(extra_info):
    if isinstance(extra_info, (list, tuple)):
        for item in extra_info:
            if isinstance(item, dict) and item.get("metric"):
                return str(item.get("metric")).strip().lower()
        return ""
    if isinstance(extra_info, dict):
        return str(extra_info.get("metric", "")).strip().lower()
    return ""


def _use_choice_labels_for_group(task, ground_truth, choice_labels=None):
    return _is_choice_group(task, ground_truth, choice_labels=choice_labels) and _has_choice_ground_truth(
        ground_truth,
        choice_labels=choice_labels,
    )


def _has_choice_ground_truth(ground_truth, choice_labels=None):
    return _normalize_choice_answer(ground_truth, choice_labels=choice_labels) != UNKNOWN_LABEL


def _extract_prefixed_choice_answer(text, choice_labels=None):
    labels = _parse_choice_labels(choice_labels)
    label_chars = _choice_char_class(labels)
    text = str(text or "")
    trigger_matches = list(
        re.finditer(
            r"(?:answer|choice|option)\s*(?:is|:)\s*",
            text,
            flags=re.IGNORECASE,
        )
    )
    candidates = []
    if trigger_matches:
        candidates.append(text[trigger_matches[-1].end():])
    candidates.append(text)

    pattern = re.compile(
        rf"^\s*(?:[:：/\\\-\*\(\[]\s*)*([{label_chars}])"
        rf"(?:\s*[\.\)\]:：,;]|(?=\s|<|\Z))",
        flags=re.IGNORECASE,
    )
    for candidate in candidates:
        match = pattern.search(candidate)
        if match:
            return match.group(1).upper()
    return UNKNOWN_LABEL


def _extract_legacy_choice_answers(solutions, model_answers, choice_labels=None):
    parsed_answers = []
    for solution, model_answer in zip(solutions, model_answers):
        prefixed_answer = _extract_prefixed_choice_answer(solution, choice_labels=choice_labels)
        parsed_answers.append(prefixed_answer if prefixed_answer != UNKNOWN_LABEL else model_answer)
    return parsed_answers


def _extract_kv_answer(text, choice_labels=None):
    matches = re.findall(
        rf"(?:^|[\r\n])\s*Answer\s*:\s*([{_choice_char_class(choice_labels)}])\b",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not matches:
        return UNKNOWN_LABEL
    return matches[-1].upper()


def _extract_choice_answer(text, choice_labels=None):
    return _normalize_choice_answer(text, choice_labels=choice_labels)


def _extract_kv_evidence(text):
    matches = re.findall(
        r"(?:^|[\r\n])\s*Evidence\s*:\s*(.*?)(?=(?:[\r\n]\s*Answer\s*:)|\Z)",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        return ""
    return matches[-1].strip()


def _token_jaccard(left, right):
    left_tokens = set(re.findall(r"[A-Za-z0-9]+", str(left or "").lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9]+", str(right or "").lower()))
    if not left_tokens and not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _reference_rewards_from_answers(task, answers, ground_truth, choice_labels=None):
    if _is_choice_group(task, ground_truth, choice_labels=choice_labels):
        normalized_ground_truth = _normalize_choice_answer(ground_truth, choice_labels=choice_labels)
        return [
            1.0 if _normalize_choice_answer(answer, choice_labels=choice_labels) == normalized_ground_truth else 0.0
            for answer in answers
        ]
    rewards, _ = auto_verify(task, answers, [ground_truth] * len(answers))
    return rewards


def _normalized_entropy(counter, total):
    if total <= 1 or len(counter) <= 1:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log(probability)

    max_entropy = math.log(len(counter))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _choice_counter(answers, choice_labels=None):
    labels = _parse_choice_labels(choice_labels)
    return Counter(ans for ans in answers if ans in labels)


def _choice_distribution(counter, choice_labels=None):
    labels = _parse_choice_labels(choice_labels)
    total = sum(counter.values())
    if total <= 0:
        return {label: 0.0 for label in labels}
    return {label: counter.get(label, 0) / total for label in labels}


def _majority_label(counter):
    if not counter:
        return UNKNOWN_LABEL, 0
    return counter.most_common(1)[0]


def _extract_answers(task, solutions, ground_truth, extra_info=None, answer_parse_mode="legacy", choice_labels=None):
    if answer_parse_mode == "kv_answer":
        return [_extract_kv_answer(solution, choice_labels=choice_labels) for solution in solutions]
    if answer_parse_mode == "mathvista_choice":
        return [_extract_choice_answer(solution, choice_labels=choice_labels) for solution in solutions]
    if answer_parse_mode == "direct_answer_short":
        return [extract_direct_answer(solution) for solution in solutions]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    if answer_parse_mode == "legacy":
        if _is_choice_group(task, ground_truth, choice_labels=choice_labels) and _has_choice_ground_truth(
            ground_truth,
            choice_labels=choice_labels,
        ):
            return _extract_legacy_choice_answers(solutions, model_answers, choice_labels=choice_labels)
        return model_answers
    if answer_parse_mode != "canonical":
        raise ValueError(f"Unsupported answer_parse_mode: {answer_parse_mode}")
    if _is_choice_group(task, ground_truth, choice_labels=choice_labels):
        return [_normalize_choice_answer(answer, choice_labels=choice_labels) for answer in model_answers]
    return [answer if answer is not None and str(answer).strip() else UNKNOWN_LABEL for answer in model_answers]

def _verify_single_answer(task, answer, ground_truth, extra_info=None, choice_labels=None):
    if answer == UNKNOWN_LABEL:
        return 0.0
    if _is_choice_group(task, ground_truth, choice_labels=choice_labels):
        return float(
            _normalize_choice_answer(answer, choice_labels=choice_labels)
            == _normalize_choice_answer(ground_truth, choice_labels=choice_labels)
        )
    return float(auto_verify(task, [answer], [ground_truth], extra_info=extra_info)[0][0])


def _compute_reference_rewards(task, solutions, ground_truth, extra_info=None):
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    return true_rewards


def _frequency_entropy_rewards(model_answers, entropy_coef):
    counter = Counter(model_answers)
    total = len(model_answers)
    reward_p = [counter[ans] / total for ans in model_answers]
    normalized_entropy = _normalized_entropy(counter, total)
    return [(r * 1) - (entropy_coef * normalized_entropy) for r in reward_p], counter, normalized_entropy


def _valid_frequency_rewards(
    model_answers,
    unknown_reward=0.0,
    all_unknown_reward=0.0,
    choice_labels=None,
    use_choice_labels=True,
    numeric_valid=False,
):
    choice_labels = _parse_choice_labels(choice_labels)
    if use_choice_labels:
        canonical_answers = [_normalize_choice_answer(answer, choice_labels=choice_labels) for answer in model_answers]
        valid_counter = _choice_counter(canonical_answers, choice_labels=choice_labels)
        is_valid = lambda answer: answer in choice_labels
        valid_answer_mode = "choice"
    elif numeric_valid:
        canonical_answers = [_normalize_numeric_answer(answer) for answer in model_answers]
        valid_counter = Counter(answer for answer in canonical_answers if answer != UNKNOWN_LABEL)
        is_valid = lambda answer: answer != UNKNOWN_LABEL
        valid_answer_mode = "numeric"
    else:
        canonical_answers = [_normalize_freeform_answer(answer) for answer in model_answers]
        valid_counter = Counter(answer for answer in canonical_answers if answer != UNKNOWN_LABEL)
        is_valid = lambda answer: answer != UNKNOWN_LABEL
        valid_answer_mode = "freeform"
    valid_total = sum(valid_counter.values())

    if valid_total <= 0:
        rewards = [all_unknown_reward for _ in canonical_answers]
    else:
        rewards = [
            valid_counter[answer] / valid_total if is_valid(answer) else unknown_reward
            for answer in canonical_answers
        ]

    details = {
        "original_answers": canonical_answers,
        "valid_counter": dict(valid_counter),
        "valid_vote_ratio": valid_total / len(canonical_answers) if canonical_answers else 0.0,
        "valid_normalized_entropy": _normalized_entropy(valid_counter, valid_total),
        "valid_answer_mode": valid_answer_mode,
    }
    return rewards, valid_counter, details


def _valid_frequency_entropy_rewards(
    model_answers,
    entropy_coef,
    unknown_reward=0.0,
    all_unknown_reward=0.0,
    choice_labels=None,
    use_choice_labels=True,
    numeric_valid=False,
):
    rewards, valid_counter, details = _valid_frequency_rewards(
        model_answers,
        unknown_reward=unknown_reward,
        all_unknown_reward=all_unknown_reward,
        choice_labels=choice_labels,
        use_choice_labels=use_choice_labels,
        numeric_valid=numeric_valid,
    )
    normalized_entropy = details["valid_normalized_entropy"]
    canonical_answers = details["original_answers"]
    if sum(valid_counter.values()) <= 0:
        rewards_en = list(rewards)
    else:
        rewards_en = [reward - entropy_coef * normalized_entropy for reward in rewards]
    details["entropy_coef"] = entropy_coef
    return rewards_en, valid_counter, details


def _numeric_distribution_rewards(
    model_answers,
    mode="kernel_density",
    kernel_sigma=0.15,
    trim_ratio=0.2,
    unknown_reward=0.0,
    all_unknown_reward=0.0,
):
    canonical_answers = [_normalize_numeric_answer(answer) for answer in model_answers]
    values = [_numeric_value(answer) for answer in canonical_answers]
    valid_indices = [idx for idx, value in enumerate(values) if value is not None]
    valid_values = [values[idx] for idx in valid_indices]
    valid_counter = Counter(canonical_answers[idx] for idx in valid_indices)
    valid_total = len(valid_values)

    details = {
        "original_answers": canonical_answers,
        "valid_counter": dict(valid_counter),
        "valid_vote_ratio": valid_total / len(canonical_answers) if canonical_answers else 0.0,
        "valid_normalized_entropy": _normalized_entropy(valid_counter, valid_total),
        "valid_answer_mode": "numeric",
        "numeric_reward_mode": mode,
        "numeric_kernel_sigma": float(kernel_sigma),
        "numeric_trim_ratio": float(trim_ratio),
        "numeric_values": [float(value) if value is not None else None for value in values],
        "numeric_valid_count": valid_total,
        "numeric_unique_count": len(valid_counter),
        "numeric_top_ratio": (max(valid_counter.values()) / valid_total) if valid_total > 0 else 0.0,
    }

    if valid_total <= 0:
        rewards = [all_unknown_reward for _ in canonical_answers]
        details.update(
            {
                "numeric_pseudo_label": UNKNOWN_LABEL,
                "numeric_pseudo_value": None,
                "numeric_pseudo_support": 0,
                "numeric_density_max": 0.0,
                "numeric_density_margin": 0.0,
                "numeric_rewards": rewards,
            }
        )
        return rewards, valid_counter, details

    mode = str(mode).lower()
    if mode == "kernel_density":
        sigma = max(float(kernel_sigma), 1e-6)
        valid_densities = []
        for value_i in valid_values:
            weights = []
            for value_j in valid_values:
                distance = _numeric_distance(value_i, value_j)
                weights.append(math.exp(-((distance / sigma) ** 2) / 2.0))
            valid_densities.append(float(np.mean(weights)) if weights else 0.0)
        best_local_idx = int(np.argmax(valid_densities))
        pseudo_value = valid_values[best_local_idx]
        sorted_densities = sorted(valid_densities, reverse=True)
        density_max = sorted_densities[0] if sorted_densities else 0.0
        density_margin = (sorted_densities[0] - sorted_densities[1]) if len(sorted_densities) > 1 else density_max
        rewards = [float(unknown_reward) for _ in canonical_answers]
        for idx, density in zip(valid_indices, valid_densities):
            rewards[idx] = float(density)
        pseudo_support = sum(1 for value in valid_values if _numeric_distance(value, pseudo_value) <= sigma)
    elif mode == "trimmed_mean":
        trim_ratio = min(max(float(trim_ratio), 0.0), 0.45)
        sorted_values = sorted(valid_values)
        trim_n = int(len(sorted_values) * trim_ratio)
        if len(sorted_values) - 2 * trim_n <= 0:
            trim_n = 0
        trimmed_values = sorted_values[trim_n:len(sorted_values) - trim_n] if trim_n > 0 else sorted_values
        pseudo_value = float(np.mean(trimmed_values)) if trimmed_values else float(np.mean(sorted_values))
        rewards = [float(unknown_reward) for _ in canonical_answers]
        for idx, value in zip(valid_indices, valid_values):
            rewards[idx] = _numeric_smape_score_value(value, pseudo_value)
        valid_rewards = [rewards[idx] for idx in valid_indices]
        sorted_rewards = sorted(valid_rewards, reverse=True)
        density_max = sorted_rewards[0] if sorted_rewards else 0.0
        density_margin = (sorted_rewards[0] - sorted_rewards[1]) if len(sorted_rewards) > 1 else density_max
        pseudo_support = sum(1 for reward in valid_rewards if reward >= 0.95)
    else:
        raise ValueError(f"Unsupported numeric reward mode: {mode}")

    details.update(
        {
            "numeric_pseudo_label": _format_numeric_label(pseudo_value),
            "numeric_pseudo_value": float(pseudo_value),
            "numeric_pseudo_support": int(pseudo_support),
            "numeric_density_max": float(density_max),
            "numeric_density_margin": float(density_margin),
            "numeric_rewards": rewards,
        }
    )
    return rewards, valid_counter, details


def _entropy_temperature_rewards(
    model_answers,
    version="v3",
    tau0=0.25,
    gamma=1.0,
    lam=0.5,
    tau_min=0.05,
    eps=1e-8,
    choice_labels=None,
):
    choice_labels = _parse_choice_labels(choice_labels)
    canonical_answers = [_normalize_choice_answer(answer, choice_labels=choice_labels) for answer in model_answers]
    valid_counter = _choice_counter(canonical_answers, choice_labels=choice_labels)
    valid_total = sum(valid_counter.values())

    p_dist = {label: 0.0 for label in choice_labels}
    q_dist = {label: 0.0 for label in choice_labels}
    p_rewards = []
    q_rewards = []

    if valid_total <= 0:
        rewards = [0.0 for _ in canonical_answers]
        p_rewards = [0.0 for _ in canonical_answers]
        q_rewards = [0.0 for _ in canonical_answers]
        normalized_entropy = 0.0
        temperature = max(float(tau_min), float(tau0))
    else:
        p_dist = {label: valid_counter[label] / valid_total for label in choice_labels}
        normalized_entropy = _normalized_entropy(valid_counter, valid_total)
        temperature = max(float(tau_min), float(tau0) + float(gamma) * normalized_entropy)
        exp_scores = {
            label: math.exp(p_dist[label] / (temperature + eps))
            for label in choice_labels
        }
        exp_total = sum(exp_scores.values())
        q_dist = {
            label: exp_scores[label] / (exp_total + eps)
            for label in choice_labels
        }

        rewards = []
        version = str(version).lower()
        for answer in canonical_answers:
            if answer not in choice_labels:
                p_reward = 0.0
                q_reward = 0.0
                reward = 0.0
            else:
                p_reward = p_dist[answer]
                q_reward = q_dist[answer]
                if version == "v1":
                    reward = math.exp(p_reward / (temperature + eps))
                elif version == "v2":
                    reward = q_reward
                elif version == "v3":
                    reward = (1.0 - float(lam)) * p_reward + float(lam) * q_reward
                else:
                    raise ValueError(f"Unsupported entropy_temperature version: {version}")
            p_rewards.append(p_reward)
            q_rewards.append(q_reward)
            rewards.append(reward)

    sorted_p = sorted(((p_dist[label], label) for label in choice_labels), reverse=True)
    sorted_q = sorted(((q_dist[label], label) for label in choice_labels), reverse=True)
    if valid_total <= 0:
        top_label = UNKNOWN_LABEL
        second_label = UNKNOWN_LABEL
        top_reward = 0.0
        second_reward = 0.0
    else:
        top_label = sorted_p[0][1] if sorted_p else UNKNOWN_LABEL
        second_label = sorted_p[1][1] if len(sorted_p) > 1 else UNKNOWN_LABEL
        top_reward = sorted_q[0][0] if sorted_q else 0.0
        second_reward = sorted_q[1][0] if len(sorted_q) > 1 else 0.0
    details = {
        "original_answers": canonical_answers,
        "valid_counter": dict(valid_counter),
        "valid_vote_ratio": valid_total / len(canonical_answers) if canonical_answers else 0.0,
        "valid_normalized_entropy": normalized_entropy,
        "temperature": temperature,
        "temperature_tau0": float(tau0),
        "temperature_gamma": float(gamma),
        "temperature_lambda": float(lam),
        "temperature_tau_min": float(tau_min),
        "temperature_version": str(version).lower(),
        "valid_distribution_p": p_dist,
        "softmax_distribution_q": q_dist,
        "p_rewards": p_rewards,
        "q_rewards": q_rewards,
        "majority_label": top_label,
        "second_label": second_label,
        "reward_margin_top2": top_reward - second_reward,
        "majority_reward": q_dist.get(top_label, 0.0),
        "second_reward": q_dist.get(second_label, 0.0),
        "flip_opportunity_rate": float(
            valid_total > 0
            and len(sorted_p) > 1
            and sorted_p[0][0] > 0.0
            and (sorted_p[0][0] - sorted_p[1][0]) <= 1.0 / valid_total
        ),
    }
    return rewards, valid_counter, details


def _soft_pseudo_label_rewards(model_answers, gamma, unknown_reward, all_unknown_reward, choice_labels=None):
    choice_labels = _parse_choice_labels(choice_labels)
    valid_counter = Counter(ans for ans in model_answers if ans in choice_labels)
    denom = sum(valid_counter[label] ** gamma for label in choice_labels)

    if denom <= 0:
        rewards = [all_unknown_reward for _ in model_answers]
        soft_dist = {label: 0.0 for label in choice_labels}
    else:
        soft_dist = {label: (valid_counter[label] ** gamma) / denom for label in choice_labels}
        rewards = [
            soft_dist[answer] if answer in choice_labels else unknown_reward
            for answer in model_answers
        ]

    return rewards, valid_counter, soft_dist


def _majority_vote_rewards(task, solutions, estimated_label, extra_info=None):
    if estimated_label == UNKNOWN_LABEL:
        return [0.0 for _ in solutions]
    rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    return rewards


def _choice_majority_vote_rewards(model_answers, choice_labels=None):
    choice_counter = _choice_counter(model_answers, choice_labels=choice_labels)
    estimated_label, majority_count = _majority_label(choice_counter)
    if estimated_label == UNKNOWN_LABEL:
        rewards = [0.0 for _ in model_answers]
    else:
        rewards = [1.0 if answer == estimated_label else 0.0 for answer in model_answers]
    return rewards, {
        "choice_majority_label": estimated_label,
        "choice_majority_count": int(majority_count),
        "choice_counter": dict(choice_counter),
        "choice_valid_ratio": (
            sum(choice_counter.values()) / len(model_answers) if model_answers else 0.0
        ),
        "choice_normalized_entropy": _normalized_entropy(
            choice_counter, sum(choice_counter.values())
        ),
    }


def _vision_self_harmony_rewards(
    task,
    original_answers,
    transform_answers,
    ground_truth,
    extra_info=None,
    eps=1e-8,
):
    original_counter = _choice_counter(original_answers)
    transform_counter = _choice_counter(transform_answers)
    original_dist = _choice_distribution(original_counter)
    transform_dist = _choice_distribution(transform_counter)

    harmonic_scores = {}
    for label in CHOICE_LABELS:
        p_original = original_dist[label]
        p_transform = transform_dist[label]
        harmonic_scores[label] = (
            2.0 * p_original * p_transform / (p_original + p_transform + eps)
            if p_original > 0.0 or p_transform > 0.0
            else 0.0
        )

    ranked_labels = sorted(CHOICE_LABELS, key=lambda label: (-harmonic_scores[label], label))
    harmony_label = ranked_labels[0]
    harmony_score_max = harmonic_scores[harmony_label]
    harmony_score_second = harmonic_scores[ranked_labels[1]] if len(ranked_labels) > 1 else 0.0
    if harmony_score_max <= 0.0:
        harmony_label = UNKNOWN_LABEL

    rewards = [1.0 if answer == harmony_label else 0.0 for answer in original_answers]

    original_majority, original_majority_count = _majority_label(original_counter)
    transform_majority, transform_majority_count = _majority_label(transform_counter)
    valid_pair_count = min(len(original_answers), len(transform_answers))
    paired_agreement = (
        sum(1 for i in range(valid_pair_count) if original_answers[i] == transform_answers[i]) / valid_pair_count
        if valid_pair_count > 0
        else 0.0
    )
    tv_distance = 0.5 * sum(abs(original_dist[label] - transform_dist[label]) for label in CHOICE_LABELS)

    details = {
        "harmony_label": harmony_label,
        "harmonic_scores": harmonic_scores,
        "original_counter": dict(original_counter),
        "transform_counter": dict(transform_counter),
        "original_distribution": original_dist,
        "transform_distribution": transform_dist,
        "original_majority": original_majority,
        "transform_majority": transform_majority,
        "original_majority_count": int(original_majority_count),
        "transform_majority_count": int(transform_majority_count),
        "harmony_score_max": float(harmony_score_max),
        "harmony_score_margin": float(harmony_score_max - harmony_score_second),
        "paired_prediction_agreement": float(paired_agreement),
        "distribution_tv_distance": float(tv_distance),
        "original_entropy": _normalized_entropy(original_counter, sum(original_counter.values())),
        "transform_entropy": _normalized_entropy(transform_counter, sum(transform_counter.values())),
        "original_invalid_ratio": 1.0 - (sum(original_counter.values()) / len(original_answers) if original_answers else 0.0),
        "transform_invalid_ratio": 1.0 - (sum(transform_counter.values()) / len(transform_answers) if transform_answers else 0.0),
        "harmony_label_accuracy": _verify_single_answer(task, harmony_label, ground_truth, extra_info=extra_info),
        "original_majority_accuracy": _verify_single_answer(task, original_majority, ground_truth, extra_info=extra_info),
        "transform_majority_accuracy": _verify_single_answer(task, transform_majority, ground_truth, extra_info=extra_info),
        "branch_majority_agreement": float(original_majority == transform_majority and original_majority != UNKNOWN_LABEL),
    }
    return rewards, details


def _as_embedding_array(response_embeddings):
    embeddings = np.asarray(response_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"feature-center reward expects 2D embeddings, got shape={embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def _feature_center_rank(response_embeddings):
    embeddings = _as_embedding_array(response_embeddings)
    centroid = embeddings.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    distances = 1.0 - np.matmul(embeddings, centroid)
    ranked_indices = sorted(range(len(distances)), key=lambda idx: (float(distances[idx]), idx))
    min_distance = float(distances[ranked_indices[0]]) if ranked_indices else 0.0
    second_distance = float(distances[ranked_indices[1]]) if len(ranked_indices) > 1 else min_distance
    return ranked_indices, distances.astype(np.float32).tolist(), min_distance, float(second_distance - min_distance)


def _select_feature_center_candidate(model_answers, response_embeddings, choice_labels=None, use_choice_labels=True):
    choice_labels = _parse_choice_labels(choice_labels)
    ranked_indices, distances, min_distance, margin = _feature_center_rank(response_embeddings)
    raw_center_index = ranked_indices[0] if ranked_indices else -1
    selected_index = raw_center_index
    for idx in ranked_indices:
        is_valid = model_answers[idx] in choice_labels if use_choice_labels else model_answers[idx] != UNKNOWN_LABEL
        if is_valid:
            selected_index = idx
            break
    pseudo_label = model_answers[selected_index] if selected_index >= 0 else UNKNOWN_LABEL
    invalid = pseudo_label not in choice_labels if use_choice_labels else pseudo_label == UNKNOWN_LABEL
    if invalid:
        pseudo_label = UNKNOWN_LABEL
    return selected_index, raw_center_index, pseudo_label, distances, min_distance, margin


def _feature_center_hard_rewards(task, model_answers, response_embeddings, ground_truth, extra_info=None):
    selected_index, raw_center_index, pseudo_label, distances, min_distance, margin = _select_feature_center_candidate(
        model_answers, response_embeddings
    )
    if pseudo_label == UNKNOWN_LABEL:
        rewards = [0.0 for _ in model_answers]
    else:
        rewards = [1.0 if answer == pseudo_label else 0.0 for answer in model_answers]

    choice_counter = _choice_counter(model_answers)
    majority_label, majority_count = _majority_label(choice_counter)
    details = {
        "feature_center_label": pseudo_label,
        "feature_center_candidate_index": int(selected_index),
        "feature_center_raw_candidate_index": int(raw_center_index),
        "feature_center_distances": distances,
        "feature_center_min_distance": min_distance,
        "feature_center_margin": margin,
        "feature_center_valid_ratio": (
            sum(choice_counter.values()) / len(model_answers) if model_answers else 0.0
        ),
        "original_majority": majority_label,
        "original_majority_count": int(majority_count),
        "original_counter": dict(choice_counter),
        "feature_center_label_accuracy": _verify_single_answer(
            task, pseudo_label, ground_truth, extra_info=extra_info
        ),
        "original_majority_accuracy": _verify_single_answer(
            task, majority_label, ground_truth, extra_info=extra_info
        ),
        "pseudo_vs_majority_agreement": float(
            pseudo_label == majority_label and pseudo_label != UNKNOWN_LABEL
        ),
    }
    return rewards, details


def _density_temperature_from_mode(mode, fixed_temperature, t_min, t_max, answer_entropy, density_entropy):
    mode = str(mode)
    if mode == "fixed":
        return max(float(fixed_temperature), 1e-6)
    if mode == "answer_entropy":
        entropy = answer_entropy
    elif mode == "density_entropy":
        entropy = density_entropy
    else:
        raise ValueError(f"Unsupported density temperature mode: {mode}")
    return max(float(t_min), float(t_min) + (float(t_max) - float(t_min)) * float(entropy))


def _density_masses_from_embeddings(valid_embeddings, temperature, eps=1e-8):
    valid_count = int(valid_embeddings.shape[0])
    if valid_count <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), 0.0, 0.0
    if valid_count == 1:
        return (
            np.ones((1,), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            0.0,
            0.0,
        )

    sim = np.matmul(valid_embeddings, valid_embeddings.T).astype(np.float32)
    off_diag_mask = ~np.eye(valid_count, dtype=bool)
    off_diag_values = sim[off_diag_mask]
    sim_mean = float(np.mean(off_diag_values)) if off_diag_values.size else 0.0
    sim_std = float(np.std(off_diag_values)) if off_diag_values.size else 0.0

    masked_sim = sim.copy()
    np.fill_diagonal(masked_sim, -np.inf)
    global_max = float(np.max(off_diag_values)) if off_diag_values.size else 0.0
    weights = np.exp((masked_sim - global_max) / max(float(temperature), eps))
    weights[~np.isfinite(weights)] = 0.0
    np.fill_diagonal(weights, 0.0)
    density = weights.sum(axis=1) / max(valid_count - 1, 1)
    density_sum = float(density.sum())
    if density_sum <= eps:
        masses = np.full((valid_count,), 1.0 / valid_count, dtype=np.float32)
    else:
        masses = (density / density_sum).astype(np.float32)
    return masses, sim, sim_mean, sim_std


def _density_entropy(masses):
    masses = np.asarray(masses, dtype=np.float32)
    if masses.size <= 1:
        return 0.0
    entropy = 0.0
    for value in masses:
        probability = float(value)
        if probability > 0:
            entropy -= probability * math.log(probability)
    max_entropy = math.log(masses.size)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _density_advantage(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(values.std())
    return (values - float(values.mean())) / (std + eps)


def _safe_corr(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.size != right.size or left.size <= 1:
        return 0.0
    if float(left.std()) <= 1e-8 or float(right.std()) <= 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _density_peak_rewards(
    task,
    model_answers,
    response_embeddings,
    ground_truth,
    extra_info=None,
    choice_labels=None,
    reward_mode="hard",
    temperature_mode="fixed",
    t0=0.2,
    t_min=0.05,
    t_max=0.8,
    embedding_scope="response_mean_pool",
):
    choice_labels = _parse_choice_labels(choice_labels)
    use_choice_labels = _use_choice_labels_for_group(task, ground_truth, choice_labels=choice_labels)
    if use_choice_labels:
        canonical_answers = [_normalize_choice_answer(answer, choice_labels=choice_labels) for answer in model_answers]
        is_valid = lambda answer: answer in choice_labels
    else:
        canonical_answers = [_normalize_freeform_answer(answer) for answer in model_answers]
        is_valid = lambda answer: answer != UNKNOWN_LABEL
    embeddings = _as_embedding_array(response_embeddings)
    if len(canonical_answers) != embeddings.shape[0]:
        raise ValueError(f"density reward expects {len(canonical_answers)} embeddings, got {embeddings.shape[0]}")

    valid_indices = [idx for idx, answer in enumerate(canonical_answers) if is_valid(answer)]
    if use_choice_labels:
        valid_counter = _choice_counter(canonical_answers, choice_labels=choice_labels)
        cluster_labels = list(choice_labels)
    else:
        valid_counter = Counter(answer for answer in canonical_answers if answer != UNKNOWN_LABEL)
        cluster_labels = sorted(valid_counter)
    valid_total = sum(valid_counter.values())
    answer_entropy = _normalized_entropy(valid_counter, valid_total)

    full_masses = np.zeros((len(canonical_answers),), dtype=np.float32)
    similarity_to_peak = [0.0 for _ in canonical_answers]
    density_entropy_fixed = 0.0
    density_entropy_value = 0.0
    sim_mean = 0.0
    sim_std = 0.0
    temperature = max(float(t0), 1e-6)
    peak_index = -1
    peak_label = UNKNOWN_LABEL
    peak_mass = 0.0

    if valid_indices:
        valid_embeddings = embeddings[valid_indices]
        fixed_masses, _, _, _ = _density_masses_from_embeddings(valid_embeddings, max(float(t0), 1e-6))
        density_entropy_fixed = _density_entropy(fixed_masses)
        temperature = _density_temperature_from_mode(
            temperature_mode,
            fixed_temperature=t0,
            t_min=t_min,
            t_max=t_max,
            answer_entropy=answer_entropy,
            density_entropy=density_entropy_fixed,
        )
        masses, _, sim_mean, sim_std = _density_masses_from_embeddings(valid_embeddings, temperature)
        density_entropy_value = _density_entropy(masses)
        local_peak = int(np.argmax(masses))
        peak_index = int(valid_indices[local_peak])
        peak_label = canonical_answers[peak_index]
        peak_mass = float(masses[local_peak])
        for local_idx, original_idx in enumerate(valid_indices):
            full_masses[original_idx] = float(masses[local_idx])
        peak_embedding = embeddings[peak_index]
        similarity_to_peak = np.matmul(embeddings, peak_embedding).astype(np.float32).tolist()

    if peak_label == UNKNOWN_LABEL:
        hard_rewards = [0.0 for _ in canonical_answers]
    else:
        hard_rewards = [1.0 if answer == peak_label else 0.0 for answer in canonical_answers]
    soft_rewards = [float(full_masses[idx]) if is_valid(canonical_answers[idx]) else 0.0 for idx in range(len(canonical_answers))]
    cluster_masses = {label: 0.0 for label in cluster_labels}
    for idx, answer in enumerate(canonical_answers):
        if answer in cluster_masses:
            cluster_masses[answer] += float(full_masses[idx])
    cluster_soft_rewards = [
        float(cluster_masses[answer]) if answer in cluster_masses else 0.0 for answer in canonical_answers
    ]
    if reward_mode == "hard":
        rewards = hard_rewards
    elif reward_mode == "cluster_soft":
        rewards = cluster_soft_rewards
    else:
        rewards = soft_rewards

    majority_label, majority_count = _majority_label(valid_counter)
    centroid_index, _, centroid_label, centroid_distances, centroid_min_distance, centroid_margin = _select_feature_center_candidate(
        canonical_answers, response_embeddings, choice_labels=choice_labels, use_choice_labels=use_choice_labels
    )
    freq_rewards, _, _ = _valid_frequency_rewards(
        canonical_answers,
        unknown_reward=0.0,
        all_unknown_reward=0.0,
        choice_labels=choice_labels,
        use_choice_labels=use_choice_labels,
    )
    adv_density = _density_advantage(rewards)
    adv_freq = _density_advantage(freq_rewards)

    details = {
        "original_answers": canonical_answers,
        "density_rewards": [float(value) for value in rewards],
        "density_hard_rewards": [float(value) for value in hard_rewards],
        "density_soft_rewards": [float(value) for value in soft_rewards],
        "density_cluster_rewards": [float(value) for value in cluster_soft_rewards],
        "density_cluster_masses": {label: float(value) for label, value in cluster_masses.items()},
        "density_masses": [float(value) for value in full_masses.tolist()],
        "similarity_to_peak": [float(value) for value in similarity_to_peak],
        "density_peak_label": peak_label,
        "density_peak_sample_index": int(peak_index),
        "density_peak_mass": float(peak_mass),
        "density_temperature": float(temperature),
        "density_temperature_mode": str(temperature_mode),
        "density_reward_mode": str(reward_mode),
        "density_embedding_scope": str(embedding_scope),
        "density_valid_counter": dict(valid_counter),
        "density_valid_ratio": valid_total / len(canonical_answers) if canonical_answers else 0.0,
        "density_answer_entropy": float(answer_entropy),
        "density_density_entropy": float(density_entropy_value),
        "density_density_entropy_fixed": float(density_entropy_fixed),
        "density_sim_mean": float(sim_mean),
        "density_sim_std": float(sim_std),
        "density_peak_label_accuracy": _verify_single_answer(
            task, peak_label, ground_truth, extra_info=extra_info, choice_labels=choice_labels
        ),
        "original_majority": majority_label,
        "original_majority_count": int(majority_count),
        "original_counter": dict(valid_counter),
        "original_majority_accuracy": _verify_single_answer(
            task, majority_label, ground_truth, extra_info=extra_info, choice_labels=choice_labels
        ),
        "arithmetic_centroid_label": centroid_label,
        "arithmetic_centroid_sample_index": int(centroid_index),
        "arithmetic_centroid_distances": centroid_distances,
        "arithmetic_centroid_min_distance": float(centroid_min_distance),
        "arithmetic_centroid_margin": float(centroid_margin),
        "arithmetic_centroid_label_accuracy": _verify_single_answer(
            task, centroid_label, ground_truth, extra_info=extra_info, choice_labels=choice_labels
        ),
        "density_vs_majority_agreement": float(peak_label == majority_label and peak_label != UNKNOWN_LABEL),
        "density_vs_centroid_agreement": float(peak_label == centroid_label and peak_label != UNKNOWN_LABEL),
        "corr_adv_density_freq": _safe_corr(adv_density, adv_freq),
        "mean_abs_diff_adv_density_freq": float(np.mean(np.abs(adv_density - adv_freq))) if len(canonical_answers) else 0.0,
    }
    return rewards, details


def _feature_center_hsr_rewards(
    task,
    solutions,
    model_answers,
    response_embeddings,
    ground_truth,
    extra_info=None,
    alpha=0.5,
    beta=0.2,
):
    if alpha < 0 or beta < 0 or alpha + beta > 1:
        raise ValueError(f"Invalid HSR weights: alpha={alpha}, beta={beta}")

    selected_index, raw_center_index, pseudo_label, distances, min_distance, margin = _select_feature_center_candidate(
        model_answers, response_embeddings
    )
    embeddings = _as_embedding_array(response_embeddings)
    pseudo_embedding = embeddings[selected_index] if selected_index >= 0 else np.zeros(embeddings.shape[1], dtype=np.float32)
    pseudo_evidence = _extract_kv_evidence(solutions[selected_index]) if selected_index >= 0 else ""

    hard_rewards = []
    jaccard_rewards = []
    embedding_rewards = []
    rewards = []
    for solution, answer, embedding in zip(solutions, model_answers, embeddings):
        if answer not in CHOICE_LABELS:
            hard_rewards.append(0.0)
            jaccard_rewards.append(0.0)
            embedding_rewards.append(0.0)
            rewards.append(0.0)
            continue
        hard = 1.0 if pseudo_label != UNKNOWN_LABEL and answer == pseudo_label else 0.0
        jaccard = _token_jaccard(_extract_kv_evidence(solution), pseudo_evidence)
        cosine = float(np.dot(embedding, pseudo_embedding))
        embedding_reward = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        reward = alpha * hard + beta * jaccard + (1.0 - alpha - beta) * embedding_reward
        hard_rewards.append(float(hard))
        jaccard_rewards.append(float(jaccard))
        embedding_rewards.append(float(embedding_reward))
        rewards.append(float(reward))

    choice_counter = _choice_counter(model_answers)
    majority_label, majority_count = _majority_label(choice_counter)
    details = {
        "feature_center_label": pseudo_label,
        "feature_center_candidate_index": int(selected_index),
        "feature_center_raw_candidate_index": int(raw_center_index),
        "feature_center_distances": distances,
        "feature_center_min_distance": min_distance,
        "feature_center_margin": margin,
        "feature_center_valid_ratio": (
            sum(choice_counter.values()) / len(model_answers) if model_answers else 0.0
        ),
        "original_majority": majority_label,
        "original_majority_count": int(majority_count),
        "original_counter": dict(choice_counter),
        "feature_center_label_accuracy": _verify_single_answer(
            task, pseudo_label, ground_truth, extra_info=extra_info
        ),
        "original_majority_accuracy": _verify_single_answer(
            task, majority_label, ground_truth, extra_info=extra_info
        ),
        "pseudo_vs_majority_agreement": float(
            pseudo_label == majority_label and pseudo_label != UNKNOWN_LABEL
        ),
        "pseudo_response_raw": solutions[selected_index] if selected_index >= 0 else "",
        "pseudo_evidence": pseudo_evidence,
        "hsr_alpha": float(alpha),
        "hsr_beta": float(beta),
        "hsr_hard_rewards": hard_rewards,
        "hsr_jaccard_rewards": jaccard_rewards,
        "hsr_embedding_rewards": embedding_rewards,
        "hsr_hard_mean": float(np.mean(hard_rewards)) if hard_rewards else 0.0,
        "hsr_jaccard_mean": float(np.mean(jaccard_rewards)) if jaccard_rewards else 0.0,
        "hsr_embedding_mean": float(np.mean(embedding_rewards)) if embedding_rewards else 0.0,
    }
    return rewards, details


def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    task="math",
    extra_info=None,
    reward_style="frequency_entropy",
    soft_label_gamma=2.0,
    unknown_reward=0.0,
    all_unknown_reward=0.0,
    entropy_coef=0.75,
    answer_parse_mode="legacy",
    transform_solutions=None,
    transform_extra_info=None,
    response_embeddings=None,
    feature_center_hsr_alpha=0.5,
    feature_center_hsr_beta=0.2,
    entropy_temperature_version="v3",
    entropy_temperature_tau0=0.25,
    entropy_temperature_gamma=1.0,
    entropy_temperature_lambda=0.5,
    entropy_temperature_tau_min=0.05,
    density_temperature_t0=0.2,
    density_temperature_t_min=0.05,
    density_temperature_t_max=0.8,
    density_embedding_scope="response_mean_pool",
    numeric_kernel_sigma=0.15,
    numeric_trim_ratio=0.2,
    answer_choice_labels=None,
    return_details=False,
):
    choice_labels = _parse_choice_labels(answer_choice_labels)
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]
    use_choice_labels = _use_choice_labels_for_group(task, ground_truth, choice_labels=choice_labels)
    use_numeric_valid = (not use_choice_labels) and _metric_from_extra_info(extra_info) == "smape"

    model_answers = _extract_answers(
        task,
        solutions,
        ground_truth,
        extra_info=extra_info,
        answer_parse_mode=answer_parse_mode,
        choice_labels=choice_labels,
    )
    counter = Counter(model_answers)
    total = len(model_answers)

    if reward_style in {"frequency_valid_only", "frequency_valid_entropy"}:
        if use_choice_labels:
            majority_answers = [_normalize_choice_answer(answer, choice_labels=choice_labels) for answer in model_answers]
            valid_counter_for_majority = _choice_counter(majority_answers, choice_labels=choice_labels)
        elif use_numeric_valid:
            majority_answers = [_normalize_numeric_answer(answer) for answer in model_answers]
            valid_counter_for_majority = Counter(ans for ans in majority_answers if ans != UNKNOWN_LABEL)
        else:
            majority_answers = [_normalize_freeform_answer(answer) for answer in model_answers]
            valid_counter_for_majority = Counter(ans for ans in majority_answers if ans != UNKNOWN_LABEL)
        counter = Counter(majority_answers)
    else:
        valid_counter_for_majority = Counter(ans for ans in model_answers if ans != UNKNOWN_LABEL)
    if valid_counter_for_majority:
        estimated_label, majority_count = valid_counter_for_majority.most_common(1)[0]
    else:
        estimated_label, majority_count = UNKNOWN_LABEL, 0

    rewards = _majority_vote_rewards(task, solutions, estimated_label, extra_info=extra_info)

    details = {}

    if reward_style == "frequency_entropy":
        rewards_en, _, normalized_entropy = _frequency_entropy_rewards(model_answers, entropy_coef=entropy_coef)
        actual_reward_mean_key = "frequency_entropy_reward"
    elif reward_style == "frequency_valid_only":
        rewards_en, valid_counter, details = _valid_frequency_rewards(
            model_answers,
            unknown_reward=unknown_reward,
            all_unknown_reward=all_unknown_reward,
            choice_labels=choice_labels,
            use_choice_labels=use_choice_labels,
            numeric_valid=use_numeric_valid,
        )
        model_answers = details["original_answers"]
        counter = Counter(model_answers)
        total = len(model_answers)
        rewards = [1.0 if answer == estimated_label and estimated_label != UNKNOWN_LABEL else 0.0 for answer in model_answers]
        normalized_entropy = details["valid_normalized_entropy"]
        actual_reward_mean_key = "frequency_valid_only_reward"
    elif reward_style == "frequency_valid_entropy":
        rewards_en, valid_counter, details = _valid_frequency_entropy_rewards(
            model_answers,
            entropy_coef=entropy_coef,
            unknown_reward=unknown_reward,
            all_unknown_reward=all_unknown_reward,
            choice_labels=choice_labels,
            use_choice_labels=use_choice_labels,
            numeric_valid=use_numeric_valid,
        )
        model_answers = details["original_answers"]
        counter = Counter(model_answers)
        total = len(model_answers)
        rewards = [1.0 if answer == estimated_label and estimated_label != UNKNOWN_LABEL else 0.0 for answer in model_answers]
        normalized_entropy = details["valid_normalized_entropy"]
        actual_reward_mean_key = "frequency_valid_entropy_reward"
    elif reward_style == "entropy_temperature_frequency":
        rewards_en, valid_counter, details = _entropy_temperature_rewards(
            model_answers,
            version=entropy_temperature_version,
            tau0=entropy_temperature_tau0,
            gamma=entropy_temperature_gamma,
            lam=entropy_temperature_lambda,
            tau_min=entropy_temperature_tau_min,
            choice_labels=choice_labels,
        )
        model_answers = details["original_answers"]
        counter = Counter(model_answers)
        total = len(model_answers)
        if valid_counter:
            estimated_label, majority_count = valid_counter.most_common(1)[0]
        else:
            estimated_label, majority_count = UNKNOWN_LABEL, 0
        rewards = [
            1.0 if answer == estimated_label and estimated_label != UNKNOWN_LABEL else 0.0
            for answer in model_answers
        ]
        normalized_entropy = details["valid_normalized_entropy"]
        actual_reward_mean_key = "entropy_temperature_frequency_reward"
    elif reward_style in NUMERIC_REWARD_STYLES:
        if not use_numeric_valid:
            raise ValueError(f"{reward_style} is only supported for numeric/smape free-form groups")
        numeric_mode = "kernel_density" if reward_style == "numeric_kernel_density" else "trimmed_mean"
        rewards_en, valid_counter, details = _numeric_distribution_rewards(
            model_answers,
            mode=numeric_mode,
            kernel_sigma=numeric_kernel_sigma,
            trim_ratio=numeric_trim_ratio,
            unknown_reward=unknown_reward,
            all_unknown_reward=all_unknown_reward,
        )
        model_answers = details["original_answers"]
        counter = Counter(model_answers)
        total = len(model_answers)
        estimated_label = details["numeric_pseudo_label"]
        majority_count = details["numeric_pseudo_support"]
        rewards = list(rewards_en)
        normalized_entropy = details["valid_normalized_entropy"]
        actual_reward_mean_key = f"{reward_style}_reward"
    elif reward_style == "soft_pseudo_label":
        rewards_en, valid_counter, _ = _soft_pseudo_label_rewards(
            model_answers,
            gamma=soft_label_gamma,
            unknown_reward=unknown_reward,
            all_unknown_reward=all_unknown_reward,
            choice_labels=choice_labels,
        )
        normalized_entropy = _normalized_entropy(Counter(ans for ans in model_answers if ans in choice_labels), sum(valid_counter.values()))
        actual_reward_mean_key = "soft_pseudo_label_reward"
    elif reward_style == "majority_vote":
        rewards_en = rewards
        normalized_entropy = _normalized_entropy(counter, total)
        actual_reward_mean_key = "majority_vote_reward"
    elif reward_style == "choice_majority_vote":
        rewards_en, details = _choice_majority_vote_rewards(model_answers, choice_labels=choice_labels)
        rewards = rewards_en
        estimated_label = details["choice_majority_label"]
        majority_count = details["choice_majority_count"]
        normalized_entropy = details["choice_normalized_entropy"]
        actual_reward_mean_key = "choice_majority_vote_reward"
    elif reward_style == "vision_self_harmony":
        if transform_solutions is None:
            raise ValueError("vision_self_harmony requires transform_solutions")
        transform_answers = _extract_answers(
            task,
            transform_solutions,
            ground_truth,
            extra_info=transform_extra_info if transform_extra_info is not None else extra_info,
            answer_parse_mode=answer_parse_mode,
            choice_labels=choice_labels,
        )
        rewards_en, details = _vision_self_harmony_rewards(
            task,
            original_answers=model_answers,
            transform_answers=transform_answers,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        normalized_entropy = details["original_entropy"]
        actual_reward_mean_key = "vision_self_harmony_reward"
    elif reward_style in DENSITY_REWARD_STYLES:
        if response_embeddings is None:
            raise ValueError(f"{reward_style} requires response_embeddings")
        if reward_style == "density_peak_soft":
            density_reward_mode = "soft"
            density_temperature_mode = "fixed"
        elif reward_style == "density_cluster_soft":
            density_reward_mode = "cluster_soft"
            density_temperature_mode = "fixed"
        elif reward_style == "density_peak_answer_entropy":
            density_reward_mode = "hard"
            density_temperature_mode = "answer_entropy"
        elif reward_style == "density_cluster_answer_entropy":
            density_reward_mode = "cluster_soft"
            density_temperature_mode = "answer_entropy"
        elif reward_style == "density_peak_density_entropy":
            density_reward_mode = "hard"
            density_temperature_mode = "density_entropy"
        elif reward_style == "density_cluster_density_entropy":
            density_reward_mode = "cluster_soft"
            density_temperature_mode = "density_entropy"
        else:
            density_reward_mode = "hard"
            density_temperature_mode = "fixed"
        rewards_en, details = _density_peak_rewards(
            task,
            model_answers=model_answers,
            response_embeddings=response_embeddings,
            ground_truth=ground_truth,
            extra_info=extra_info,
            choice_labels=choice_labels,
            reward_mode=density_reward_mode,
            temperature_mode=density_temperature_mode,
            t0=density_temperature_t0,
            t_min=density_temperature_t_min,
            t_max=density_temperature_t_max,
            embedding_scope=density_embedding_scope,
        )
        model_answers = details["original_answers"]
        counter = Counter(model_answers)
        total = len(model_answers)
        rewards = details["density_hard_rewards"]
        estimated_label = details["density_peak_label"]
        majority_count = sum(1 for answer in model_answers if answer == estimated_label)
        choice_counter = _choice_counter(model_answers, choice_labels=choice_labels)
        normalized_entropy = details["density_answer_entropy"]
        actual_reward_mean_key = f"{reward_style}_reward"
    elif reward_style == "feature_center_hard":
        if response_embeddings is None:
            raise ValueError("feature_center_hard requires response_embeddings")
        rewards_en, details = _feature_center_hard_rewards(
            task,
            model_answers=model_answers,
            response_embeddings=response_embeddings,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        rewards = rewards_en
        estimated_label = details["feature_center_label"]
        majority_count = sum(1 for answer in model_answers if answer == estimated_label)
        choice_counter = _choice_counter(model_answers)
        normalized_entropy = _normalized_entropy(choice_counter, sum(choice_counter.values()))
        actual_reward_mean_key = "feature_center_hard_reward"
    elif reward_style == "feature_center_hsr":
        if response_embeddings is None:
            raise ValueError("feature_center_hsr requires response_embeddings")
        rewards_en, details = _feature_center_hsr_rewards(
            task,
            solutions=solutions,
            model_answers=model_answers,
            response_embeddings=response_embeddings,
            ground_truth=ground_truth,
            extra_info=extra_info,
            alpha=feature_center_hsr_alpha,
            beta=feature_center_hsr_beta,
        )
        rewards = rewards_en
        estimated_label = details["feature_center_label"]
        majority_count = sum(1 for answer in model_answers if answer == estimated_label)
        choice_counter = _choice_counter(model_answers)
        normalized_entropy = _normalized_entropy(choice_counter, sum(choice_counter.values()))
        actual_reward_mean_key = "feature_center_hsr_reward"
    else:
        raise ValueError(f"Unsupported TTRL reward_style: {reward_style}")

    hit_rate = _verify_single_answer(task, estimated_label, ground_truth, extra_info=extra_info, choice_labels=choice_labels)
    majority_ratio = majority_count / len(solutions)
    

    if _is_choice_group(task, ground_truth, choice_labels=choice_labels) and _has_choice_ground_truth(
        ground_truth,
        choice_labels=choice_labels,
    ):
        true_rewards = _reference_rewards_from_answers(task, model_answers, ground_truth, choice_labels=choice_labels)
    else:
        true_rewards = _compute_reference_rewards(task, solutions, ground_truth, extra_info=extra_info)
    
    rewards_hit_rate = 0
    for reward, true_reward in zip(rewards, true_rewards):
        if reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_ratio,
        "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        "majority_voting_reward": sum(rewards) / len(rewards),
        "normalized_entropy": normalized_entropy,
        "unknown_ratio": counter.get(UNKNOWN_LABEL, 0) / len(model_answers),
        actual_reward_mean_key: sum(rewards_en) / len(rewards_en),
        f"pass@{len(solutions)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
    }
    if reward_style == "soft_pseudo_label":
        ttrl_metrics["valid_vote_ratio"] = sum(valid_counter.values()) / len(model_answers)
        ttrl_metrics["soft_label_gamma"] = soft_label_gamma
    if reward_style in {"frequency_valid_only", "frequency_valid_entropy"}:
        ttrl_metrics["valid_vote_ratio"] = details["valid_vote_ratio"]
        ttrl_metrics["valid_normalized_entropy"] = details["valid_normalized_entropy"]
        if details.get("valid_answer_mode") == "numeric":
            ttrl_metrics["numeric_valid_ratio"] = details["valid_vote_ratio"]
        if reward_style == "frequency_valid_entropy":
            ttrl_metrics["valid_entropy_coef"] = details["entropy_coef"]
    if reward_style in NUMERIC_REWARD_STYLES:
        true_reward_pairs = list(zip(rewards_en, true_rewards))
        reward_values = [float(reward) for reward in rewards_en]
        true_values = [float(true_reward) for true_reward in true_rewards]
        correctish_rewards = [float(reward) for reward, true_reward in true_reward_pairs if float(true_reward) >= 0.9]
        weak_rewards = [float(reward) for reward, true_reward in true_reward_pairs if float(true_reward) < 0.9]
        ttrl_metrics["numeric_valid_ratio"] = details["valid_vote_ratio"]
        ttrl_metrics["numeric_valid_count"] = details["numeric_valid_count"]
        ttrl_metrics["numeric_unique_count"] = details["numeric_unique_count"]
        ttrl_metrics["numeric_top_ratio"] = details["numeric_top_ratio"]
        ttrl_metrics["numeric_pseudo_label_score"] = hit_rate
        ttrl_metrics["numeric_pseudo_support_ratio"] = details["numeric_pseudo_support"] / len(model_answers) if model_answers else 0.0
        ttrl_metrics["numeric_density_max"] = details["numeric_density_max"]
        ttrl_metrics["numeric_density_margin"] = details["numeric_density_margin"]
        ttrl_metrics["numeric_reward_std"] = float(np.std(reward_values)) if reward_values else 0.0
        ttrl_metrics["numeric_candidate_score_mean"] = float(np.mean(true_values)) if true_values else 0.0
        ttrl_metrics["reward_correctish_mean"] = float(np.mean(correctish_rewards)) if correctish_rewards else 0.0
        ttrl_metrics["reward_weak_mean"] = float(np.mean(weak_rewards)) if weak_rewards else 0.0
        ttrl_metrics["reward_correctish_minus_weak"] = (
            ttrl_metrics["reward_correctish_mean"] - ttrl_metrics["reward_weak_mean"]
        )
        ttrl_metrics["invalid_ratio"] = 1.0 - details["valid_vote_ratio"]
    if reward_style == "entropy_temperature_frequency":
        true_reward_pairs = list(zip(rewards_en, true_rewards))
        correct_rewards = [float(reward) for reward, true_reward in true_reward_pairs if true_reward]
        wrong_rewards = [float(reward) for reward, true_reward in true_reward_pairs if not true_reward]
        ttrl_metrics["temp_H"] = details["valid_normalized_entropy"]
        ttrl_metrics["temp_T"] = details["temperature"]
        ttrl_metrics["temp_valid_ratio"] = details["valid_vote_ratio"]
        ttrl_metrics["temp_reward_margin_top2"] = details["reward_margin_top2"]
        ttrl_metrics["temp_majority_reward"] = details["majority_reward"]
        ttrl_metrics["temp_second_reward"] = details["second_reward"]
        ttrl_metrics["temp_label_accuracy"] = hit_rate
        ttrl_metrics["temp_flip_opportunity_rate"] = details["flip_opportunity_rate"]
        ttrl_metrics["temp_oracle_rescue_rate"] = float(
            _normalize_choice_answer(ground_truth, choice_labels=choice_labels) == details["second_label"]
            and details["majority_label"] != details["second_label"]
        )
        ttrl_metrics["reward_correct_mean"] = float(np.mean(correct_rewards)) if correct_rewards else 0.0
        ttrl_metrics["reward_wrong_mean"] = float(np.mean(wrong_rewards)) if wrong_rewards else 0.0
        ttrl_metrics["reward_correct_minus_wrong"] = (
            ttrl_metrics["reward_correct_mean"] - ttrl_metrics["reward_wrong_mean"]
        )
    if reward_style == "choice_majority_vote":
        ttrl_metrics["choice_valid_ratio"] = details["choice_valid_ratio"]
    if reward_style == "vision_self_harmony":
        for key in [
            "harmony_label_accuracy",
            "original_majority_accuracy",
            "transform_majority_accuracy",
            "branch_majority_agreement",
            "paired_prediction_agreement",
            "original_entropy",
            "transform_entropy",
            "distribution_tv_distance",
            "harmony_score_max",
            "harmony_score_margin",
            "original_invalid_ratio",
            "transform_invalid_ratio",
        ]:
            ttrl_metrics[key] = details[key]
    if reward_style in DENSITY_REWARD_STYLES:
        true_reward_pairs = list(zip(rewards_en, true_rewards))
        correct_rewards = [float(reward) for reward, true_reward in true_reward_pairs if true_reward]
        wrong_rewards = [float(reward) for reward, true_reward in true_reward_pairs if not true_reward]
        correct_mean = float(np.mean(correct_rewards)) if correct_rewards else 0.0
        wrong_mean = float(np.mean(wrong_rewards)) if wrong_rewards else 0.0
        for key in [
            "density_peak_label_accuracy",
            "original_majority_accuracy",
            "arithmetic_centroid_label_accuracy",
            "density_valid_ratio",
            "density_peak_mass",
            "density_temperature",
            "density_answer_entropy",
            "density_density_entropy",
            "density_sim_mean",
            "density_sim_std",
            "density_vs_majority_agreement",
            "density_vs_centroid_agreement",
            "corr_adv_density_freq",
            "mean_abs_diff_adv_density_freq",
        ]:
            ttrl_metrics[key] = details[key]
        ttrl_metrics["density_reward_std"] = float(np.std(rewards_en)) if rewards_en else 0.0
        ttrl_metrics["reward_correct_mean"] = correct_mean
        ttrl_metrics["reward_wrong_mean"] = wrong_mean
        ttrl_metrics["reward_correct_minus_wrong"] = correct_mean - wrong_mean
        ttrl_metrics["invalid_ratio"] = 1.0 - details["density_valid_ratio"]
    if reward_style in {"feature_center_hard", "feature_center_hsr"}:
        true_reward_pairs = list(zip(rewards_en, true_rewards))
        correct_rewards = [float(reward) for reward, true_reward in true_reward_pairs if true_reward]
        wrong_rewards = [float(reward) for reward, true_reward in true_reward_pairs if not true_reward]
        correct_mean = float(np.mean(correct_rewards)) if correct_rewards else 0.0
        wrong_mean = float(np.mean(wrong_rewards)) if wrong_rewards else 0.0
        for key in [
            "feature_center_label_accuracy",
            "feature_center_valid_ratio",
            "feature_center_min_distance",
            "feature_center_margin",
            "original_majority_accuracy",
            "pseudo_vs_majority_agreement",
        ]:
            ttrl_metrics[key] = details[key]
        ttrl_metrics["reward_correct_mean"] = correct_mean
        ttrl_metrics["reward_wrong_mean"] = wrong_mean
        ttrl_metrics["reward_correct_minus_wrong"] = correct_mean - wrong_mean
        ttrl_metrics["invalid_ratio"] = 1.0 - details["feature_center_valid_ratio"]
        if reward_style == "feature_center_hsr":
            for key in ["hsr_hard_mean", "hsr_jaccard_mean", "hsr_embedding_mean"]:
                ttrl_metrics[key] = details[key]
    if return_details:
        details["original_answers"] = model_answers
        details["transform_answers"] = transform_answers if reward_style == "vision_self_harmony" else []
        details["evidences"] = [_extract_kv_evidence(solution) for solution in solutions]
        return rewards_en, ttrl_metrics, details
    return rewards_en, ttrl_metrics

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math",
    extra_info=None,
    answer_parse_mode="legacy",
    answer_choice_labels=None,
):
    choice_labels = _parse_choice_labels(answer_choice_labels)
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = _extract_answers(
        task,
        solutions,
        ground_truth,
        extra_info=extra_info,
        answer_parse_mode=answer_parse_mode,
        choice_labels=choice_labels,
    )

    if _is_choice_group(task, ground_truth, choice_labels=choice_labels) and _has_choice_ground_truth(
        ground_truth,
        choice_labels=choice_labels,
    ):
        true_rewards = _reference_rewards_from_answers(task, model_answers, ground_truth, choice_labels=choice_labels)
    else:
        true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)

    rewards_hit_rate = sum(
        1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)
    ) / len(pred_rewards)

    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics
