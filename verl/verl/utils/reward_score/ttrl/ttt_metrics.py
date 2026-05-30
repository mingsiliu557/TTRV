from collections import Counter
from typing import List
import math
import re
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify

CHOICE_LABELS = ("A", "B", "C", "D")
UNKNOWN_LABEL = "unknown"


def _is_choice_group(task, ground_truth):
    if task == "gpqa":
        return True
    return str(ground_truth).strip().upper() in CHOICE_LABELS


def _normalize_choice_answer(answer):
    if answer is None:
        return UNKNOWN_LABEL

    answer = str(answer).strip()
    if not answer:
        return UNKNOWN_LABEL

    upper_answer = answer.upper()
    if upper_answer in CHOICE_LABELS:
        return upper_answer

    matches = re.findall(r"\b([A-D])\b", upper_answer)
    if matches:
        return matches[-1]
    return UNKNOWN_LABEL


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


def _choice_counter(answers):
    return Counter(ans for ans in answers if ans in CHOICE_LABELS)


def _choice_distribution(counter):
    total = sum(counter.values())
    if total <= 0:
        return {label: 0.0 for label in CHOICE_LABELS}
    return {label: counter.get(label, 0) / total for label in CHOICE_LABELS}


def _majority_label(counter):
    if not counter:
        return UNKNOWN_LABEL, 0
    return counter.most_common(1)[0]


def _extract_answers(task, solutions, ground_truth, extra_info=None, answer_parse_mode="legacy"):
    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    if answer_parse_mode == "legacy":
        return model_answers
    if answer_parse_mode != "canonical":
        raise ValueError(f"Unsupported answer_parse_mode: {answer_parse_mode}")
    if _is_choice_group(task, ground_truth):
        return [_normalize_choice_answer(answer) for answer in model_answers]
    return [answer if answer is not None and str(answer).strip() else UNKNOWN_LABEL for answer in model_answers]


def _verify_single_answer(task, answer, ground_truth, extra_info=None):
    if answer == UNKNOWN_LABEL:
        return 0.0
    return 1.0 if auto_verify(task, [answer], [ground_truth], extra_info=extra_info)[0][0] else 0.0


def _compute_reference_rewards(task, solutions, ground_truth, extra_info=None):
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    return true_rewards


def _frequency_entropy_rewards(model_answers, entropy_coef):
    counter = Counter(model_answers)
    total = len(model_answers)
    reward_p = [counter[ans] / total for ans in model_answers]
    normalized_entropy = _normalized_entropy(counter, total)
    return [(r * 1) - (entropy_coef * normalized_entropy) for r in reward_p], counter, normalized_entropy


def _soft_pseudo_label_rewards(model_answers, gamma, unknown_reward, all_unknown_reward):
    valid_counter = Counter(ans for ans in model_answers if ans in CHOICE_LABELS)
    denom = sum(valid_counter[label] ** gamma for label in CHOICE_LABELS)

    if denom <= 0:
        rewards = [all_unknown_reward for _ in model_answers]
        soft_dist = {label: 0.0 for label in CHOICE_LABELS}
    else:
        soft_dist = {label: (valid_counter[label] ** gamma) / denom for label in CHOICE_LABELS}
        rewards = [
            soft_dist[answer] if answer in CHOICE_LABELS else unknown_reward
            for answer in model_answers
        ]

    return rewards, valid_counter, soft_dist


def _majority_vote_rewards(task, solutions, estimated_label, extra_info=None):
    if estimated_label == UNKNOWN_LABEL:
        return [0.0 for _ in solutions]
    rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    return rewards


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
    return_details=False,
):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = _extract_answers(
        task,
        solutions,
        ground_truth,
        extra_info=extra_info,
        answer_parse_mode=answer_parse_mode,
    )
    counter = Counter(model_answers)
    total = len(model_answers)

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
    elif reward_style == "soft_pseudo_label":
        rewards_en, valid_counter, _ = _soft_pseudo_label_rewards(
            model_answers,
            gamma=soft_label_gamma,
            unknown_reward=unknown_reward,
            all_unknown_reward=all_unknown_reward,
        )
        normalized_entropy = _normalized_entropy(Counter(ans for ans in model_answers if ans in CHOICE_LABELS), sum(valid_counter.values()))
        actual_reward_mean_key = "soft_pseudo_label_reward"
    elif reward_style == "majority_vote":
        rewards_en = rewards
        normalized_entropy = _normalized_entropy(counter, total)
        actual_reward_mean_key = "majority_vote_reward"
    elif reward_style == "vision_self_harmony":
        if transform_solutions is None:
            raise ValueError("vision_self_harmony requires transform_solutions")
        transform_answers = _extract_answers(
            task,
            transform_solutions,
            ground_truth,
            extra_info=transform_extra_info if transform_extra_info is not None else extra_info,
            answer_parse_mode=answer_parse_mode,
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
    else:
        raise ValueError(f"Unsupported TTRL reward_style: {reward_style}")

    hit_rate = _verify_single_answer(task, estimated_label, ground_truth, extra_info=extra_info)
    majority_ratio = majority_count / len(solutions)
    

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
    if return_details:
        details["original_answers"] = model_answers
        details["transform_answers"] = transform_answers if reward_style == "vision_self_harmony" else []
        return rewards_en, ttrl_metrics, details
    return rewards_en, ttrl_metrics

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = _extract_answers(task, solutions, ground_truth, extra_info=extra_info)

    # counter = Counter(model_answers)
    
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)

    # Compare pred_rewards with true_rewards to calculate reward hit rate
    rewards_hit_rate = sum(
        1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)
    ) / len(pred_rewards)



    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics
