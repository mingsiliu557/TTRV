from collections import defaultdict
import re

from tqdm import tqdm

from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.direct_answer import (
    aokvqa_direct_answer_score, direct_answer_exact_score,
)
from verl.utils.reward_score.ttrl.qwen.qwen_eval import (qwen_reward_fn,
                                                         qwen_reward_fn_gpqa,
                                                         simplerl_reward_fn, qwen_reward_fn_spatial)


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extra_info_at(extra_info, idx):
    if isinstance(extra_info, (list, tuple)):
        if idx < len(extra_info) and isinstance(extra_info[idx], dict):
            return extra_info[idx]
        return {}
    if isinstance(extra_info, dict):
        return extra_info
    return {}


def _metric_at(extra_info, idx):
    return str(_extra_info_at(extra_info, idx).get("metric", "")).strip().lower()


def _parse_number(value):
    text = str(value or "").replace(",", "").strip()
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _capture_smape_score(prediction, label):
    pred_num = _parse_number(prediction)
    label_num = _parse_number(label)
    if pred_num is None or label_num is None:
        return 0.0
    denom = abs(pred_num) + abs(label_num)
    if denom == 0:
        return 1.0 if pred_num == label_num else 0.0
    return max(0.0, 1.0 - abs(pred_num - label_num) / denom)


def auto_verify(task, all_outputs, all_labels, extra_info=None):

    task2verify = {
        "math": qwen_reward_fn,
        "simplerl_math": simplerl_reward_fn,
        "gpqa": qwen_reward_fn_gpqa,
        "bbox": qwen_reward_fn_spatial,
        "vqa_da": None,
        "ocr": None,
    }
    assert task in task2verify, f"{task} not in {list(task2verify.keys())}"
    verify_fn = task2verify[task]
    verify_extra_info = defaultdict(list)

    all_outputs = auto_extract(task, all_outputs, extra_info=extra_info)

    rewards = []
    exact_acc = []
    metrics = []
    for idx, (output, label) in enumerate(zip(all_outputs, all_labels)):
        metric = _metric_at(extra_info, idx)
        metrics.append(metric)
        if metric == "smape":
            rewards.append(_capture_smape_score(output, label))
            exact_acc.append(float(verify_fn(output, label)))
        elif task in {"vqa_da", "ocr"}:
            item_info = _extra_info_at(extra_info, idx)
            references = (
                item_info.get("official_answers")
                or item_info.get("direct_answers")
                or item_info.get("answers")
                or label
            )
            if metric == "aokvqa_direct_answer":
                reward = aokvqa_direct_answer_score(output, references)
            else:
                reward = direct_answer_exact_score(output, references)
            rewards.append(float(reward))
            exact_acc.append(float(direct_answer_exact_score(output, references)))
        else:
            reward = verify_fn(output, label)
            rewards.append(reward)
            exact_acc.append(float(reward))

    verify_extra_info["acc"] = rewards
    verify_extra_info["exact_acc"] = exact_acc
    verify_extra_info["metric"] = metrics
    verify_extra_info["pred"] = all_outputs

    return rewards, verify_extra_info
