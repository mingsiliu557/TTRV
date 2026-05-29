import math
import re
from collections import Counter, defaultdict
from typing import Iterable


MODELNET40_CLASSES = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]

CLASS_ALIASES = {
    "airplane": ["aircraft", "plane", "jet"],
    "bathtub": ["bath tub", "tub"],
    "bookshelf": ["book shelf", "bookcase"],
    "flower_pot": ["flower pot", "flowerpot", "plant pot", "pot of flowers"],
    "glass_box": ["glass box", "glass case", "glass cabinet"],
    "laptop": ["notebook computer"],
    "mantel": ["mantle", "fireplace mantel"],
    "monitor": ["computer monitor", "screen", "display"],
    "night_stand": ["night stand", "nightstand", "bedside table"],
    "range_hood": ["range hood", "kitchen hood", "exhaust hood"],
    "sofa": ["couch"],
    "tv_stand": ["tv stand", "television stand", "tv cabinet"],
    "xbox": ["game console", "gaming console"],
}


def _normalize_text(text: str) -> str:
    text = text.strip().lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _normalize_text(phrase)
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def map_to_modelnet40_class(text: str) -> str:
    """Map free-form text to a ModelNet40 class index string, or "unknown"."""
    if text is None:
        return "unknown"
    norm_text = _normalize_text(str(text))
    candidates = []
    for idx, cls in enumerate(MODELNET40_CLASSES):
        terms = {cls, cls.replace("_", " ")}
        terms.update(CLASS_ALIASES.get(cls, []))
        for term in terms:
            candidates.append((idx, cls, _normalize_text(term)))

    for idx, _cls, term in sorted(candidates, key=lambda item: -len(item[2])):
        if _contains_phrase(norm_text, term):
            return str(idx)
    return "unknown"


def _label_to_index(label) -> str:
    if label is None:
        return "unknown"
    if isinstance(label, (int, float)) and int(label) == label:
        return str(int(label))
    label_text = str(label).strip()
    if label_text.isdigit():
        return label_text
    return map_to_modelnet40_class(label_text)


def _entropy(mapped: Iterable[str]) -> float:
    mapped = list(mapped)
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


def ttrv_score_freeform(
    group_responses: list[str],
    ground_truth: str = None,
    alpha: float = 0.75,
    unknown_reward: float = -1.0,
) -> list[float]:
    mapped = [map_to_modelnet40_class(response) for response in group_responses]
    total = len(mapped)
    counts = Counter(mapped)
    entropy = _normalized_entropy(mapped)
    return [
        unknown_reward if pred == "unknown" else (counts[pred] / total) - alpha * entropy
        for pred in mapped
    ]


def compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    alpha: float = 0.75,
    unknown_reward: float = -1.0,
    train_group_size: int | None = None,
    **_kwargs,
):
    """Batch reward function for ModelNet40 free-form TTRV.

    Training groups multiple rollouts by extra_info["index"] and returns
    frequency - alpha * normalized_entropy for mapped ModelNet40 classes.
    Unmapped free-form responses get a fixed penalty so all-unknown groups
    cannot receive maximal self-consistency reward. Validation normally has
    one rollout per sample, so score is exact-match accuracy after free-form
    normalization.
    """
    outputs = [None] * len(solution_strs)
    groups = defaultdict(list)
    for i, extra in enumerate(extra_infos):
        extra = extra or {}
        key = extra.get("index", i)
        groups[key].append(i)

    for indices in groups.values():
        responses = [solution_strs[i] for i in indices]
        mapped = [map_to_modelnet40_class(response) for response in responses]
        counts = Counter(mapped)
        entropy = _normalized_entropy(mapped)
        group_size = len(indices)
        is_eval = group_size == 1 and (train_group_size is None or train_group_size != 1)

        for local_i, global_i in enumerate(indices):
            extra = extra_infos[global_i] or {}
            label_id = extra.get("label_id", ground_truths[global_i])
            target = _label_to_index(label_id)
            pred = mapped[local_i]
            acc = 1.0 if pred == target else 0.0
            freq = counts[pred] / group_size
            if is_eval:
                reward = acc
            elif pred == "unknown":
                reward = unknown_reward
            else:
                reward = freq - alpha * entropy
            outputs[global_i] = {
                "score": float(reward),
                "acc": float(acc),
                "pred": pred,
                "unknown": 1.0 if pred == "unknown" else 0.0,
                "freq": float(freq),
                "entropy": float(entropy),
            }

    return outputs
