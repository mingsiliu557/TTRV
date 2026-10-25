"""Data-only InternVL prompt rules; no model construction or environment CoT."""
from scripts.official_data.upstream import module

build_multi_choice_prompt = module("scoring").build_multi_choice_prompt


def build_prompt(record, dataset):
    if dataset == "AI2D_TEST":
        text = build_multi_choice_prompt(record, dataset)
    elif dataset == "MME":
        # Exact non-CoT MME branch in pinned internvl_chat.py::build_prompt.
        text = record["question"] + " Answer the question using a single word or phrase."
    else:
        raise ValueError(f"Unsupported official dataset: {dataset}")
    if "<image>" in text:
        raise ValueError("Unexpected image placeholder in official question")
    return "<image>\n" + text
