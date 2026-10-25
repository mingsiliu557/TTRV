#!/usr/bin/env python3
"""Extract final-validation cases where our checkpoint is correct and TTRV is wrong."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Comparison:
    name: str
    kind: str
    ours_path: str
    ttrv_path: str
    ttrv_variant: str


COMPARISONS = (
    Comparison(
        name="MME",
        kind="closed",
        ours_path="/data/ttrv/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/best_ckpt_pass8_mme20/validation_groups.jsonl",
        ttrv_path="/data/ttrv/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/ttrv_official_mme20_train_s10_s20_pass8/validation_groups.jsonl",
        ttrv_variant="official step20",
    ),
    Comparison(
        name="AI2D",
        kind="closed",
        ours_path="/data/ttrv/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/best_ckpt_pass8_ai2d20/validation_groups.jsonl",
        ttrv_path="/data/ttrv/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/ttrv_official_ai2d20_train_s10_s20_pass8/validation_groups.jsonl",
        ttrv_variant="official step20",
    ),
    Comparison(
        name="VisualSimpleQA",
        kind="open",
        ours_path="/data/ttrv/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/visualsimpleqa_mask045_coef002_z1/validation_flat.jsonl",
        ttrv_path="/data/ttrv/experiments/visualsimpleqa_da_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_visualsimpleqa_frequency_valid_entropy/validation_flat.jsonl",
        ttrv_variant="frequency_valid_entropy final",
    ),
    Comparison(
        name="OCRBench",
        kind="open",
        ours_path="/data/ttrv/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/ocrbench_v1_mask030_coef005/validation_flat.jsonl",
        ttrv_path="/data/ttrv/experiments/ocrbench_v1_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_ocrbench_v1_frequency_valid_entropy/validation_flat.jsonl",
        ttrv_variant="frequency_valid_entropy final",
    ),
    Comparison(
        name="A-OKVQA",
        kind="open",
        ours_path="/data/ttrv/experiments/visual_dep_reward_direct_answer_20260706_163934/aokvqa_da_val_coef005/validation_flat.jsonl",
        ttrv_path="/data/ttrv/experiments/aokvqa_da_val_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_aokvqa_ttrv_official_frequency_entropy/validation_flat.jsonl",
        ttrv_variant="official frequency_entropy final",
    ),
)


CURATED_IDS = (
    "mme_20-test-738",
    "mme_20-test-447",
    "mme_20-test-21",
    "ai2d_20-test-33",
    "ai2d_20-test-1954",
    "ai2d_20-test-2394",
    "visualsimpleqa_da_20_hq_major_correct-test-184",
    "visualsimpleqa_da_20_hq_major_correct-test-111",
    "ocrbench_v1_20_hq_major_correct-test-647",
    "ocrbench_v1_20_hq_major_correct-test-529",
    "ocrbench_v1_20_hq_major_correct-test-112",
    "aokvqa_da_val_20_hq_major_correct-test-672",
    "aokvqa_da_val_20_hq_major_correct-test-751",
    "aokvqa_da_val_20_hq_major_correct-test-936",
)


RATIONALES = {
    "mme_20-test-738": "Explicit spatial-relation evidence: the two elephants and their depth ordering are visible.",
    "mme_20-test-447": "A compact geometry question whose numerical evidence is entirely in the image.",
    "mme_20-test-21": "Code-reading case: the screenshot contains the program needed to reject the proposed output.",
    "ai2d_20-test-33": "Diagram counting: the answer follows from the number of depicted stages.",
    "ai2d_20-test-1954": "Diagram-label grounding: the model must map a moon phase to the labeled position.",
    "ai2d_20-test-2394": "Visual morphology and label grounding: identify the lobed leaf and map it to its letter.",
    "visualsimpleqa_da_20_hq_major_correct-test-184": "Valid semantic correction on a place/country identification question.",
    "visualsimpleqa_da_20_hq_major_correct-test-111": "Fine-grained material recognition; included as a secondary, potentially ambiguous case.",
    "ocrbench_v1_20_hq_major_correct-test-647": "Valid semantic OCR/chart error: TTRV names the measured quantity rather than the red-bar category.",
    "ocrbench_v1_20_hq_major_correct-test-529": "Answer precision case: TTRV contains the answer but appends an address, failing exact evaluation.",
    "ocrbench_v1_20_hq_major_correct-test-112": "Invalid-output collapse case: our model reads the word while TTRV emits no usable answer.",
    "aokvqa_da_val_20_hq_major_correct-test-672": "Valid semantic correction: tusks versus trunk.",
    "aokvqa_da_val_20_hq_major_correct-test-751": "Valid semantic correction: Egypt versus Bulgaria.",
    "aokvqa_da_val_20_hq_major_correct-test-936": "Valid semantic correction with opposite actions: wake up versus sleep.",
}


def load_last_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """Keep the last occurrence, which is the final validation for appended logs."""
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["index"])] = row
    return rows


def unique(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def prompt_body(prompt: Any) -> str:
    text = str(prompt)
    if "</img>" in text:
        text = text.split("</img>", 1)[1]
    text = text.replace("<|im_end|><|im_start|>assistant", "")
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    return text.strip()


def parse_question_and_options(prompt: Any) -> tuple[str, dict[str, str]]:
    body = prompt_body(prompt)
    question = body
    for marker in (
        "Please answer with yes or no.",
        "Choose the correct answer from the options below",
        "Answer with a short direct answer.",
        "Answer with the exact text from the image.",
    ):
        if marker in question:
            question = question.split(marker, 1)[0]
            break

    options: dict[str, str] = {}
    if "Options:" in body:
        option_text = body.split("Options:", 1)[1]
        matches = list(re.finditer(r"(?:^|\n)\s*([A-F])\.\s*", option_text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(option_text)
            value = option_text[match.end() : end].strip()
            options[match.group(1)] = re.sub(r"\s+", " ", value)

    return re.sub(r"\s+", " ", question).strip(), options


def representative_response(row: dict[str, Any], prediction: str) -> str:
    for sample in row.get("samples", []):
        if str(sample.get("prediction")) == prediction:
            return str(sample.get("response_raw", ""))
    return str(row.get("response_raw", row.get("single_response_raw", "")))


def is_invalid_prediction(prediction: str, options: dict[str, str], kind: str) -> bool:
    if kind == "closed":
        return prediction not in options
    normalized = re.sub(r"[^a-z0-9]+", "", prediction.lower())
    return normalized in {"", "unknown", "invalid", "none"}


def make_case(
    comparison: Comparison,
    question_id: str,
    ours: dict[str, Any],
    ttrv: dict[str, Any],
) -> dict[str, Any] | None:
    question, options = parse_question_and_options(ours.get("prompt", ""))
    ground_truth = str(ours.get("ground_truth", ""))

    if comparison.kind == "closed":
        ours_prediction = str(ours.get("majority_prediction", ""))
        ttrv_prediction = str(ttrv.get("majority_prediction", ""))
        ours_correct = ours_prediction == ground_truth
        ttrv_correct = ttrv_prediction == ground_truth
        protocol = "final majority@8"
        ours_raw = representative_response(ours, ours_prediction)
        ttrv_raw = representative_response(ttrv, ttrv_prediction)
    else:
        ours_prediction = str(ours.get("prediction", ""))
        ttrv_prediction = str(ttrv.get("prediction", ""))
        ours_correct = bool(ours.get("correct"))
        ttrv_correct = bool(ttrv.get("correct"))
        protocol = "final greedy"
        ours_raw = str(ours.get("response_raw", ""))
        ttrv_raw = str(ttrv.get("response_raw", ""))

    if not ours_correct or ttrv_correct:
        return None

    extra_info = ours.get("extra_info", {}) or {}
    official_answers = unique(list(extra_info.get("official_answers", [])))
    normalized_official = {
        re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip() for answer in official_answers
    }
    normalized_ours = re.sub(r"[^a-z0-9]+", " ", ours_prediction.lower()).strip()

    case: dict[str, Any] = {
        "dataset": comparison.name,
        "kind": comparison.kind,
        "question_id": question_id,
        "protocol": protocol,
        "ttrv_variant": comparison.ttrv_variant,
        "question": question,
        "options": options,
        "ground_truth": ground_truth,
        "ground_truth_option_text": options.get(ground_truth),
        "official_answers": official_answers,
        "ours_prediction": ours_prediction,
        "ours_prediction_text": options.get(ours_prediction),
        "ours_response_raw": ours_raw,
        "ttrv_prediction": ttrv_prediction,
        "ttrv_prediction_text": options.get(ttrv_prediction),
        "ttrv_response_raw": ttrv_raw,
        "ttrv_error_type": (
            "invalid_or_empty"
            if is_invalid_prediction(ttrv_prediction, options, comparison.kind)
            else "valid_wrong"
        ),
        "ours_exact_accepted_answer": bool(official_answers and normalized_ours in normalized_official),
        "image_path": extra_info.get("image_path"),
        "category": extra_info.get("category", extra_info.get("mme_category")),
        "question_type": extra_info.get("question_type"),
        "ours_source": comparison.ours_path,
        "ttrv_source": comparison.ttrv_path,
    }
    if comparison.kind == "closed":
        case.update(
            {
                "ours_prediction_counter": ours.get("prediction_counter", {}),
                "ttrv_prediction_counter": ttrv.get("prediction_counter", {}),
                "ours_samples": ours.get("samples", []),
                "ttrv_samples": ttrv.get("samples", []),
            }
        )
    return case


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_curated_images(output_dir: Path, cases: list[dict[str, Any]]) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        source_value = case.get("image_path")
        if not source_value:
            continue
        source = Path(source_value)
        if not source.exists():
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case["question_id"])
        target = image_dir / f"{case['dataset'].lower().replace('-', '_')}__{safe_id}{source.suffix.lower()}"
        shutil.copy2(source, target)
        case["image_relpath"] = str(target.relative_to(output_dir))


def prediction_display(case: dict[str, Any], prefix: str) -> str:
    prediction = str(case[f"{prefix}_prediction"])
    text = case.get(f"{prefix}_prediction_text")
    if text:
        return f"{prediction} ({text})"
    return prediction


def inline_code(value: Any) -> str:
    return str(value).replace(chr(96), chr(92) + chr(96)).replace("\r", "\\r").replace("\n", "\\n")


def markdown_report(
    output_dir: Path,
    all_cases: list[dict[str, Any]],
    curated: list[dict[str, Any]],
    stats: list[dict[str, Any]],
) -> None:
    lines = [
        "# Ours-Correct / TTRV-Wrong Case Study",
        "",
        "## Comparison protocol",
        "",
        "- Ours means the best recorded KL-as-reward visual-dependency checkpoint for each dataset.",
        "- Closed-choice cases use the same saved final `majority@8` protocol for both checkpoints. "
        "Only MME and AI2D have matching step20 TTRV group files under this protocol.",
        "- Open-ended cases use final greedy per-example outputs. VisualSimpleQA and OCRBench compare "
        "against `frequency_valid_entropy`; A-OKVQA compares against official-style `frequency_entropy`.",
        "- Appended validation logs are deduplicated by question ID, retaining the last occurrence.",
        "- A-OKVQA correctness follows its saved soft VQA evaluator. Curated A-OKVQA examples also "
        "require our normalized answer to exactly match at least one human answer.",
        "",
        "## Candidate counts",
        "",
        "| Dataset | Protocol | Common questions | Ours correct / TTRV wrong | TTRV valid wrong | TTRV invalid/empty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| {row['dataset']} | {row['protocol']} | {row['common']} | {row['candidates']} | "
            f"{row['valid_wrong']} | {row['invalid_or_empty']} |"
        )
    lines.extend(
        [
            "",
            f"Full machine-readable candidate set: [`all_candidates.jsonl`](all_candidates.jsonl) "
            f"({len(all_cases)} rows).",
            "",
            "## Curated closed-choice cases",
            "",
        ]
    )

    number = 0
    for case in curated:
        if case["kind"] != "closed":
            continue
        number += 1
        lines.extend(render_case(number, case))

    lines.extend(["## Curated open-ended cases", ""])
    number = 0
    for case in curated:
        if case["kind"] != "open":
            continue
        number += 1
        lines.extend(render_case(number, case))

    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- These are selected qualitative examples, not an unbiased estimate of effect size. Use the "
            "candidate-count table and the main benchmark table for aggregate claims.",
            "- `valid_wrong` cases support a semantic/visual correction claim more directly. "
            "`invalid_or_empty` cases primarily document output collapse or parser-visible failure.",
            "- Exact-answer tasks can mark a semantically containing but verbose response wrong. OCRBench "
            "case `test-529` is retained intentionally to show this output-precision mechanism.",
            "- The closed-choice cases are sampled comparisons (`majority@8`), while open-ended cases are "
            "greedy comparisons; they should not be pooled into one numerical accuracy estimate.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))


def render_case(number: int, case: dict[str, Any]) -> list[str]:
    lines = [f"### {number}. {case['dataset']} / `{case['question_id']}`", ""]
    if case.get("image_relpath"):
        lines.extend([f"![{case['question_id']}]({case['image_relpath']})", ""])
    lines.extend(
        [
            f"- **Question:** {case['question']}",
            f"- **Ground truth:** {format_ground_truth(case)}",
            f"- **Ours:** {prediction_display(case, 'ours')} (correct)",
            f"- **TTRV:** {prediction_display(case, 'ttrv')} (wrong; `{case['ttrv_error_type']}`)",
        ]
    )
    if case["kind"] == "closed":
        lines.extend(
            [
                f"- **Ours vote counter:** `{json.dumps(case['ours_prediction_counter'], ensure_ascii=False)}`",
                f"- **TTRV vote counter:** `{json.dumps(case['ttrv_prediction_counter'], ensure_ascii=False)}`",
            ]
        )
    lines.extend(
        [
            f"- **Ours raw:** `{inline_code(case['ours_response_raw'])}`",
            f"- **TTRV raw:** `{inline_code(case['ttrv_response_raw'])}`",
            f"- **Why selected:** {RATIONALES.get(case['question_id'], '')}",
            "",
        ]
    )
    return lines


def format_ground_truth(case: dict[str, Any]) -> str:
    ground_truth = str(case["ground_truth"])
    option_text = case.get("ground_truth_option_text")
    if option_text:
        return f"{ground_truth} ({option_text})"
    accepted = case.get("official_answers", [])
    if accepted:
        return " / ".join(accepted)
    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("case_studies/ours_correct_ttrv_wrong_20260726"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_cases: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        ours = load_last_by_id(Path(comparison.ours_path))
        ttrv = load_last_by_id(Path(comparison.ttrv_path))
        common = sorted(set(ours) & set(ttrv))
        cases = [
            case
            for question_id in common
            if (
                case := make_case(
                    comparison,
                    question_id,
                    ours[question_id],
                    ttrv[question_id],
                )
            )
            is not None
        ]
        all_cases.extend(cases)
        error_counts: dict[str, int] = {"valid_wrong": 0, "invalid_or_empty": 0}
        for case in cases:
            error_counts[case["ttrv_error_type"]] += 1
        stats.append(
            {
                "dataset": comparison.name,
                "protocol": "majority@8" if comparison.kind == "closed" else "greedy",
                "common": len(common),
                "candidates": len(cases),
                **error_counts,
            }
        )

    case_by_id = {case["question_id"]: case for case in all_cases}
    missing = [question_id for question_id in CURATED_IDS if question_id not in case_by_id]
    if missing:
        raise RuntimeError(f"Curated IDs are no longer valid candidates: {missing}")
    curated = [case_by_id[question_id] for question_id in CURATED_IDS]
    copy_curated_images(args.output_dir, curated)

    jsonl_write(args.output_dir / "all_candidates.jsonl", all_cases)
    jsonl_write(args.output_dir / "curated_cases.jsonl", curated)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"stats": stats, "curated_count": len(curated)}, indent=2)
    )
    markdown_report(args.output_dir, all_cases, curated, stats)

    print(json.dumps({"output_dir": str(args.output_dir), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
