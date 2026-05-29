#!/usr/bin/env python3
import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "verl"))

from verl.utils.reward_score.sqa3d_ttrv import (  # noqa: E402
    _load_answer_dict,
    _load_answer_prior,
    canonicalize_sqa3d_freeform,
    evaluate_sqa3d_em,
    evaluate_sqa3d_official_freeform,
    evaluate_sqa3d_strict_freeform,
    map_to_sqa3d_answer,
    normalize_sqa3d_answer,
)
from summarize_sqa3d_results import parse_step_metrics  # noqa: E402


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESPONSE_RE = re.compile(r"\[response\]\s*(.*)")
SCORE_RE = re.compile(r"\[score\]\s*(\{.*\})")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def load_dataset_rows(path: Path):
    return list(Dataset.from_parquet(str(path)))


def ground_truth_coverage(rows, answer_dict, answer_prior):
    primary_unknown = 0
    any_unknown = 0
    primary_top = Counter()
    any_top = Counter()
    examples = []
    for row in rows:
        gt = row.get("reward_model", {}).get("ground_truth", "")
        extra = row.get("extra_info", {}) or {}
        answers = extra.get("answers") or [gt]
        primary = map_to_sqa3d_answer(gt, answer_dict, answer_prior=answer_prior)
        mapped_answers = [map_to_sqa3d_answer(answer, answer_dict, answer_prior=answer_prior) for answer in answers]
        primary_top[primary] += 1
        for answer in mapped_answers:
            any_top[answer] += 1
        if primary == "unknown":
            primary_unknown += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "question_id": extra.get("question_id"),
                        "gt": gt,
                        "normalized": normalize_sqa3d_answer(gt),
                        "answers": answers,
                    }
                )
        if all(answer == "unknown" for answer in mapped_answers):
            any_unknown += 1
    total = max(len(rows), 1)
    return {
        "total": len(rows),
        "closed_primary_gt_unknown_rate": primary_unknown / total,
        "closed_all_gt_answers_unknown_rate": any_unknown / total,
        "closed_primary_gt_top10": primary_top.most_common(10),
        "closed_all_gt_answer_top10": any_top.most_common(10),
        "closed_primary_gt_unknown_examples": examples,
    }


def parse_log(log_file: Path, answer_dict, answer_prior):
    responses = []
    scores = []
    for line in strip_ansi(log_file.read_text(errors="ignore")).splitlines():
        response_match = RESPONSE_RE.search(line)
        if response_match:
            responses.append(response_match.group(1))
            continue
        score_match = SCORE_RE.search(line)
        if score_match:
            try:
                scores.append(ast.literal_eval(score_match.group(1)))
            except (SyntaxError, ValueError):
                pass

    freeform_preds = [canonicalize_sqa3d_freeform(response) for response in responses]
    closed_mapped = [map_to_sqa3d_answer(response, answer_dict, answer_prior=answer_prior) for response in responses]
    score_preds = [str(score.get("pred", "")) for score in scores if isinstance(score, dict)]
    score_closed_preds = [str(score.get("closed_pred", "")) for score in scores if isinstance(score, dict)]
    score_unknowns = [float(score.get("unknown", 0.0)) for score in scores if isinstance(score, dict)]
    score_closed_unknowns = [float(score.get("closed_unknown", 0.0)) for score in scores if isinstance(score, dict)]
    score_accs = [float(score.get("acc", score.get("official_acc", 0.0))) for score in scores if isinstance(score, dict)]
    score_strict_accs = [float(score.get("strict_acc", 0.0)) for score in scores if isinstance(score, dict)]
    score_closed_accs = [float(score.get("closed_acc", 0.0)) for score in scores if isinstance(score, dict)]
    total = max(len(responses), 1)
    summary = {
        "responses": len(responses),
        "scores": len(scores),
        "blank_rate": sum(1 for response in responses if not response.strip()) / total,
        "freeform_unknown_rate": sum(1 for pred in freeform_preds if pred == "unknown") / total,
        "closed_mapped_unknown_rate": sum(1 for pred in closed_mapped if pred == "unknown") / total,
        "score_official_unknown_rate": sum(score_unknowns) / max(len(score_unknowns), 1) if score_unknowns else None,
        "score_closed_unknown_rate": (
            sum(score_closed_unknowns) / max(len(score_closed_unknowns), 1) if score_closed_unknowns else None
        ),
        "score_official_em": sum(score_accs) / max(len(score_accs), 1) if score_accs else None,
        "score_strict_em": sum(score_strict_accs) / max(len(score_strict_accs), 1) if score_strict_accs else None,
        "score_closed_em": sum(score_closed_accs) / max(len(score_closed_accs), 1) if score_closed_accs else None,
        "freeform_pred_top20": Counter(freeform_preds).most_common(20),
        "closed_mapped_pred_top20": Counter(closed_mapped).most_common(20),
        "score_pred_top20": Counter(score_preds).most_common(20),
        "score_closed_pred_top20": Counter(score_closed_preds).most_common(20),
        "raw_response_examples": responses[:20],
    }
    val_steps = [
        (step, metrics)
        for step, metrics in parse_step_metrics(log_file)
        if "val-sqa3d/sqa3d/em/overall/baseline" in metrics
    ]
    if val_steps:
        first_step, first_metrics = val_steps[0]
        final_step, final_metrics = val_steps[-1]
        summary["validation_initial"] = {
            "step": first_step,
            "official_em": first_metrics.get("val-sqa3d/sqa3d/em/overall/baseline"),
            "strict_em": first_metrics.get("val-sqa3d/sqa3d/strict_em/overall/baseline"),
            "closed_em": first_metrics.get("val-sqa3d/sqa3d/closed_em/overall/baseline"),
            "unknown": first_metrics.get("val-sqa3d/sqa3d/unknown/overall/baseline"),
            "closed_unknown": first_metrics.get("val-sqa3d/sqa3d/closed_unknown/overall/baseline"),
            "blank": first_metrics.get("val-sqa3d/sqa3d/blank/overall/baseline"),
        }
        summary["validation_final"] = {
            "step": final_step,
            "official_em": final_metrics.get("val-sqa3d/sqa3d/em/overall/baseline"),
            "strict_em": final_metrics.get("val-sqa3d/sqa3d/strict_em/overall/baseline"),
            "closed_em": final_metrics.get("val-sqa3d/sqa3d/closed_em/overall/baseline"),
            "unknown": final_metrics.get("val-sqa3d/sqa3d/unknown/overall/baseline"),
            "closed_unknown": final_metrics.get("val-sqa3d/sqa3d/closed_unknown/overall/baseline"),
            "blank": final_metrics.get("val-sqa3d/sqa3d/blank/overall/baseline"),
        }
    return summary


def maybe_eval_rows_against_responses(rows, responses, answer_dict, answer_prior):
    if not responses:
        return None
    total = min(len(rows), len(responses))
    if total == 0:
        return None
    correct = 0
    for row, response in zip(rows[:total], responses[:total]):
        extra = row.get("extra_info", {}) or {}
        answers = extra.get("answers") or [row.get("reward_model", {}).get("ground_truth", "")]
        correct += int(evaluate_sqa3d_em(response, answers, answer_dict, answer_prior=answer_prior))
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-file", required=True)
    parser.add_argument("--answer-dict-path", default=str(REPO_ROOT / "data/sqa3d_data/answer_dict.json"))
    parser.add_argument("--log-file")
    args = parser.parse_args()

    answer_dict = _load_answer_dict(args.answer_dict_path)
    answer_prior = _load_answer_prior(args.answer_dict_path)
    rows = load_dataset_rows(Path(args.parquet_file))
    print("[dataset]")
    for key, value in ground_truth_coverage(rows, answer_dict, answer_prior).items():
        print(f"{key}: {value}")

    if args.log_file:
        log_summary = parse_log(Path(args.log_file), answer_dict, answer_prior)
        print("\n[log]")
        for key, value in log_summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
