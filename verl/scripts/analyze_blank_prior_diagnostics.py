#!/usr/bin/env python3
"""Recompute greedy-only blank-image prior diagnostics from existing artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


BLANK_ROOT = Path(
    "/data/ttrv/experiments/"
    "blank_image_drop_closed_choice_20260710_064900"
)
TTRV_HISTORICAL_GREEDY_MACRO_ACC = 64.75
TTRV_GREEDY_LAUNCH_DIR = Path(
    "/workspace/TTRV/outputs/launch_scripts/"
    "p0_valid_ttrv_official_fix_20260612_095939"
)
TTRV_CLEANUP_MANIFEST = Path(
    "/workspace/TTRV/outputs/"
    "cleanup_sota_20260617_deleted_dirs.txt"
)
TTRV_RETAINED_AGGREGATE = BLANK_ROOT / "blank_image_drop_summary.md"


DATASETS: dict[str, dict[str, Any]] = {
    "mme": {
        "name": "MME20",
        "labels": "AB",
        "ours_normal": Path(
            "/data/ttrv/experiments/"
            "visual_dep_reward_coef_sweep_A_20260705_131304/"
            "mme_coef002_ori000/validation_step10_20_flat.jsonl"
        ),
        "ours_white": BLANK_ROOT / "our_mme20_white/validation_flat.jsonl",
        "ttrv_white": BLANK_ROOT / "ttrv_mme20_white/validation_flat.jsonl",
    },
    "ai2d": {
        "name": "AI2D20",
        "labels": "ABCD",
        "ours_normal": Path(
            "/data/ttrv/experiments/"
            "visual_dep_reward_coef_sweep_A_20260705_131304/"
            "ai2d_coef003_ori000/validation_step10_20_flat.jsonl"
        ),
        "ours_white": BLANK_ROOT / "our_ai2d20_white/validation_flat.jsonl",
        "ttrv_white": BLANK_ROOT / "ttrv_ai2d20_white/validation_flat.jsonl",
    },
    "rwq": {
        "name": "RealWorldQA-HQ",
        "labels": "ABCD",
        "ours_normal": Path(
            "/data/ttrv/experiments/"
            "visual_dep_reward_coef_sweep_remaining_20260706_092444/"
            "rwq_coef005_ori000/validation_step10_20_flat.jsonl"
        ),
        "ours_white": BLANK_ROOT / "our_rwq20_hq_white/validation_flat.jsonl",
        "ttrv_white": BLANK_ROOT / "ttrv_rwq20_hq_white/validation_flat.jsonl",
    },
    "seed": {
        "name": "SEED20",
        "labels": "ABCD",
        "ours_normal": Path(
            "/data/ttrv/experiments/"
            "visual_dep_reward_coef_sweep_A_20260705_131304/"
            "seed_coef003_ori003/validation_step10_20_flat.jsonl"
        ),
        "ours_white": BLANK_ROOT / "our_seed20_white/validation_flat.jsonl",
        "ttrv_white": BLANK_ROOT / "ttrv_seed20_white/validation_flat.jsonl",
    },
    "mathvista": {
        "name": "MathVista-HQ",
        "labels": "ABCDEF",
        "ours_normal": Path(
            "/data/ttrv/experiments/"
            "visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/"
            "mathvista20_hq/validation_step10_20_flat.jsonl"
        ),
        "ours_white": BLANK_ROOT / "our_mathvista20_hq_white/validation_flat.jsonl",
        "ttrv_white": BLANK_ROOT / "ttrv_mathvista20_hq_white/validation_flat.jsonl",
    },
    "mathverse": {
        "name": "MathVerse-fixed",
        "labels": "ABCDEF",
        "ours_normal": Path(
            "/data/ttrv/experiments/"
            "visual_dep_reward_coef_sweep_remaining_20260706_092444/"
            "mathverse_coef005_ori002/validation_step10_20_flat.jsonl"
        ),
        "ours_white": BLANK_ROOT
        / "our_mathverse20_fixed_white/validation_flat.jsonl",
        "ttrv_white": BLANK_ROOT
        / "ttrv_mathverse20_fixed_white/validation_flat.jsonl",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_by_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        records[str(record["index"])] = record
    return records


def unique_by_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    records = {str(row["index"]): row for row in rows}
    if len(records) != len(rows):
        raise ValueError(f"Expected unique indices in {path}: {len(rows)} rows")
    return records


def canonical(value: Any, labels: str) -> str:
    text = "" if value is None else str(value).strip().upper()
    return text if text in labels else "unknown"


def pct(count: int | float, total: int) -> float:
    return 100.0 * float(count) / total if total else math.nan


def category(record: dict[str, Any]) -> str:
    extra = record.get("extra_info") or {}
    for key in ("mme_category", "category", "source"):
        if extra.get(key) is not None:
            return str(extra[key])
    return "unknown"


def pair_metrics(
    normal: dict[str, dict[str, Any]],
    white: dict[str, dict[str, Any]],
    labels: str,
    normal_prediction: Callable[[dict[str, Any]], Any],
    normal_correct: Callable[[dict[str, Any], str], bool],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = sorted(set(normal) & set(white))
    if len(ids) != len(normal) or len(ids) != len(white):
        raise ValueError(
            f"Index mismatch: normal={len(normal)} white={len(white)} joined={len(ids)}"
        )

    counters = {
        "normal_correct": 0,
        "white_correct": 0,
        "flip": 0,
        "valid_valid": 0,
        "valid_valid_flip": 0,
        "pure_prior_correct": 0,
        "normal_correct_white_wrong": 0,
        "normal_wrong_white_correct": 0,
        "same_prediction": 0,
    }
    cases: list[dict[str, Any]] = []
    for index in ids:
        normal_row = normal[index]
        white_row = white[index]
        ground_truth = canonical(normal_row.get("ground_truth"), labels)
        normal_pred = canonical(normal_prediction(normal_row), labels)
        white_pred = canonical(white_row.get("prediction"), labels)
        normal_ok = bool(normal_correct(normal_row, ground_truth))
        white_ok = bool(white_row.get("correct"))
        flipped = normal_pred != white_pred
        both_valid = normal_pred != "unknown" and white_pred != "unknown"
        pure_prior = normal_pred == white_pred and normal_ok and white_ok

        counters["normal_correct"] += int(normal_ok)
        counters["white_correct"] += int(white_ok)
        counters["flip"] += int(flipped)
        counters["valid_valid"] += int(both_valid)
        counters["valid_valid_flip"] += int(both_valid and flipped)
        counters["pure_prior_correct"] += int(pure_prior)
        counters["normal_correct_white_wrong"] += int(normal_ok and not white_ok)
        counters["normal_wrong_white_correct"] += int(not normal_ok and white_ok)
        counters["same_prediction"] += int(not flipped)

        if flipped or pure_prior:
            cases.append(
                {
                    "case_type": "normal_white_pair",
                    "dataset_index": index,
                    "ground_truth": ground_truth,
                    "normal_prediction": normal_pred,
                    "white_prediction": white_pred,
                    "normal_correct": normal_ok,
                    "white_correct": white_ok,
                    "prediction_flipped": flipped,
                    "pure_prior_correct": pure_prior,
                    "category": category(normal_row),
                }
            )

    n = len(ids)
    normal_correct_n = counters["normal_correct"]
    return (
        {
            "n": n,
            **counters,
            "normal_acc": pct(normal_correct_n, n),
            "white_acc": pct(counters["white_correct"], n),
            "normal_to_white_drop": pct(
                normal_correct_n - counters["white_correct"], n
            ),
            "flip_rate": pct(counters["flip"], n),
            "valid_valid_flip_rate": pct(
                counters["valid_valid_flip"], counters["valid_valid"]
            ),
            "pure_prior_correct_rate": pct(counters["pure_prior_correct"], n),
            "white_failure_given_normal_correct": pct(
                counters["normal_correct_white_wrong"], normal_correct_n
            ),
        },
        cases,
    )


def white_metrics(
    records: dict[str, dict[str, Any]], labels: str
) -> dict[str, Any]:
    n = len(records)
    correct = sum(bool(row.get("correct")) for row in records.values())
    invalid = sum(
        canonical(row.get("prediction"), labels) == "unknown"
        for row in records.values()
    )
    return {
        "n": n,
        "correct": correct,
        "acc": pct(correct, n),
        "invalid": invalid,
        "invalid_rate": pct(invalid, n),
    }


def fmt(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join("---" for _ in headers) + "|")
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=BLANK_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "analysis_protocol": "greedy_only",
        "base_white_available": False,
        "delta_blank_status": (
            "not computable from existing artifacts: no base/step0 white-image "
            "validation output was found"
        ),
        "ttrv_normal_greedy_pair_status": (
            "not computable from retained artifacts: the historical greedy run "
            "configuration remains, but its logs and per-case validation files were deleted"
        ),
        "pair_metrics": {},
        "white_metrics": {},
        "sources": {},
    }
    all_cases: list[dict[str, Any]] = []

    for key, cfg in DATASETS.items():
        labels = str(cfg["labels"])
        ours_normal = latest_by_index(cfg["ours_normal"])
        ours_white = unique_by_index(cfg["ours_white"])
        ttrv_white = unique_by_index(cfg["ttrv_white"])

        ours_pair, ours_cases = pair_metrics(
            ours_normal,
            ours_white,
            labels,
            normal_prediction=lambda row: row.get("prediction"),
            normal_correct=lambda row, _gt: bool(row.get("correct")),
        )
        summary["pair_metrics"][key] = {
            "dataset": cfg["name"],
            "ours_greedy": ours_pair,
            "ttrv_greedy": None,
        }
        summary["white_metrics"][key] = {
            "dataset": cfg["name"],
            "ours": white_metrics(ours_white, labels),
            "ttrv": white_metrics(ttrv_white, labels),
        }
        summary["sources"][key] = {
            name: str(cfg[name])
            for name in ("ours_normal", "ours_white", "ttrv_white")
        }
        for case in ours_cases:
            case.update(dataset=cfg["name"], method="ours", decode="greedy")
            all_cases.append(case)

    ours_macro_normal = sum(
        entry["ours_greedy"]["normal_acc"]
        for entry in summary["pair_metrics"].values()
    ) / len(DATASETS)
    ours_macro_white = sum(
        entry["ours"]["acc"] for entry in summary["white_metrics"].values()
    ) / len(DATASETS)
    ttrv_macro_white = sum(
        entry["ttrv"]["acc"] for entry in summary["white_metrics"].values()
    ) / len(DATASETS)
    summary["macro"] = {
        "ours": {
            "normal_greedy_acc": ours_macro_normal,
            "white_greedy_acc": ours_macro_white,
            "drop": ours_macro_normal - ours_macro_white,
            "protocol_matched": True,
        },
        "ttrv": {
            "historical_normal_greedy_acc": TTRV_HISTORICAL_GREEDY_MACRO_ACC,
            "current_step10_white_greedy_acc": ttrv_macro_white,
            "arithmetic_gap": TTRV_HISTORICAL_GREEDY_MACRO_ACC - ttrv_macro_white,
            "protocol_matched": False,
            "reason": (
                "normal macro and white outputs come from different training runs/checkpoints; "
                "the gap is descriptive, not a paired causal estimate"
            ),
        },
    }
    summary["historical_ttrv_greedy_reference"] = {
        "macro_normal_acc": TTRV_HISTORICAL_GREEDY_MACRO_ACC,
        "launch_dir": str(TTRV_GREEDY_LAUNCH_DIR),
        "retained_aggregate": str(TTRV_RETAINED_AGGREGATE),
        "cleanup_manifest": str(TTRV_CLEANUP_MANIFEST),
        "decode": "do_sample=False, n=1, temperature=0.0",
        "per_case_outputs_available": False,
    }

    summary_path = args.output_dir / "blank_prior_diagnostics_recomputed.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cases_path = args.output_dir / "blank_prior_diagnostics_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in all_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    report: list[str] = [
        "# Recomputed Greedy-Only Blank-Image Diagnostics",
        "",
        "This report uses existing artifacts only. No model inference was rerun.",
        "Every result shown below uses greedy decoding. Sampled and voting metrics are "
        "intentionally excluded.",
        "",
        "## Evidence availability",
        "",
        "- Trained white-image greedy validation is complete for both methods and all six datasets.",
        "- Our normal-image and white-image files are checkpoint-matched greedy outputs, so "
        "per-case flips and pure-prior counts are valid paired statistics.",
        "- The historical TTRV greedy run is confirmed by the retained launcher: "
        "do_sample=False, n=1, temperature=0.0.",
        "- Its six experiment directories and the 25 MB sequence log were explicitly deleted "
        "during the 2026-06-17 cleanup. Only the recorded macro normal accuracy, 64.7500, remains.",
        "- Therefore TTRV per-case normal-to-white flips and pure-prior counts cannot be "
        "reconstructed without a new normal-image greedy inference from the exact white-eval checkpoint.",
        "- Base/step0 white-image outputs are also absent, so "
        "Delta_blank = trained_white_acc - base_white_acc remains unavailable.",
        "",
        "## Trained white-image greedy accuracy",
        "",
    ]
    rows = []
    for key, cfg in DATASETS.items():
        white = summary["white_metrics"][key]
        ours = white["ours"]
        ttrv = white["ttrv"]
        rows.append(
            [
                cfg["name"],
                fmt(ours["acc"]),
                f"{ours['correct']}/{ours['n']}",
                fmt(ttrv["acc"]),
                f"{ttrv['correct']}/{ttrv['n']}",
                fmt(ours["acc"] - ttrv["acc"]),
            ]
        )
    report.extend(
        markdown_table(
            [
                "Dataset",
                "Our white Acc",
                "Our correct",
                "TTRV white Acc",
                "TTRV correct",
                "Our - TTRV",
            ],
            rows,
        )
    )

    report.extend(["", "## Exact normal-to-white paired diagnostics", ""])
    rows = []
    for key, cfg in DATASETS.items():
        result = summary["pair_metrics"][key]["ours_greedy"]
        rows.append(
            [
                cfg["name"],
                result["n"],
                fmt(result["normal_acc"]),
                fmt(result["white_acc"]),
                fmt(result["normal_to_white_drop"]),
                fmt(result["flip_rate"]),
                f"{result['pure_prior_correct']}/{result['n']}",
                fmt(result["pure_prior_correct_rate"]),
                fmt(result["white_failure_given_normal_correct"]),
            ]
        )
    report.extend(
        markdown_table(
            [
                "Dataset",
                "N",
                "Normal Acc",
                "White Acc",
                "Acc drop",
                "Answer flip %",
                "Same and both correct",
                "Pure-prior correct %",
                "P(white wrong | normal correct) %",
            ],
            rows,
        )
    )
    report.extend(
        [
            "",
            "Only our method appears in this paired table because no retained TTRV normal-image "
            "greedy per-case file matches the checkpoint used for the TTRV white-image inference.",
            "",
            "## Macro reference",
            "",
        ]
    )
    report.extend(
        markdown_table(
            ["Method", "Normal greedy Acc", "White greedy Acc", "Difference", "Matched"],
            [
                [
                    "Our method",
                    fmt(ours_macro_normal),
                    fmt(ours_macro_white),
                    fmt(ours_macro_normal - ours_macro_white),
                    "yes",
                ],
                [
                    "TTRV historical aggregate",
                    fmt(TTRV_HISTORICAL_GREEDY_MACRO_ACC),
                    fmt(ttrv_macro_white),
                    fmt(TTRV_HISTORICAL_GREEDY_MACRO_ACC - ttrv_macro_white),
                    "no",
                ],
            ],
        )
    )
    report.extend(
        [
            "",
            "The TTRV arithmetic difference is not a causal checkpoint-matched drop: its "
            "normal macro comes from the deleted historical two-episode run, while the white "
            "outputs come from the retained one-episode step10 checkpoints.",
            "",
            "## Interpretation constraints",
            "",
            "- A normal-to-white flip measures causal sensitivity to image replacement, but "
            "does not by itself prove use of the correct visual evidence.",
            "- Same-answer-and-both-correct is a conservative count of questions solvable "
            "without the original image; it can include question, option, dataset, and format priors.",
            "- White-image accuracy can rise or fall because blank images also change invalid-output "
            "behavior, so it is not simply the complement of visual grounding.",
            "",
            "## Audit paths",
            "",
            f"- Historical TTRV greedy launchers: {TTRV_GREEDY_LAUNCH_DIR}",
            f"- Cleanup manifest: {TTRV_CLEANUP_MANIFEST}",
            f"- Retained aggregate record: {TTRV_RETAINED_AGGREGATE}",
            f"- Summary JSON: {summary_path}",
            f"- Paired cases JSONL: {cases_path}",
        ]
    )
    report_path = args.output_dir / "blank_prior_diagnostics_recomputed.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)
    print(summary_path)
    print(cases_path)


if __name__ == "__main__":
    main()
