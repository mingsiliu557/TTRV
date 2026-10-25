#!/usr/bin/env python3
"""Compare base, TTRV, and ours under the same Obs3 real/shuffle protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


VERL_ROOT = Path(__file__).resolve().parents[1]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score.ttrl.ttt_metrics import (
    _extract_answers,
    _normalize_choice_answer,
)


BENCHMARKS = (
    ("mme20", "MME20", set("AB")),
    ("ai2d20", "AI2D20", set("ABCD")),
    ("realworldqa20_hq", "RealWorldQA-HQ", set("ABCD")),
    ("seed20", "SEED20", set("ABCD")),
    ("mathvista20_hq", "MathVista-HQ", set("ABCDEF")),
    ("mathverse20_fixed", "MathVerse-fixed", set("ABCDEF")),
)
MODELS = (
    ("base", "base(step0)"),
    ("ttrv", "TTRV step20"),
    ("ours", "ours step20"),
)
CONDITIONS = ("real", "shuffle")
PARSER_CONFIG = {
    "mme20": ("legacy", "AB"),
    "ai2d20": ("legacy", "A-D"),
    "realworldqa20_hq": ("legacy", "A-D"),
    "seed20": ("legacy", "A-D"),
    "mathvista20_hq": ("legacy", "A-F"),
    "mathverse20_fixed": ("mathvista_choice", "A-F"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--base-real-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_choice(value: Any) -> str:
    return str(value or "").strip().upper()


def load_condition(
    path: Path,
    benchmark: str,
    condition: str,
    labels: set[str],
    expected_qids: set[str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "counter": Counter(),
            "indices": set(),
            "ground_truth": None,
            "shuffle_src_qid": None,
        }
    )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                preview = repr(line[:160])
                raise ValueError(
                    f"{path}:{line_number}: malformed JSONL, preview={preview}"
                ) from exc
            if row.get("benchmark") != benchmark or row.get("condition") != condition:
                raise ValueError(f"{path}:{line_number}: benchmark/condition mismatch")
            qid = str(row["question_id"])
            if qid not in expected_qids:
                raise ValueError(f"{path}:{line_number}: unexpected question_id={qid}")
            group = groups[qid]
            sample_idx = int(row["sample_idx"])
            if sample_idx in group["indices"]:
                raise ValueError(f"{path}:{line_number}: duplicate sample_idx={sample_idx}")
            group["indices"].add(sample_idx)

            ground_truth = normalize_choice(row.get("ground_truth"))
            if ground_truth not in labels:
                raise ValueError(f"{path}:{line_number}: invalid ground truth={ground_truth!r}")
            if group["ground_truth"] is None:
                group["ground_truth"] = ground_truth
            elif group["ground_truth"] != ground_truth:
                raise ValueError(f"{path}:{line_number}: inconsistent ground truth")

            shuffle_src = row.get("shuffle_src_qid")
            shuffle_src = None if shuffle_src is None else str(shuffle_src)
            if condition == "shuffle":
                if group["shuffle_src_qid"] is None:
                    group["shuffle_src_qid"] = shuffle_src
                elif group["shuffle_src_qid"] != shuffle_src:
                    raise ValueError(f"{path}:{line_number}: inconsistent shuffle source")
            elif shuffle_src is not None:
                raise ValueError(f"{path}:{line_number}: real record has shuffle source")

            if bool(row.get("is_valid")):
                answer = normalize_choice(row.get("parsed_answer"))
                if answer not in labels:
                    raise ValueError(
                        f"{path}:{line_number}: is_valid=true but parsed_answer={answer!r}"
                    )
                group["counter"][answer] += 1

    if set(groups) != expected_qids:
        raise ValueError(
            f"{path}: question set mismatch; missing={sorted(expected_qids-set(groups))[:5]}, "
            f"extra={sorted(set(groups)-expected_qids)[:5]}"
        )

    summaries: dict[str, dict[str, Any]] = {}
    for qid, group in groups.items():
        if group["indices"] != set(range(32)):
            raise ValueError(f"{path}:{qid}: sample indices are not exactly 0..31")
        counter = group["counter"]
        if counter:
            majority_answer, majority_count = counter.most_common(1)[0]
            max_count = max(counter.values())
            tie = int(sum(value == max_count for value in counter.values()) > 1)
        else:
            majority_answer, majority_count, tie = None, 0, 0
        n_valid = sum(counter.values())
        summaries[qid] = {
            "ground_truth": group["ground_truth"],
            "n_valid": n_valid,
            "n_invalid": 32 - n_valid,
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "majority_tie": tie,
            "consensus_x": majority_count / n_valid if n_valid else None,
            "shuffle_src_qid": group["shuffle_src_qid"],
        }
    return summaries


def load_group_condition(
    path: Path,
    benchmark: str,
    condition: str,
    labels: set[str],
    parser_mode: str,
    choice_labels: str,
    expected_qids: set[str],
) -> dict[str, dict[str, Any]]:
    """Replay the official parser over raw grouped responses.

    Some large finalized JSONL files on the network disk contain sparse NUL
    regions. Replaying the same parser used by finalize_diagnostic_stage_a.py
    avoids trusting either those corrupt files or the legacy invalid_count.
    """
    summaries: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                group = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed grouped JSONL") from exc
            extra = group.get("extra_info")
            if not isinstance(extra, dict):
                raise ValueError(f"{path}:{line_number}: missing extra_info")
            observed_condition = str(extra.get("diagnostic_condition") or "")
            if observed_condition != condition:
                continue
            if str(extra.get("diagnostic_benchmark")) != benchmark:
                raise ValueError(f"{path}:{line_number}: benchmark mismatch")
            qid = str(extra.get("diagnostic_question_id") or "")
            if qid not in expected_qids:
                raise ValueError(f"{path}:{line_number}: unexpected question_id={qid}")
            if qid in summaries:
                raise ValueError(f"{path}:{line_number}: duplicate question_id={qid}")
            samples = group.get("samples")
            if not isinstance(samples, list) or len(samples) != 32:
                raise ValueError(f"{path}:{line_number}: expected 32 samples")
            ground_truth_raw = group.get("ground_truth")
            ground_truth = _normalize_choice_answer(
                ground_truth_raw, choice_labels=choice_labels
            )
            if ground_truth not in labels:
                raise ValueError(
                    f"{path}:{line_number}: invalid normalized ground truth={ground_truth!r}"
                )
            raw_texts = [str(sample.get("response_raw") or "") for sample in samples]
            parsed_answers = _extract_answers(
                "gpqa",
                raw_texts,
                ground_truth_raw,
                extra_info=[extra] * 32,
                answer_parse_mode=parser_mode,
                choice_labels=choice_labels,
            )
            counter: Counter[str] = Counter()
            for parsed in parsed_answers:
                normalized = _normalize_choice_answer(parsed, choice_labels=choice_labels)
                if normalized != "unknown":
                    if normalized not in labels:
                        raise ValueError(
                            f"{path}:{line_number}: parser returned invalid label={normalized!r}"
                        )
                    counter[normalized] += 1
            if counter:
                majority_answer, majority_count = counter.most_common(1)[0]
                max_count = max(counter.values())
                tie = int(sum(value == max_count for value in counter.values()) > 1)
            else:
                majority_answer, majority_count, tie = None, 0, 0
            n_valid = sum(counter.values())
            shuffle_src = extra.get("diagnostic_shuffle_src_qid")
            shuffle_src = None if shuffle_src is None else str(shuffle_src)
            if condition == "real" and shuffle_src is not None:
                raise ValueError(f"{path}:{line_number}: real group has shuffle source")
            summaries[qid] = {
                "ground_truth": ground_truth,
                "n_valid": n_valid,
                "n_invalid": 32 - n_valid,
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "majority_tie": tie,
                "consensus_x": majority_count / n_valid if n_valid else None,
                "shuffle_src_qid": shuffle_src,
            }
    if set(summaries) != expected_qids:
        raise ValueError(
            f"{path}/{condition}: question set mismatch; "
            f"missing={sorted(expected_qids-set(summaries))[:5]}, "
            f"extra={sorted(set(summaries)-expected_qids)[:5]}"
        )
    return summaries


def group_path(
    base_root: Path,
    base_real_source: Path,
    comparison_root: Path,
    model: str,
    benchmark: str,
    condition: str,
) -> Path:
    if model == "base":
        if condition == "real":
            return base_real_source / "raw" / benchmark / "real" / "selection_groups.jsonl"
        return base_root / "raw" / benchmark / "counterfactual" / "selection_groups.jsonl"
    return comparison_root / "models" / model / "raw" / benchmark / "selection_groups.jsonl"


def dump_root(base_root: Path, comparison_root: Path, model: str) -> Path:
    if model == "base":
        return base_root / "dumps"
    return comparison_root / "models" / model / "dumps"


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def pp(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:+.2f}"


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| ckpt | n | Acc_real | Acc_shuf | Delta acc (pp) | Invariant rate | mean x | x=1 rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model_display} | {n} | {acc_real} | {acc_shuffle} | {delta} | "
            "{invariant} | {mean_x} | {x_eq_1} |".format(
                model_display=row["model_display"],
                n=row["n"],
                acc_real=pct(row["acc_real"]),
                acc_shuffle=pct(row["acc_shuffle"]),
                delta=pp(row["delta_acc"]),
                invariant=pct(row["invariant_rate"]),
                mean_x=f"{row['mean_x']:.4f}",
                x_eq_1=pct(row["x_eq_1_rate"]),
            )
        )
    return lines


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_display = dict(MODELS)
    all_groups: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    integrity_rows: list[dict[str, Any]] = []

    for benchmark, _, labels in BENCHMARKS:
        base_subset = [str(value) for value in read_json(args.base_root / f"subset_{benchmark}.json")]
        comparison_subset = [
            str(value) for value in read_json(args.comparison_root / f"subset_{benchmark}.json")
        ]
        if base_subset != comparison_subset:
            raise ValueError(f"{benchmark}: ordered subset mismatch")
        base_pairing = read_json(args.base_root / f"shuffle_pairing_{benchmark}.json")
        comparison_pairing = read_json(
            args.comparison_root / f"shuffle_pairing_{benchmark}.json"
        )
        if base_pairing != comparison_pairing:
            raise ValueError(f"{benchmark}: shuffle pairing mismatch")
        pair_map = {
            str(row["target_qid"]): str(row["shuffle_src_qid"])
            for row in base_pairing["pairs"]
        }
        expected_qids = set(base_subset)
        all_groups[benchmark] = {}
        for model, _ in MODELS:
            all_groups[benchmark][model] = {}
            for condition in CONDITIONS:
                path = group_path(
                    args.base_root,
                    args.base_real_source,
                    args.comparison_root,
                    model,
                    benchmark,
                    condition,
                )
                parser_mode, choice_labels = PARSER_CONFIG[benchmark]
                summaries = load_group_condition(
                    path,
                    benchmark,
                    condition,
                    labels,
                    parser_mode,
                    choice_labels,
                    expected_qids,
                )
                if condition == "shuffle":
                    mismatch = [
                        qid
                        for qid, summary in summaries.items()
                        if summary["shuffle_src_qid"] != pair_map[qid]
                    ]
                    if mismatch:
                        raise ValueError(
                            f"{benchmark}/{model}: shuffle source mismatch for {mismatch[:5]}"
                        )
                all_groups[benchmark][model][condition] = summaries
                integrity_rows.append(
                    {
                        "benchmark": benchmark,
                        "model": model,
                        "condition": condition,
                        "questions": len(summaries),
                        "samples": 32 * len(summaries),
                        "all_invalid_questions": sum(
                            row["n_valid"] == 0 for row in summaries.values()
                        ),
                    }
                )

    question_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    included_qids_by_benchmark: dict[str, list[str]] = {}

    for benchmark, benchmark_display, _ in BENCHMARKS:
        expected_qids = set(all_groups[benchmark]["base"]["real"])
        common_real_valid = {
            qid
            for qid in expected_qids
            if all(
                all_groups[benchmark][model]["real"][qid]["n_valid"] > 0
                for model, _ in MODELS
            )
        }
        included_qids_by_benchmark[benchmark] = sorted(common_real_valid)
        for model, _ in MODELS:
            real = all_groups[benchmark][model]["real"]
            shuffle = all_groups[benchmark][model]["shuffle"]
            invalid_rows.append(
                {
                    "benchmark": benchmark,
                    "benchmark_display": benchmark_display,
                    "model": model,
                    "model_display": model_display[model],
                    "full_question_n": len(expected_qids),
                    "common_real_valid_n": len(common_real_valid),
                    "union_real_all_invalid_excluded_n": len(expected_qids - common_real_valid),
                    "model_real_all_invalid_n": sum(row["n_valid"] == 0 for row in real.values()),
                    "model_shuffle_all_invalid_full_n": sum(
                        row["n_valid"] == 0 for row in shuffle.values()
                    ),
                    "model_shuffle_all_invalid_included_n": sum(
                        shuffle[qid]["n_valid"] == 0 for qid in common_real_valid
                    ),
                }
            )
            for qid in sorted(expected_qids):
                real_row = real[qid]
                shuffle_row = shuffle[qid]
                if real_row["ground_truth"] != shuffle_row["ground_truth"]:
                    raise ValueError(f"{benchmark}/{model}/{qid}: ground truth mismatch")
                included = qid in common_real_valid
                majority_real = real_row["majority_answer"]
                majority_shuffle = shuffle_row["majority_answer"]
                correct_real = (
                    int(majority_real == real_row["ground_truth"])
                    if majority_real is not None
                    else 0
                )
                correct_shuffle = (
                    int(majority_shuffle == real_row["ground_truth"])
                    if majority_shuffle is not None
                    else 0
                )
                invariant = int(
                    majority_shuffle is not None
                    and majority_real is not None
                    and majority_real == majority_shuffle
                )
                question_rows.append(
                    {
                        "benchmark": benchmark,
                        "benchmark_display": benchmark_display,
                        "model": model,
                        "model_display": model_display[model],
                        "question_id": qid,
                        "included_common_real_valid": int(included),
                        "ground_truth": real_row["ground_truth"],
                        "n_valid_real": real_row["n_valid"],
                        "n_invalid_real": real_row["n_invalid"],
                        "majority_real": majority_real,
                        "majority_count_real": real_row["majority_count"],
                        "majority_tie_real": real_row["majority_tie"],
                        "consensus_x": real_row["consensus_x"],
                        "correct_real": correct_real,
                        "n_valid_shuffle": shuffle_row["n_valid"],
                        "n_invalid_shuffle": shuffle_row["n_invalid"],
                        "majority_shuffle": majority_shuffle,
                        "majority_count_shuffle": shuffle_row["majority_count"],
                        "majority_tie_shuffle": shuffle_row["majority_tie"],
                        "correct_shuffle": correct_shuffle,
                        "shuffle_all_invalid": int(majority_shuffle is None),
                        "invariant": invariant,
                        "shuffle_src_qid": shuffle_row["shuffle_src_qid"],
                    }
                )

            selected = [
                row
                for row in question_rows
                if row["benchmark"] == benchmark
                and row["model"] == model
                and row["included_common_real_valid"] == 1
            ]
            n = len(selected)
            if n != len(common_real_valid) or n == 0:
                raise ValueError(f"{benchmark}/{model}: invalid selected n={n}")
            acc_real = sum(row["correct_real"] for row in selected) / n
            acc_shuffle = sum(row["correct_shuffle"] for row in selected) / n
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "benchmark_display": benchmark_display,
                    "model": model,
                    "model_display": model_display[model],
                    "n": n,
                    "acc_real": acc_real,
                    "acc_shuffle": acc_shuffle,
                    "delta_acc": acc_shuffle - acc_real,
                    "invariant_rate": sum(row["invariant"] for row in selected) / n,
                    "mean_x": fmean(float(row["consensus_x"]) for row in selected),
                    "x_eq_1_rate": sum(
                        abs(float(row["consensus_x"]) - 1.0) < 1e-12 for row in selected
                    )
                    / n,
                    "shuffle_all_invalid_n": sum(row["shuffle_all_invalid"] for row in selected),
                }
            )

    macro_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    metric_keys = (
        "acc_real",
        "acc_shuffle",
        "delta_acc",
        "invariant_rate",
        "mean_x",
        "x_eq_1_rate",
    )
    for model, _ in MODELS:
        rows = [row for row in summary_rows if row["model"] == model]
        macro_rows.append(
            {
                "model": model,
                "model_display": model_display[model],
                "n": sum(int(row["n"]) for row in rows),
                **{key: fmean(float(row[key]) for row in rows) for key in metric_keys},
            }
        )
        selected = [
            row
            for row in question_rows
            if row["model"] == model and row["included_common_real_valid"] == 1
        ]
        n = len(selected)
        acc_real = sum(row["correct_real"] for row in selected) / n
        acc_shuffle = sum(row["correct_shuffle"] for row in selected) / n
        micro_rows.append(
            {
                "model": model,
                "model_display": model_display[model],
                "n": n,
                "acc_real": acc_real,
                "acc_shuffle": acc_shuffle,
                "delta_acc": acc_shuffle - acc_real,
                "invariant_rate": sum(row["invariant"] for row in selected) / n,
                "mean_x": fmean(float(row["consensus_x"]) for row in selected),
                "x_eq_1_rate": sum(
                    abs(float(row["consensus_x"]) - 1.0) < 1e-12 for row in selected
                )
                / n,
            }
        )

    write_csv(args.output_dir / "obs3_question_level.csv", question_rows)
    write_csv(args.output_dir / "obs3_dataset_summary.csv", summary_rows)
    write_csv(args.output_dir / "obs3_invalid_diagnostics.csv", invalid_rows)
    write_csv(args.output_dir / "obs3_integrity.csv", integrity_rows)

    markdown = [
        "# Obs3 checkpoint comparison",
        "",
        "## Protocol",
        "",
        "- Scope: six closed-choice benchmarks shared by base, TTRV step20, and ours step20.",
        "- Each real and shuffled-image condition uses N=32 with the same ordered question subset and shuffle pairing.",
        "- Majority and consensus use only responses accepted by an exact replay of the official finalizer parser; this reproduces finalized `is_valid=true` without trusting corrupt sparse dump files.",
        "- The headline denominator is the cross-checkpoint common real-valid intersection. If any checkpoint is real all-invalid, that question is removed from all three rows.",
        "- A shuffle all-invalid question remains in the denominator and is assigned `correct_shuffle=0` and `invariant=0`.",
        "- `Delta acc = Acc_shuf - Acc_real` in percentage points.",
        "",
    ]
    for benchmark, benchmark_display, _ in BENCHMARKS:
        rows = [row for row in summary_rows if row["benchmark"] == benchmark]
        diagnostics = [row for row in invalid_rows if row["benchmark"] == benchmark]
        markdown.extend(
            [
                f"## {benchmark_display}",
                "",
                f"Common real-valid n: **{rows[0]['n']} / {diagnostics[0]['full_question_n']}**; "
                f"union-excluded real all-invalid: **{diagnostics[0]['union_real_all_invalid_excluded_n']}**.",
                "",
                *markdown_table(rows),
                "",
                "| ckpt | real all-invalid | shuffle all-invalid (full/common) |",
                "|---|---:|---:|",
            ]
        )
        for row in diagnostics:
            markdown.append(
                f"| {row['model_display']} | {row['model_real_all_invalid_n']} | "
                f"{row['model_shuffle_all_invalid_full_n']} / "
                f"{row['model_shuffle_all_invalid_included_n']} |"
            )
        markdown.append("")

    markdown.extend(
        [
            "## Average performance",
            "",
            "Macro average gives each benchmark equal weight. `n` is the total number of common-real-valid question rows.",
            "",
            *markdown_table(macro_rows),
            "",
            "## Pooled micro performance",
            "",
            *markdown_table(micro_rows),
            "",
        ]
    )
    (args.output_dir / "obs3_ckpt_comparison.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    summary = {
        "base_root": str(args.base_root),
        "comparison_root": str(args.comparison_root),
        "base_real_source": str(args.base_real_source),
        "validity_source": "official parser replay from raw selection_groups",
        "output_dir": str(args.output_dir),
        "included_question_ids": included_qids_by_benchmark,
        "dataset_summary": summary_rows,
        "invalid_diagnostics": invalid_rows,
        "macro_average": macro_rows,
        "micro_average": micro_rows,
    }
    (args.output_dir / "obs3_ckpt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "question_rows": len(question_rows),
                "dataset_rows": len(summary_rows),
                "common_question_total": sum(
                    len(qids) for qids in included_qids_by_benchmark.values()
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
