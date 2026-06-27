#!/usr/bin/env python
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CHOICES = ("A", "B", "C", "D")


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _prediction(record: dict) -> str:
    pred = str(record.get("prediction", "")).strip().upper()
    return pred if pred in CHOICES else "unknown"


def _is_correct(record: dict) -> bool:
    return bool(record.get("correct", False))


def _record_key(record: dict) -> str:
    extra = record.get("extra_info") or {}
    return str(record.get("index") or extra.get("index") or extra.get("id") or record.get("batch_index"))


def _category(record: dict) -> str:
    extra = record.get("extra_info") or {}
    return str(extra.get("mme_category") or "unknown")


def _option_map(prompt: str) -> dict[str, str]:
    options = {}
    for match in re.finditer(r"(?im)^\s*([A-D])\.\s*(.+?)(?=\s*$)", prompt or ""):
        options[match.group(1).upper()] = match.group(2).strip()
    if not options:
        flat = (prompt or "").replace("\n", " ")
        for match in re.finditer(r"\b([A-D])\.\s*([^A-D]+?)(?=\s+[A-D]\.|$)", flat):
            options[match.group(1).upper()] = match.group(2).strip()
    return options


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _semantic_choice(record: dict) -> str:
    response = _norm_text(record.get("response_raw", ""))
    options = _option_map(record.get("prompt", ""))
    hits = []
    for label, option_text in options.items():
        option_norm = _norm_text(option_text)
        if option_norm and re.search(rf"\b{re.escape(option_norm)}\b", response):
            hits.append(label)
    if len(hits) == 1:
        return hits[0]

    yes_no = {"yes": None, "no": None}
    for label, option_text in options.items():
        opt = _norm_text(option_text)
        if opt in yes_no:
            yes_no[opt] = label
    response_words = set(response.split())
    if yes_no["yes"] and yes_no["no"]:
        has_yes = "yes" in response_words
        has_no = "no" in response_words
        if has_yes != has_no:
            return yes_no["yes"] if has_yes else yes_no["no"]
    return "unknown"


def _split_validation(records: list[dict]) -> dict[int, list[dict]]:
    by_call = defaultdict(list)
    for record in records:
        by_call[int(record.get("eval_dump_call", 0))].append(record)
    return dict(by_call)


def _summarize_records(records: list[dict]) -> dict:
    category_total = Counter()
    category_correct = Counter()
    invalid = 0
    for record in records:
        cat = _category(record)
        category_total[cat] += 1
        category_correct[cat] += int(_is_correct(record))
        invalid += int(_prediction(record) == "unknown")
    return {
        "n": len(records),
        "accuracy": sum(category_correct.values()) / len(records) if records else 0.0,
        "invalid_ratio": invalid / len(records) if records else 0.0,
        "by_category": {
            cat: {
                "n": category_total[cat],
                "accuracy": category_correct[cat] / category_total[cat],
            }
            for cat in sorted(category_total)
        },
    }


def _run_dirs(exp_root: Path) -> list[Path]:
    if (exp_root / "validation_flat.jsonl").exists() or (exp_root / "validation_step0_and_final_flat.jsonl").exists():
        return [exp_root]
    return sorted(path for path in exp_root.iterdir() if path.is_dir())


def _load_run(run_dir: Path) -> dict:
    validation_path = run_dir / "validation_step0_and_final_flat.jsonl"
    if not validation_path.exists():
        validation_path = run_dir / "validation_flat.jsonl"
    validation_records = _read_jsonl(validation_path)
    by_call = _split_validation(validation_records)
    calls = sorted(by_call)
    final_call = calls[-1] if calls else None
    return {
        "name": run_dir.name,
        "dir": run_dir,
        "validation_path": validation_path,
        "validation_records": validation_records,
        "by_call": by_call,
        "step0": by_call.get(0, []),
        "final": by_call.get(final_call, []) if final_call is not None else [],
        "final_call": final_call,
        "train_rollouts": _read_jsonl(run_dir / "train_rollouts.jsonl"),
    }


def _index_records(records: list[dict]) -> dict[str, dict]:
    return {_record_key(record): record for record in records}


def _case_record(case_type: str, base: dict, final: dict, run_name: str) -> dict:
    return {
        "case_type": case_type,
        "run": run_name,
        "index": _record_key(base),
        "mme_category": _category(base),
        "ground_truth": base.get("ground_truth"),
        "base_prediction": _prediction(base),
        "base_correct": _is_correct(base),
        "base_semantic_choice": _semantic_choice(base),
        "base_response_raw": base.get("response_raw"),
        "final_prediction": _prediction(final),
        "final_correct": _is_correct(final),
        "final_response_raw": final.get("response_raw"),
        "image_path": (base.get("extra_info") or {}).get("image_path"),
        "prompt": base.get("prompt"),
    }


def _compare_cases(base_records: list[dict], final_records: list[dict], run_name: str, limit: int) -> dict[str, list[dict]]:
    base_by_key = _index_records(base_records)
    final_by_key = _index_records(final_records)
    cases = defaultdict(list)
    for key, base in base_by_key.items():
        final = final_by_key.get(key)
        if not final:
            continue
        base_correct = _is_correct(base)
        final_correct = _is_correct(final)
        base_pred = _prediction(base)
        final_pred = _prediction(final)
        semantic_choice = _semantic_choice(base)
        gt = str(base.get("ground_truth", "")).strip().upper()

        if not base_correct and final_correct:
            cases["base_wrong_to_final_correct"].append(_case_record("base_wrong_to_final_correct", base, final, run_name))
        if base_correct and not final_correct:
            cases["base_correct_to_final_wrong"].append(_case_record("base_correct_to_final_wrong", base, final, run_name))
        if not base_correct and not final_correct and base_pred != final_pred:
            cases["wrong_changed_wrong"].append(_case_record("wrong_changed_wrong", base, final, run_name))
        if not base_correct and not final_correct and base_pred == final_pred:
            cases["stable_wrong"].append(_case_record("stable_wrong", base, final, run_name))
        if semantic_choice == gt and base_pred != gt and final_correct:
            cases["parser_or_format_recovery_candidate"].append(
                _case_record("parser_or_format_recovery_candidate", base, final, run_name)
            )
    return {key: value[:limit] for key, value in sorted(cases.items())}


def _summarize_train_rollouts(records: list[dict]) -> dict:
    summary = {
        "n_groups": len(records),
        "majority_correct": 0,
        "majority_wrong": 0,
        "correct_present": 0,
        "correct_minority": 0,
        "harmony_label_correct": 0,
        "by_category": defaultdict(lambda: Counter()),
        "examples": defaultdict(list),
    }
    for record in records:
        gt = str(record.get("ground_truth", "")).strip().upper()
        majority = str(record.get("majority_prediction", "")).strip().upper()
        extra = record.get("extra_info") or {}
        cat = str(extra.get("mme_category") or "unknown")
        majority_correct = majority == gt
        correct_present = bool(record.get("correct_present", False))
        correct_minority = bool(record.get("correct_is_minority", False))
        summary["majority_correct"] += int(majority_correct)
        summary["majority_wrong"] += int(not majority_correct)
        summary["correct_present"] += int(correct_present)
        summary["correct_minority"] += int(correct_minority)
        if "harmony_label_correct" in record:
            summary["harmony_label_correct"] += int(bool(record.get("harmony_label_correct")))
        summary["by_category"][cat]["n"] += 1
        summary["by_category"][cat]["majority_correct"] += int(majority_correct)
        summary["by_category"][cat]["correct_minority"] += int(correct_minority)
        if not majority_correct and len(summary["examples"]["majority_wrong"]) < 10:
            summary["examples"]["majority_wrong"].append(record)
        if correct_minority and len(summary["examples"]["correct_minority"]) < 10:
            summary["examples"]["correct_minority"].append(record)
    summary["by_category"] = {cat: dict(counter) for cat, counter in sorted(summary["by_category"].items())}
    summary["examples"] = dict(summary["examples"])
    return summary


def _fmt_float(value: float) -> str:
    return f"{value:.4f}"


def _short(text: str | None, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _write_markdown(path: Path, runs: list[dict], base_run: dict | None, comparisons: dict, rollout_summaries: dict) -> None:
    lines = ["# MME20 TTRV Case Analysis", ""]
    lines.append("## Validation Summary")
    lines.append("")
    lines.append("| Run | eval calls | final call | final n | final acc | final invalid |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        final_summary = _summarize_records(run["final"])
        lines.append(
            f"| `{run['name']}` | {len(run['by_call'])} | {run['final_call']} | {final_summary['n']} | "
            f"{_fmt_float(final_summary['accuracy'])} | {_fmt_float(final_summary['invalid_ratio'])} |"
        )

    if base_run and base_run["step0"]:
        base_summary = _summarize_records(base_run["step0"])
        lines.extend(
            [
                "",
                "## Base Step 0",
                "",
                f"- Base run: `{base_run['name']}`",
                f"- n: {base_summary['n']}",
                f"- acc: {_fmt_float(base_summary['accuracy'])}",
                f"- invalid: {_fmt_float(base_summary['invalid_ratio'])}",
                "",
                "| MME category | n | acc |",
                "| --- | ---: | ---: |",
            ]
        )
        for cat, info in base_summary["by_category"].items():
            lines.append(f"| `{cat}` | {info['n']} | {_fmt_float(info['accuracy'])} |")

    lines.extend(["", "## Final By Category", ""])
    for run in runs:
        summary = _summarize_records(run["final"])
        lines.append(f"### {run['name']}")
        lines.append("")
        lines.append("| MME category | n | acc |")
        lines.append("| --- | ---: | ---: |")
        for cat, info in summary["by_category"].items():
            lines.append(f"| `{cat}` | {info['n']} | {_fmt_float(info['accuracy'])} |")
        lines.append("")

    lines.extend(["", "## Step 0 vs Final Case Buckets", ""])
    if comparisons:
        for run_name, buckets in comparisons.items():
            lines.append(f"### {run_name}")
            lines.append("")
            for bucket, examples in buckets.items():
                lines.append(f"- `{bucket}`: {len(examples)} examples shown")
                for example in examples[:5]:
                    lines.append(
                        f"  - `{example['index']}` `{example['mme_category']}` "
                        f"gt={example['ground_truth']} base={example['base_prediction']} "
                        f"semantic={example['base_semantic_choice']} final={example['final_prediction']}"
                    )
                    lines.append(f"    - base: {_short(example['base_response_raw'])}")
                    lines.append(f"    - final: {_short(example['final_response_raw'])}")
            lines.append("")
    else:
        lines.append("No step0/final comparison available. The base run must include `trainer.val_before_train=True`.")
        lines.append("")

    lines.extend(["", "## Train Rollout Pseudo-Label Summary", ""])
    for run_name, summary in rollout_summaries.items():
        n = summary["n_groups"]
        if not n:
            continue
        lines.append(f"### {run_name}")
        lines.append("")
        lines.append(f"- groups: {n}")
        lines.append(f"- majority_correct: {_fmt_float(summary['majority_correct'] / n)}")
        lines.append(f"- majority_wrong: {_fmt_float(summary['majority_wrong'] / n)}")
        lines.append(f"- correct_present: {_fmt_float(summary['correct_present'] / n)}")
        lines.append(f"- correct_minority: {_fmt_float(summary['correct_minority'] / n)}")
        if summary["harmony_label_correct"]:
            lines.append(f"- harmony_label_correct: {_fmt_float(summary['harmony_label_correct'] / n)}")
        lines.append("")
        lines.append("| MME category | n | majority_correct | correct_minority |")
        lines.append("| --- | ---: | ---: | ---: |")
        for cat, counter in summary["by_category"].items():
            cat_n = counter.get("n", 0)
            lines.append(
                f"| `{cat}` | {cat_n} | "
                f"{_fmt_float(counter.get('majority_correct', 0) / cat_n if cat_n else 0.0)} | "
                f"{_fmt_float(counter.get('correct_minority', 0) / cat_n if cat_n else 0.0)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `parser_or_format_recovery_candidate` is heuristic: it detects option text or yes/no semantics in the raw response when the parser did not return the ground-truth option.",
            "- Majority statistics come from training rollouts unless validation was run with `VAL_N>1` and group output enabled.",
            "- This report does not use ground truth to train; ground truth is used only for post-hoc analysis.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MME20 step0/final validation and rollout cases.")
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--base-run", default="mme20_ttrv_official")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--case-limit", type=int, default=25)
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    runs = [_load_run(path) for path in _run_dirs(exp_root)]
    runs = [run for run in runs if run["validation_records"] or run["train_rollouts"]]
    base_run = next((run for run in runs if run["name"] == args.base_run), None)
    if base_run is None and runs:
        base_run = runs[0]

    comparisons = {}
    if base_run and base_run["step0"]:
        for run in runs:
            if run["final"]:
                comparisons[run["name"]] = _compare_cases(
                    base_run["step0"],
                    run["final"],
                    run["name"],
                    limit=args.case_limit,
                )

    rollout_summaries = {run["name"]: _summarize_train_rollouts(run["train_rollouts"]) for run in runs}
    output_md = Path(args.output_md) if args.output_md else exp_root / "mme20_case_analysis.md"
    output_json = Path(args.output_json) if args.output_json else exp_root / "mme20_case_examples.json"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(output_md, runs, base_run, comparisons, rollout_summaries)
    output_json.write_text(
        json.dumps(
            {
                "base_run": base_run["name"] if base_run else None,
                "comparisons": comparisons,
                "rollout_summaries": rollout_summaries,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[analyze_mme20_cases] wrote {output_md}")
    print(f"[analyze_mme20_cases] wrote {output_json}")


if __name__ == "__main__":
    main()
