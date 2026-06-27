#!/usr/bin/env python
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CHOICES = {"A", "B", "C", "D", "E"}
GENERIC_EVIDENCE_MARKERS = {
    "",
    "...",
    "short visual clue",
    "visual clue",
    "visible information",
    "the image shows",
    "i can see",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _category(record: dict) -> str:
    extra = record.get("extra_info") or {}
    return str(extra.get("mme_category") or "unknown")


def _record_id(record: dict) -> str:
    extra = record.get("extra_info") or {}
    return str(extra.get("index") or extra.get("id") or record.get("index") or record.get("batch_index"))


def _parse_evidence_answer(text: str, variant: str) -> tuple[str, str, bool, bool, bool]:
    raw = text or ""
    if variant in {"p1", "p2", "p3"}:
        evidence_match = re.search(r"<evidence>(.*?)</evidence>", raw, flags=re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"<answer>\s*([A-E])\s*</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
        evidence = _norm(evidence_match.group(1)) if evidence_match else ""
        answer = answer_match.group(1).upper() if answer_match else ""
        tag_compliance = bool(evidence_match and answer_match)
    elif variant == "kv_short":
        evidence_match = re.search(r"(?:^|[\r\n])\s*Evidence\s*:\s*(.*?)(?=[\r\n]\s*Answer\s*:|$)", raw, flags=re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"(?:^|[\r\n])\s*Answer\s*:\s*([A-E])\b", raw, flags=re.IGNORECASE)
        evidence = _norm(evidence_match.group(1)) if evidence_match else ""
        answer = answer_match.group(1).upper() if answer_match else ""
        tag_compliance = bool(evidence_match and answer_match)
    elif variant == "ans_short":
        evidence_match = re.search(r"(?:^|[\r\n])\s*OBS\s*=\s*(.*?)(?=[\r\n]\s*ANS\s*=|$)", raw, flags=re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"(?:^|[\r\n])\s*ANS\s*=\s*([A-E])\b", raw, flags=re.IGNORECASE)
        evidence = _norm(evidence_match.group(1)) if evidence_match else ""
        answer = answer_match.group(1).upper() if answer_match else ""
        tag_compliance = bool(evidence_match and answer_match)
    elif variant == "json_short":
        evidence = ""
        answer = ""
        tag_compliance = False
        json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                evidence = _norm(payload.get("evidence", ""))
                answer = str(payload.get("answer", "")).strip().upper()[:1]
                tag_compliance = True
            except Exception:
                pass
    else:
        evidence = ""
        answer = ""
        tag_compliance = False
    evidence_valid = bool(evidence)
    answer_valid = answer in CHOICES
    return evidence, answer, evidence_valid, answer_valid, tag_compliance

def _is_generic_evidence(evidence: str) -> bool:
    clean = re.sub(r"[^a-z0-9 ]+", " ", evidence.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if clean in GENERIC_EVIDENCE_MARKERS:
        return True
    if len(clean.split()) <= 2:
        return True
    return False


def _parse_record(record: dict, variant: str) -> dict:
    raw = record.get("response_raw", "")
    gt = str(record.get("ground_truth", "")).strip().upper()
    if variant == "p0":
        answer = str(record.get("prediction", "")).strip().upper()
        answer_valid = answer in CHOICES
        evidence = ""
        evidence_valid = False
        tag_compliance = True
        correct = bool(record.get("correct", False))
    else:
        evidence, answer, evidence_valid, answer_valid, tag_compliance = _parse_evidence_answer(raw, variant)
        correct = bool(answer_valid and answer == gt)
    return {
        "id": _record_id(record),
        "category": _category(record),
        "prompt_variant": variant.upper(),
        "ground_truth": gt,
        "response_raw": raw,
        "evidence": evidence,
        "answer": answer if answer_valid else "",
        "answer_valid": answer_valid,
        "evidence_valid": evidence_valid,
        "tag_compliance": tag_compliance,
        "correct": correct,
        "response_token_len": int(record.get("response_token_len", 0) or 0),
        "generic_evidence": _is_generic_evidence(evidence) if variant != "p0" else False,
        "image_path": (record.get("extra_info") or {}).get("image_path"),
        "prompt": record.get("prompt"),
    }


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    by_category_total = Counter()
    by_category_correct = Counter()
    token_lens = [r["response_token_len"] for r in records]
    for record in records:
        by_category_total[record["category"]] += 1
        by_category_correct[record["category"]] += int(record["correct"])

    def ratio(key: str) -> float:
        return sum(int(bool(r[key])) for r in records) / n if n else 0.0

    return {
        "n": n,
        "answer_acc": ratio("correct"),
        "answer_parse_rate": ratio("answer_valid"),
        "evidence_parse_rate": ratio("evidence_valid"),
        "tag_compliance_rate": ratio("tag_compliance"),
        "avg_response_token_len": sum(token_lens) / n if n else 0.0,
        "overlong_rate_128": sum(int(t >= 128) for t in token_lens) / n if n else 0.0,
        "empty_evidence_rate": 1.0 - ratio("evidence_valid"),
        "generic_evidence_rate": ratio("generic_evidence"),
        "by_category": {
            cat: {
                "n": by_category_total[cat],
                "answer_acc": by_category_correct[cat] / by_category_total[cat],
            }
            for cat in sorted(by_category_total)
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _case_examples(parsed_by_variant: dict[str, list[dict]], limit: int) -> dict[str, list[dict]]:
    p0 = {record["id"]: record for record in parsed_by_variant.get("p0", [])}
    examples = defaultdict(list)
    for variant, records in parsed_by_variant.items():
        if variant == "p0":
            continue
        for record in records:
            base = p0.get(record["id"])
            if not base:
                continue
            payload = {
                "id": record["id"],
                "variant": variant.upper(),
                "category": record["category"],
                "ground_truth": record["ground_truth"],
                "base_answer": base["answer"],
                "base_correct": base["correct"],
                "evidence_answer": record["answer"],
                "evidence_correct": record["correct"],
                "evidence_valid": record["evidence_valid"],
                "answer_valid": record["answer_valid"],
                "evidence": record["evidence"],
                "base_response_raw": base["response_raw"],
                "evidence_response_raw": record["response_raw"],
                "image_path": record["image_path"],
            }
            if base["correct"] and not record["correct"]:
                examples["original_correct_evidence_wrong"].append(payload)
            if not base["correct"] and record["correct"]:
                examples["original_wrong_evidence_correct"].append(payload)
            if not record["answer_valid"] or not record["tag_compliance"]:
                examples["format_failure"].append(payload)
            if not record["evidence_valid"] or record["generic_evidence"]:
                examples["evidence_missing_or_generic"].append(payload)
    return {key: value[:limit] for key, value in sorted(examples.items())}


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and summarize MME20 evidence-reasoning pilot outputs.")
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--variants", nargs="+", default=["p0", "p1", "p2", "p3"])
    parser.add_argument("--raw-name", default="validation.jsonl")
    parser.add_argument("--case-limit", type=int, default=20)
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    parsed_by_variant: dict[str, list[dict]] = {}
    summary = {"exp_root": str(exp_root), "variants": {}}

    for variant in args.variants:
        run_dir = exp_root / variant
        raw_path = run_dir / args.raw_name
        parsed_path = run_dir / "parsed.jsonl"
        raw_records = _read_jsonl(raw_path)
        parsed = [_parse_record(record, variant) for record in raw_records]
        _write_jsonl(parsed_path, parsed)
        parsed_by_variant[variant] = parsed
        summary["variants"][variant] = {
            "raw_path": str(raw_path),
            "parsed_path": str(parsed_path),
            **_summarize(parsed),
        }

    summary["cases"] = _case_examples(parsed_by_variant, args.case_limit)
    summary_path = exp_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["# MME20 Evidence-Reasoning Pilot Summary", ""]
    lines.append("| variant | n | answer acc | answer parse | evidence parse | tag compliance | avg tokens | generic evidence |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant in args.variants:
        item = summary["variants"][variant]
        lines.append(
            "| {variant} | {n} | {acc} | {ap} | {ep} | {tc} | {tok:.1f} | {gen} |".format(
                variant=variant.upper(),
                n=item["n"],
                acc=_format_pct(item["answer_acc"]),
                ap=_format_pct(item["answer_parse_rate"]),
                ep=_format_pct(item["evidence_parse_rate"]),
                tc=_format_pct(item["tag_compliance_rate"]),
                tok=item["avg_response_token_len"],
                gen=_format_pct(item["generic_evidence_rate"]),
            )
        )
    lines.append("")
    lines.append("## Case Counts")
    for key, examples in summary["cases"].items():
        lines.append(f"- `{key}`: {len(examples)} shown")
    md_path = exp_root / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
