"""Join complete raw predictions to frozen official records and call pinned scorers."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def join_predictions(official, predictions, protocol_id):
    by_id = {}
    for record in predictions:
        ident = str(record.get("index"))
        if ident in by_id:
            raise ValueError(f"Duplicate prediction: {ident}; use one validation pass per file")
        if record.get("generation_status", "ok") != "ok":
            raise ValueError(f"Generation failure: {ident}")
        if "response_raw" not in record:
            raise ValueError(f"Missing response_raw: {ident}")
        if record.get("extra_info", {}).get("protocol_id") != protocol_id:
            raise ValueError(f"Prediction protocol mismatch: {ident}")
        by_id[ident] = record
    ids = [row["extra_info"]["index"] for row in official]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError(f"Prediction coverage mismatch: expected={len(ids)}, got={len(by_id)}, "
                         f"missing={sorted(set(ids)-set(by_id))[:10]}, extra={sorted(set(by_id)-set(ids))[:10]}")
    return [(row, by_id[row["extra_info"]["index"]]) for row in official]


def score(data_dir, predictions_path, output, method, run_id, checkpoint_id="in_trainer", smoke=False):
    import pandas as pd
    from scripts.official_data.upstream import module
    upstream = module("scoring")
    MME_rating, eval_vanilla = upstream.MME_rating, upstream.eval_vanilla
    extract_answer_from_item, report_acc = upstream.extract_answer_from_item, upstream.report_acc
    YOrN_Extraction = module("yesno").YOrN_Extraction
    data_dir, output = Path(data_dir), Path(output)
    lock = json.loads((data_dir / "protocol.lock.json").read_text())
    from scripts.official_data.prepare import canonical_hash
    if lock["dataset"] not in {"AI2D_TEST", "MME"} or lock["external_judge"]:
        raise ValueError("Unsupported scoring protocol")
    if canonical_hash({k: v for k, v in lock.items() if k != "protocol_id"}) != lock["protocol_id"]:
        raise ValueError("Protocol lock hash mismatch")
    official = read_jsonl(data_dir / "official_records.jsonl")
    if smoke:
        import pyarrow.parquet as pq
        smoke_ids = {r["extra_info"]["index"] for r in pq.read_table(data_dir / "smoke.parquet").to_pylist()}
        official = [r for r in official if r["extra_info"]["index"] in smoke_ids]
    pairs = join_predictions(official, read_jsonl(predictions_path), lock["protocol_id"])
    if any(original["extra_info"]["protocol_id"] != lock["protocol_id"] for original, _ in pairs):
        raise ValueError("Official metadata protocol mismatch")
    if lock["dataset"] == "MME":
        counts = Counter(row["extra_info"]["pair_id"] for row, _ in pairs)
        if any(count != 2 for count in counts.values()):
            raise ValueError("MME pair coverage failure")
    output.mkdir(parents=True, exist_ok=False)  # Never reuse an upstream result/cache.
    scored, audit = [], []
    for original, prediction in pairs:
        raw, info = original["raw"], original["extra_info"]
        # Only tokenizer-produced decoding is accepted. Never strip text heuristically.
        decoded = prediction.get("decoded_text", prediction["response_raw"])
        if not isinstance(decoded, str) or not isinstance(prediction["response_raw"], str):
            raise ValueError("Predictions must be strings, including empty invalid responses")
        item = dict(raw, prediction=decoded, GT=raw["answer"])
        if lock["dataset"] == "AI2D_TEST":
            extracted = extract_answer_from_item(None, item, dataset_name="AI2D_TEST")["opt"]
            result = eval_vanilla(None, item, dataset_name="AI2D_TEST")
            correct, invalid = bool(result["hit"]), extracted not in info["choice_labels"]
            item.update(result)
        else:
            extracted = YOrN_Extraction(decoded)
            # Exact upstream comparison; official raw labels are retained, not recoded.
            correct, invalid = extracted == raw["answer"], extracted == "Unknown"
            item["score"] = int(correct)
        scored.append(item)
        audit.append(dict(run_id=run_id, method=method, checkpoint_id=checkpoint_id,
                          protocol_id=lock["protocol_id"], official_index=raw["index"],
                          response_raw=prediction["response_raw"], decoded_text=decoded,
                          reward_prediction=prediction.get("prediction"), official_prediction=extracted,
                          official_correct=correct, invalid_official=invalid,
                          category=raw.get("category"), image_id=info["image_id"], generation_status="ok"))
    frame = pd.DataFrame(scored)
    scorer_input = output / "predictions.for_official_scorer.jsonl"
    frame.to_json(scorer_input, orient="records", lines=True, force_ascii=False)
    if lock["dataset"] == "AI2D_TEST":
        metrics = report_acc(frame).to_dict(orient="records")
    elif smoke:
        # Full MME_rating requires all 14 categories. Do not mislabel subset accuracy as MME total.
        metrics = [{"smoke_only_question_accuracy": float(frame["score"].mean())}]
    else:
        metrics = MME_rating(scorer_input).to_dict(orient="records")
        for metric in metrics:
            metric["total"] = metric["perception"] + metric["reasoning"]
    report = dict(count=len(audit), official_invalid_rate=sum(r["invalid_official"] for r in audit) / len(audit),
                  reward_invalid_rate=sum(str(r["reward_prediction"]).lower() in {"unknown", "", "none"} for r in audit) / len(audit),
                  generation_failure_rate=0.0, full_coverage=True, smoke_only=smoke,
                  protocol_id=lock["protocol_id"])
    for name, value in {"official_scores.json": metrics, "coverage_and_invalid.json": report}.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    with open(output / "predictions.audited.jsonl", "x") as stream:
        for row in audit:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["base", "ttrv", "prove"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(score(args.data_dir, args.predictions, args.output, args.method, args.run_id, smoke=args.smoke))


if __name__ == "__main__":
    main()
