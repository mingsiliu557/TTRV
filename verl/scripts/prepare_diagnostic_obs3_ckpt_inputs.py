#!/usr/bin/env python3
"""Prepare a protocol-locked real/shuffle subset for checkpoint Obs3 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


CLOSED_BENCHMARKS = (
    "mme20",
    "ai2d20",
    "realworldqa20_hq",
    "seed20",
    "mathvista20_hq",
    "mathverse20_fixed",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extra(row: pd.Series) -> dict[str, Any]:
    value = row.get("extra_info")
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected dict extra_info, got {type(value)!r}")


def _qid(row: pd.Series) -> str:
    extra = _extra(row)
    value = extra.get("diagnostic_question_id")
    if value is None:
        raise ValueError("Missing diagnostic_question_id")
    return str(value)


def _condition(row: pd.Series) -> str:
    value = _extra(row).get("diagnostic_condition")
    return "" if value is None else str(value)


def _shuffle_source(row: pd.Series) -> str:
    value = _extra(row).get("diagnostic_shuffle_src_qid")
    return "" if value is None else str(value)


def _ordered_rows(
    frame: pd.DataFrame,
    ids: list[str],
    condition: str,
    *,
    allow_extras: bool = False,
) -> pd.DataFrame:
    filtered = frame[frame.apply(_condition, axis=1) == condition].copy()
    by_id: dict[str, int] = {}
    for index, row in filtered.iterrows():
        qid = _qid(row)
        if qid in by_id:
            raise ValueError(f"Duplicate {condition} row for question_id={qid}")
        by_id[qid] = index
    missing = [qid for qid in ids if qid not in by_id]
    extras = sorted(set(by_id) - set(ids))
    if missing or (extras and not allow_extras):
        raise ValueError(
            f"{condition} ID mismatch: missing={missing[:5]} extras={extras[:5]}"
        )
    return filtered.loc[[by_id[qid] for qid in ids]].reset_index(drop=True)


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["benchmark"]: row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(CLOSED_BENCHMARKS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    unsupported = sorted(set(args.benchmarks) - set(CLOSED_BENCHMARKS))
    if unsupported:
        raise ValueError(f"Unsupported closed-choice benchmarks: {unsupported}")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    source_manifest_path = args.source_root / "run_manifest.tsv"
    source_manifest = _read_manifest(source_manifest_path)
    args.output_root.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "source_root": str(args.source_root),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "limit": args.limit,
        "benchmarks": {},
    }

    for key in args.benchmarks:
        if key not in source_manifest:
            raise ValueError(f"Benchmark {key!r} absent from source manifest")
        spec = source_manifest[key]
        source_subset = args.source_root / f"subset_{key}.json"
        source_pairing = args.source_root / f"shuffle_pairing_{key}.json"
        ids = [str(value) for value in _read_json(source_subset)]
        if args.limit:
            ids = ids[: args.limit]

        pairing_payload = _read_json(source_pairing)
        pair_by_target = {
            str(item["target_qid"]): str(item["shuffle_src_qid"])
            for item in pairing_payload["pairs"]
        }
        if any(qid not in pair_by_target for qid in ids):
            raise ValueError(f"{key}: selected IDs missing from shuffle pairing")

        real_source = Path(spec["real_parquet"])
        counterfactual_source = Path(spec["counterfactual_parquet"])
        allow_extras = bool(args.limit)
        real = _ordered_rows(
            pd.read_parquet(real_source), ids, "real", allow_extras=allow_extras
        )
        shuffle = _ordered_rows(
            pd.read_parquet(counterfactual_source),
            ids,
            "shuffle",
            allow_extras=allow_extras,
        )

        for _, row in shuffle.iterrows():
            qid = _qid(row)
            observed = _shuffle_source(row)
            expected = pair_by_target[qid]
            if observed != expected:
                raise ValueError(
                    f"{key}/{qid}: shuffle source mismatch observed={observed} expected={expected}"
                )

        prepared_dir = args.output_root / "prepared" / key
        prepared_dir.mkdir(parents=True, exist_ok=True)
        real_path = prepared_dir / "real.parquet"
        shuffle_path = prepared_dir / "shuffle.parquet"
        combined_path = prepared_dir / "real_shuffle.parquet"
        real.to_parquet(real_path, index=False)
        shuffle.to_parquet(shuffle_path, index=False)
        pd.concat([real, shuffle], ignore_index=True).to_parquet(combined_path, index=False)

        subset_target = args.output_root / f"subset_{key}.json"
        pairing_target = args.output_root / f"shuffle_pairing_{key}.json"
        if args.limit:
            _write_json(subset_target, ids)
            limited_pairing = {
                **pairing_payload,
                "pairs": [item for item in pairing_payload["pairs"] if str(item["target_qid"]) in set(ids)],
                "diagnostic_smoke_limit": args.limit,
            }
            _write_json(pairing_target, limited_pairing)
        else:
            shutil.copy2(source_subset, subset_target)
            shutil.copy2(source_pairing, pairing_target)

        output_rows.append(
            {
                "benchmark": key,
                "task": spec["task"],
                "parser": spec["parser"],
                "labels": spec["labels"],
                "train_parquet": spec["train_parquet"],
                "eval_parquet": str(combined_path),
                "expected_questions": len(ids),
            }
        )
        summary["benchmarks"][key] = {
            "questions": len(ids),
            "source_real_parquet": str(real_source),
            "source_shuffle_parquet": str(counterfactual_source),
            "real_parquet": str(real_path),
            "shuffle_parquet": str(shuffle_path),
            "combined_parquet": str(combined_path),
            "subset_sha256": _sha256(subset_target),
            "shuffle_pairing_sha256": _sha256(pairing_target),
            "real_parquet_sha256": _sha256(real_path),
            "shuffle_parquet_sha256": _sha256(shuffle_path),
        }

    manifest_path = args.output_root / "run_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    summary["output_manifest"] = str(manifest_path)
    summary["output_manifest_sha256"] = _sha256(manifest_path)
    _write_json(args.output_root / "protocol_input_integrity.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
