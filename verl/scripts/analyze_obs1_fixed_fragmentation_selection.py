#!/usr/bin/env python3
"""Compare string majority with density selection under fixed Obs1 fragmentation bins.

The script never regenerates responses. It reconstructs the exact DeBERTa
bidirectional-entailment clusters from the cached Obs1 pair predictions, fixes a
question-to-bin mapping, and replays the repository evidence-density scoring on
the same step-0 N=32 candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VERL_ROOT = REPO_ROOT / "verl"
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, VERL_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from attention_visualization.attention_viz import _prepare_inputs
from replay_obs2_density_selection import (
    append_jsonl,
    atomic_csv,
    atomic_json,
    embed_queries,
    load_model,
    normalized_response_text,
    read_jsonl,
)
from verl.trainer.ppo.ray_trainer import _DENSITY_EVIDENCE_DEFAULT_TEMPLATE
from verl.utils.reward_score.ttrl.direct_answer import UNKNOWN_LABEL, normalize_direct_answer
from verl.utils.reward_score.ttrl.ttt_metrics import (
    _density_entropy,
    _density_masses_from_embeddings,
    _density_temperature_from_mode,
    _normalized_entropy,
)


BENCHMARK_ORDER = ("visualsimpleqa", "ocrbench_v1", "aokvqa_da_val")
BIN_ORDER = ("no_frag", "low_frag", "high_frag")
K_BIN_ORDER = ("k1", "k2", "k3plus")
EXPECTED_QUESTIONS = 250
EXPECTED_SAMPLES = 32
T0 = 0.20
T_MIN = 0.10
T_MAX = 0.30
EMBEDDING_SCOPE = "evidence_query_mean_pool"
TEMPERATURE_MODE = "answer_entropy"


def canonical_answer(record: dict[str, Any]) -> str:
    if not bool(record.get("is_valid")):
        return UNKNOWN_LABEL
    answer = normalize_direct_answer(record.get("parsed_answer"))
    return answer if answer != UNKNOWN_LABEL else UNKNOWN_LABEL


def load_dump_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keep = {
        "benchmark",
        "question_id",
        "sample_idx",
        "subset_position",
        "parsed_answer",
        "is_valid",
        "is_correct",
        "verifier_reward",
        "ground_truth",
        "official_answers",
        "raw_text",
    }
    for raw in read_jsonl(path):
        grouped[str(raw["question_id"])].append({key: raw.get(key) for key in keep})
    if len(grouped) != EXPECTED_QUESTIONS:
        raise ValueError(f"{path}: expected {EXPECTED_QUESTIONS} questions, got {len(grouped)}")
    for qid, records in grouped.items():
        records.sort(key=lambda row: int(row["sample_idx"]))
        indices = [int(row["sample_idx"]) for row in records]
        if indices != list(range(EXPECTED_SAMPLES)):
            raise ValueError(f"{path}/{qid}: expected sample_idx=0..31, got {indices}")
    return grouped


def load_nli_predictions(path: Path) -> dict[tuple[str, str], bool]:
    predictions: dict[tuple[str, str], bool] = {}
    for row in read_jsonl(path):
        key = (str(row["premise"]), str(row["hypothesis"]))
        value = bool(row["is_entailment"])
        if key in predictions and predictions[key] != value:
            raise ValueError(f"Conflicting NLI judgment for {key}")
        predictions[key] = value
    if not predictions:
        raise ValueError(f"Empty NLI cache: {path}")
    return predictions


def load_obs1_question_cache(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            benchmark = str(row["benchmark"])
            if benchmark not in BENCHMARK_ORDER:
                continue
            key = (benchmark, str(row["question_id"]))
            if key in result:
                raise ValueError(f"Duplicate Obs1 cache key: {key}")
            result[key] = row
    if len(result) != len(BENCHMARK_ORDER) * EXPECTED_QUESTIONS:
        raise ValueError(f"Expected 750 open-ended Obs1 cache rows, got {len(result)}")
    return result


def cluster_answers(
    answers: list[str],
    counter: Counter[str],
    nli_predictions: dict[tuple[str, str], bool],
) -> list[list[str]]:
    def equivalent(left: str, right: str) -> bool:
        if left == right:
            return True
        if UNKNOWN_LABEL in {left, right}:
            return False
        forward = (left, right)
        reverse = (right, left)
        if forward not in nli_predictions or reverse not in nli_predictions:
            raise KeyError(f"Missing cached bidirectional NLI pair: {left!r} <-> {right!r}")
        return bool(nli_predictions[forward] and nli_predictions[reverse])

    clusters: list[list[str]] = []
    for answer in answers:
        for cluster in clusters:
            if equivalent(answer, cluster[0]):
                cluster.append(answer)
                break
        else:
            clusters.append([answer])
    if sum(sum(counter[answer] for answer in cluster) for cluster in clusters) != EXPECTED_SAMPLES:
        raise AssertionError("Semantic-cluster masses do not sum to 32")
    return clusters


def int_or_none(value: Any) -> int | None:
    text = str(value or "").strip()
    return None if not text else int(float(text))


def build_fixed_map(
    dump_root: Path,
    nli_predictions: dict[tuple[str, str], bool],
    obs1_cache: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    provisional: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_ORDER:
        groups = load_dump_groups(dump_root / benchmark / "real.jsonl")
        qids = sorted(groups, key=lambda qid: int(groups[qid][0]["subset_position"]))
        for qid in qids:
            records = groups[qid]
            canonical = [canonical_answer(record) for record in records]
            counter = Counter(canonical)
            answers = list(counter)
            clusters = cluster_answers(answers, counter, nli_predictions)
            invalid_count = sum(not bool(record.get("is_valid")) for record in records)
            answer_correct: dict[str, bool] = {}
            for answer, record in zip(canonical, records):
                answer_correct[answer] = bool(answer_correct.get(answer, False) or bool(record.get("is_correct")))
            cluster_rows = []
            correct_clusters = []
            for cluster_index, cluster in enumerate(clusters):
                mass = sum(counter[answer] for answer in cluster)
                row = {
                    "cluster_index": cluster_index,
                    "answers": cluster,
                    "mass": mass,
                    "contains_evaluator_correct": any(answer_correct.get(answer, False) for answer in cluster),
                }
                cluster_rows.append(row)
                if row["contains_evaluator_correct"]:
                    correct_clusters.append(row)
            correct_clusters.sort(key=lambda row: (-int(row["mass"]), int(row["cluster_index"])))
            selected_correct = correct_clusters[0] if correct_clusters else None

            valid_answers = [answer for answer in answers if answer != UNKNOWN_LABEL]
            valid_clusters = [cluster for cluster in clusters if UNKNOWN_LABEL not in cluster]
            cache = obs1_cache[(benchmark, qid)]
            cached_n_valid = int(cache["n_valid"])
            n_valid = EXPECTED_SAMPLES - invalid_count
            if cached_n_valid != n_valid:
                raise ValueError(f"{benchmark}/{qid}: n_valid drift {n_valid} vs {cached_n_valid}")
            cached_string = int_or_none(cache.get("string_cluster_count"))
            cached_semantic = int_or_none(cache.get("semantic_cluster_count"))
            if n_valid == 0:
                if cached_string is not None or cached_semantic is not None:
                    raise ValueError(f"{benchmark}/{qid}: expected blank all-invalid cache counts")
            elif cached_string != len(valid_answers) or cached_semantic != len(valid_clusters):
                raise ValueError(
                    f"{benchmark}/{qid}: DeBERTa cluster drift strings={len(valid_answers)}/{cached_string} "
                    f"semantic={len(valid_clusters)}/{cached_semantic}"
                )

            string_count = len(counter)
            semantic_count = len(clusters)
            fragmentation = string_count / semantic_count
            provisional.append(
                {
                    "benchmark": benchmark,
                    "question_id": qid,
                    "subset_position": int(records[0]["subset_position"]),
                    "n_total": EXPECTED_SAMPLES,
                    "n_valid": n_valid,
                    "n_invalid": invalid_count,
                    "string_cluster_count": string_count,
                    "semantic_cluster_count": semantic_count,
                    "fragmentation_ratio": fragmentation,
                    "string_counter": dict(counter),
                    "semantic_clusters": cluster_rows,
                    "correct_semantic_cluster_count": len(correct_clusters),
                    "correct_cluster_k": len(selected_correct["answers"]) if selected_correct else None,
                    "correct_cluster_answers": selected_correct["answers"] if selected_correct else [],
                    "correct_cluster_mass": int(selected_correct["mass"]) if selected_correct else None,
                }
            )

    fragmented_values = [float(row["fragmentation_ratio"]) for row in provisional if row["string_cluster_count"] > row["semantic_cluster_count"]]
    if not fragmented_values:
        raise ValueError("No fragmented questions found")
    median = float(statistics.median(fragmented_values))
    for row in provisional:
        if row["string_cluster_count"] == row["semantic_cluster_count"]:
            label = "no_frag"
        elif float(row["fragmentation_ratio"]) <= median:
            label = "low_frag"
        else:
            label = "high_frag"
        row["frag_bin"] = label
        row["fragmentation_median_over_f_gt_1"] = median
    counts = Counter(row["frag_bin"] for row in provisional)
    if len(provisional) != len(BENCHMARK_ORDER) * EXPECTED_QUESTIONS:
        raise AssertionError(f"Expected 750 fixed-map rows, got {len(provisional)}")
    if sum(counts.values()) != 750:
        raise AssertionError(f"Invalid fixed-bin counts: {counts}")
    return provisional


def serializable_map_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                **row,
                "string_counter": json.dumps(row["string_counter"], ensure_ascii=False, sort_keys=True),
                "semantic_clusters": json.dumps(row["semantic_clusters"], ensure_ascii=False),
                "correct_cluster_answers": json.dumps(row["correct_cluster_answers"], ensure_ascii=False),
            }
        )
    return output


def load_manifest(path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            benchmark = str(row.get("benchmark") or "")
            parquet = str(row.get("real_parquet") or "")
            if benchmark in BENCHMARK_ORDER and parquet:
                result[benchmark] = Path(parquet)
    missing = set(BENCHMARK_ORDER) - set(result)
    if missing:
        raise ValueError(f"Manifest missing open-ended parquet paths: {sorted(missing)}")
    return result


def load_parquet_rows(path: Path) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    result: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        extra = row.get("extra_info")
        if not isinstance(extra, dict):
            continue
        qid = str(extra.get("diagnostic_question_id") or "")
        if not qid:
            continue
        if qid in result:
            raise ValueError(f"Duplicate diagnostic_question_id={qid} in {path}")
        result[qid] = row
    return result


def evidence_query(response_text: str) -> str:
    return _DENSITY_EVIDENCE_DEFAULT_TEMPLATE.format(
        response=response_text,
        answer=response_text,
        candidate=response_text,
        option_label="unknown",
        option_text="",
    )


def all_output_density_rewards(
    canonical_answers: list[str],
    embeddings: np.ndarray,
) -> tuple[list[float], dict[str, Any]]:
    if len(canonical_answers) != EXPECTED_SAMPLES or embeddings.shape[0] != EXPECTED_SAMPLES:
        raise ValueError("All-output density expects exactly 32 candidates")
    counter = Counter(canonical_answers)
    answer_entropy = _normalized_entropy(counter, EXPECTED_SAMPLES)
    fixed_masses, _, _, _ = _density_masses_from_embeddings(embeddings, T0)
    fixed_density_entropy = _density_entropy(fixed_masses)
    temperature = _density_temperature_from_mode(
        TEMPERATURE_MODE,
        fixed_temperature=T0,
        t_min=T_MIN,
        t_max=T_MAX,
        answer_entropy=answer_entropy,
        density_entropy=fixed_density_entropy,
    )
    masses, _, sim_mean, sim_std = _density_masses_from_embeddings(embeddings, temperature)
    cluster_masses = {answer: 0.0 for answer in counter}
    for answer, mass in zip(canonical_answers, masses):
        cluster_masses[answer] += float(mass)
    rewards = [float(cluster_masses[answer]) for answer in canonical_answers]
    return rewards, {
        "answer_entropy_all_outputs": float(answer_entropy),
        "density_temperature": float(temperature),
        "density_entropy_fixed": float(fixed_density_entropy),
        "density_entropy_adaptive": float(_density_entropy(masses)),
        "density_sim_mean": float(sim_mean),
        "density_sim_std": float(sim_std),
        "density_candidate_masses": [float(value) for value in masses],
        "density_cluster_masses": cluster_masses,
    }


def process_question(
    benchmark: str,
    qid: str,
    records: list[dict[str, Any]],
    parquet_row: dict[str, Any],
    fixed: dict[str, Any],
    model: Any,
    tokenizer: Any,
    processor: Any,
    device: str,
    micro_batch_size: int,
) -> dict[str, Any]:
    canonical = [canonical_answer(record) for record in records]
    counter = Counter(canonical)
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_index = canonical.index(majority_answer)
    answer_correct: dict[str, bool] = {}
    answer_verifier: dict[str, float] = {}
    for answer, record in zip(canonical, records):
        answer_correct[answer] = bool(answer_correct.get(answer, False) or bool(record.get("is_correct")))
        answer_verifier[answer] = max(float(answer_verifier.get(answer, 0.0)), float(record.get("verifier_reward") or 0.0))

    prepared, _ = _prepare_inputs(processor, parquet_row)
    responses = [normalized_response_text(tokenizer, str(record.get("raw_text") or "")) for record in records]
    queries = [evidence_query(response) for response in responses]
    unique_queries = sorted(set(queries))
    query_embeddings = embed_queries(model, tokenizer, prepared, unique_queries, device, micro_batch_size)
    embeddings = np.stack([query_embeddings[query] for query in queries]).astype(np.float32)
    rewards, details = all_output_density_rewards(canonical, embeddings)
    reward_index = int(np.argmax(np.asarray(rewards, dtype=np.float32)))
    ours_answer = canonical[reward_index]
    max_reward = float(rewards[reward_index])
    tie_count = sum(abs(float(value) - max_reward) <= 1e-8 for value in rewards)

    return {
        "benchmark": benchmark,
        "question_id": qid,
        "frag_bin": fixed["frag_bin"],
        "fragmentation_ratio": float(fixed["fragmentation_ratio"]),
        "fragmentation_median_over_f_gt_1": float(fixed["fragmentation_median_over_f_gt_1"]),
        "string_cluster_count": int(fixed["string_cluster_count"]),
        "semantic_cluster_count": int(fixed["semantic_cluster_count"]),
        "n_invalid": int(fixed["n_invalid"]),
        "correct_cluster_k": fixed["correct_cluster_k"],
        "correct_cluster_answers": fixed["correct_cluster_answers"],
        "correct_semantic_cluster_count": int(fixed["correct_semantic_cluster_count"]),
        "string_counter_all_outputs": dict(counter),
        "string_majority_answer": majority_answer,
        "string_majority_count": int(majority_count),
        "string_majority_sample_index": int(majority_index),
        "string_majority_correct": bool(answer_correct.get(majority_answer, False)),
        "string_majority_verifier_reward": float(answer_verifier.get(majority_answer, 0.0)),
        "ours_answer": ours_answer,
        "ours_sample_index": int(reward_index),
        "ours_correct": bool(answer_correct.get(ours_answer, False)),
        "ours_verifier_reward": float(answer_verifier.get(ours_answer, 0.0)),
        "ours_selection_score": max_reward,
        "ours_selection_tie_count": int(tie_count),
        "selection_answer_differs": bool(ours_answer != majority_answer),
        "majority_is_invalid": bool(majority_answer == UNKNOWN_LABEL),
        "ours_is_invalid": bool(ours_answer == UNKNOWN_LABEL),
        "reward_variant": "all_output_density_plus_answer_entropy_temperature_no_counterfactual",
        "counterfactual_available": False,
        "counterfactual_omission_reason": "Stage-A real dump has no original/counterfactual token log-prob",
        "invalid_policy": "unknown is an ordinary answer category in majority and density",
        "density_embedding_scope": EMBEDDING_SCOPE,
        "density_temperature_mode": TEMPERATURE_MODE,
        "density_t0": T0,
        "density_t_min": T_MIN,
        "density_t_max": T_MAX,
        "candidate_rewards": [float(value) for value in rewards],
        **details,
    }


def metric_row(scope: str, bin_name: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(selected)
    majority_correct = sum(bool(row["string_majority_correct"]) for row in selected)
    ours_correct = sum(bool(row["ours_correct"]) for row in selected)
    return {
        "benchmark": scope,
        "frag_bin": bin_name,
        "n": n,
        "string_majority_correct": majority_correct,
        "string_majority_at_32": majority_correct / n if n else None,
        "ours_correct": ours_correct,
        "ours_selection_at_32": ours_correct / n if n else None,
        "delta_ours_minus_majority": (ours_correct - majority_correct) / n if n else None,
        "selection_answer_differs_count": sum(bool(row["selection_answer_differs"]) for row in selected),
        "ours_wins": sum(bool(row["ours_correct"]) and not bool(row["string_majority_correct"]) for row in selected),
        "ours_losses": sum(bool(row["string_majority_correct"]) and not bool(row["ours_correct"]) for row in selected),
        "majority_invalid_count": sum(bool(row["majority_is_invalid"]) for row in selected),
        "ours_invalid_count": sum(bool(row["ours_is_invalid"]) for row in selected),
    }


def summarize_main(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scopes = [(benchmark, [row for row in records if row["benchmark"] == benchmark]) for benchmark in BENCHMARK_ORDER]
    scopes.append(("OPEN_MICRO", records))
    bins: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    for scope, selected in scopes:
        for bin_name in BIN_ORDER:
            bins.append(metric_row(scope, bin_name, [row for row in selected if row["frag_bin"] == bin_name]))
        totals.append(metric_row(scope, "overall", selected))
    return bins, totals


def k_bin(value: int) -> str:
    if value <= 1:
        return "k1"
    if value == 2:
        return "k2"
    return "k3plus"


def summarize_gt_k(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scopes = [(benchmark, [row for row in records if row["benchmark"] == benchmark]) for benchmark in BENCHMARK_ORDER]
    scopes.append(("OPEN_MICRO", records))
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for scope, selected_all in scopes:
        eligible = [row for row in selected_all if row["correct_cluster_k"] is not None]
        exclusions.append(
            {
                "benchmark": scope,
                "questions_total": len(selected_all),
                "eligible_correct_answer_observed": len(eligible),
                "excluded_no_correct_answer_in_rollouts": len(selected_all) - len(eligible),
                "multiple_correct_semantic_clusters": sum(int(row["correct_semantic_cluster_count"]) > 1 for row in eligible),
            }
        )
        for label in K_BIN_ORDER:
            bucket = [row for row in eligible if k_bin(int(row["correct_cluster_k"])) == label]
            metric = metric_row(scope, label, bucket)
            metric["gt_k_bin"] = metric.pop("frag_bin")
            rows.append(metric)
    return rows, exclusions


def pct(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def display_bin(value: str) -> str:
    return {"no_frag": "No-frag", "low_frag": "Low-frag", "high_frag": "High-frag"}[value]


def write_metric_table(lines: list[str], rows: dict[str, dict[str, Any]], bin_order: Iterable[str]) -> None:
    lines.extend(
        [
            "| Frag bin | n | string-majority@32 | Ours-Selection | Delta | Different answers |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in bin_order:
        row = rows[label]
        lines.append(
            f"| {display_bin(label)} | {row['n']} | {row['string_majority_correct']}/{row['n']} "
            f"({pct(row['string_majority_at_32'])}) | {row['ours_correct']}/{row['n']} "
            f"({pct(row['ours_selection_at_32'])}) | {pct(row['delta_ours_minus_majority'])} "
            f"| {row['selection_answer_differs_count']} |"
        )


def write_report(
    path: Path,
    fixed_map: list[dict[str, Any]],
    main_bins: list[dict[str, Any]],
    totals: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    nli_path: Path,
) -> None:
    median = float(fixed_map[0]["fragmentation_median_over_f_gt_1"])
    main_by_key = {(row["benchmark"], row["frag_bin"]): row for row in main_bins}
    total_by_scope = {row["benchmark"]: row for row in totals}
    gt_by_key = {(row["benchmark"], row["gt_k_bin"]): row for row in gt_rows}
    excluded_by_scope = {row["benchmark"]: row for row in exclusions}
    lines = [
        "# Fixed Obs1 Fragmentation Bins: String-Majority@32 vs Ours-Selection",
        "",
        "- Responses: fixed step-0 N=32 real rollouts; no response regeneration.",
        f"- Semantic clustering: cached DeBERTa-large-MNLI raw-input bidirectional entailment (`{nli_path}`).",
        "- Invalid handling: every invalid response maps to one ordinary `unknown` answer category; it is not filtered.",
        f"- Fixed pooled median among questions with F>1: `{median:.12g}`.",
        "- Ours replay: evidence-query semantic density with answer-entropy temperature T in [0.10, 0.30].",
        "- Counterfactual token log-prob was not stored, so Ours here is density + temperature only.",
        "- The fixed question-to-bin map is written before selection and shared by both rules.",
        "",
        "## Pooled Main Table",
        "",
    ]
    write_metric_table(lines, {label: main_by_key[("OPEN_MICRO", label)] for label in BIN_ORDER}, BIN_ORDER)
    total = total_by_scope["OPEN_MICRO"]
    lines.append(
        f"| **Overall** | **{total['n']}** | **{total['string_majority_correct']}/{total['n']} "
        f"({pct(total['string_majority_at_32'])})** | **{total['ours_correct']}/{total['n']} "
        f"({pct(total['ours_selection_at_32'])})** | **{pct(total['delta_ours_minus_majority'])}** "
        f"| **{total['selection_answer_differs_count']}** |"
    )
    lines.extend(["", "## Per-dataset Appendix", ""])
    for benchmark in BENCHMARK_ORDER:
        lines.extend([f"### {benchmark}", ""])
        write_metric_table(lines, {label: main_by_key[(benchmark, label)] for label in BIN_ORDER}, BIN_ORDER)
        total = total_by_scope[benchmark]
        lines.append(
            f"| **Overall** | **{total['n']}** | **{total['string_majority_correct']}/{total['n']} "
            f"({pct(total['string_majority_at_32'])})** | **{total['ours_correct']}/{total['n']} "
            f"({pct(total['ours_selection_at_32'])})** | **{pct(total['delta_ours_minus_majority'])}** "
            f"| **{total['selection_answer_differs_count']}** |"
        )
        lines.append("")

    lines.extend(
        [
            "## GT-based Robustness: Correct-cluster String Variants",
            "",
            "If multiple evaluator-correct semantic clusters occur, the largest-mass correct cluster is used; ties use the original cluster order.",
            "",
            "| k bin | n | string-majority@32 | Ours-Selection | Delta | Different answers |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, display in (("k1", "k=1"), ("k2", "k=2"), ("k3plus", "k>=3")):
        row = gt_by_key[("OPEN_MICRO", label)]
        lines.append(
            f"| {display} | {row['n']} | {row['string_majority_correct']}/{row['n']} "
            f"({pct(row['string_majority_at_32'])}) | {row['ours_correct']}/{row['n']} "
            f"({pct(row['ours_selection_at_32'])}) | {pct(row['delta_ours_minus_majority'])} "
            f"| {row['selection_answer_differs_count']} |"
        )
    lines.extend(["", "### Exclusions", "", "| Dataset | total | eligible | excluded: no correct rollout | multiple correct semantic clusters |", "|---|---:|---:|---:|---:|"])
    for scope in (*BENCHMARK_ORDER, "OPEN_MICRO"):
        row = excluded_by_scope[scope]
        lines.append(
            f"| {scope} | {row['questions_total']} | {row['eligible_correct_answer_observed']} | "
            f"{row['excluded_no_correct_answer_in_rollouts']} | {row['multiple_correct_semantic_clusters']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- The main bins are GT-free and fixed solely from step-0 output fragmentation.",
            "- This diagnostic intentionally includes invalid as an ordinary category in both rules; it does not use the training implementation's invalid mask.",
            "- Density and adaptive temperature are changed together relative to string majority. This comparison tests the combined selection rule, not the isolated causal effect of temperature.",
            "- Any non-monotonic result is retained as observed; no thresholds are retuned after seeing accuracy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nli-pairs", type=Path, required=True)
    parser.add_argument("--obs1-question-cache", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--benchmarks", nargs="*", choices=BENCHMARK_ORDER)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nli_predictions = load_nli_predictions(args.nli_pairs)
    obs1_cache = load_obs1_question_cache(args.obs1_question_cache)
    fixed_rows = build_fixed_map(args.dump_root, nli_predictions, obs1_cache)
    fixed_path = args.output_dir / "fixed_fragmentation_bin_map.csv"
    serializable = serializable_map_rows(fixed_rows)
    if fixed_path.exists():
        with fixed_path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        current_keys = [
            (row["benchmark"], str(row["question_id"]), row["frag_bin"], f"{float(row['fragmentation_ratio']):.16g}")
            for row in fixed_rows
        ]
        existing_keys = [
            (row["benchmark"], str(row["question_id"]), row["frag_bin"], f"{float(row['fragmentation_ratio']):.16g}")
            for row in existing
        ]
        if existing_keys != current_keys:
            raise ValueError("Existing fixed fragmentation map differs from recomputed step-0 map")
    else:
        atomic_csv(fixed_path, serializable)
    fixed_map = {(row["benchmark"], str(row["question_id"])): row for row in fixed_rows}
    atomic_json(
        args.output_dir / "fixed_map_summary.json",
        {
            "questions": len(fixed_rows),
            "fragmentation_median_over_f_gt_1": fixed_rows[0]["fragmentation_median_over_f_gt_1"],
            "bin_counts": dict(Counter(row["frag_bin"] for row in fixed_rows)),
            "nli_directional_pairs": len(nli_predictions),
            "invalid_policy": "unknown is an ordinary standalone semantic/string category",
        },
    )
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "fixed_map": str(fixed_path)}, ensure_ascii=False))
        return 0

    selected_benchmarks = tuple(args.benchmarks or BENCHMARK_ORDER)
    parquet_paths = load_manifest(args.manifest)
    result_path = args.output_dir / "question_selection.jsonl"
    completed = {
        (str(row["benchmark"]), str(row["question_id"]))
        for row in read_jsonl(result_path)
    } if result_path.exists() else set()

    model, tokenizer, processor = load_model(args.base_model, args.device)
    for benchmark in selected_benchmarks:
        groups = load_dump_groups(args.dump_root / benchmark / "real.jsonl")
        parquet_rows = load_parquet_rows(parquet_paths[benchmark])
        qids = sorted(groups, key=lambda qid: int(groups[qid][0]["subset_position"]))
        if args.limit_per_dataset > 0:
            qids = qids[: args.limit_per_dataset]
        for position, qid in enumerate(qids, start=1):
            key = (benchmark, qid)
            if key in completed:
                continue
            if qid not in parquet_rows:
                raise KeyError(f"Missing prepared parquet row for {key}")
            result = process_question(
                benchmark,
                qid,
                groups[qid],
                parquet_rows[qid],
                fixed_map[key],
                model,
                tokenizer,
                processor,
                args.device,
                args.micro_batch_size,
            )
            append_jsonl(result_path, result)
            completed.add(key)
            if position % 10 == 0 or position == len(qids):
                print(f"[obs1-replay] {benchmark} {position}/{len(qids)}", flush=True)

    results = [row for row in read_jsonl(result_path) if row["benchmark"] in selected_benchmarks]
    expected = len(selected_benchmarks) * (args.limit_per_dataset if args.limit_per_dataset > 0 else EXPECTED_QUESTIONS)
    if len(results) != expected:
        raise ValueError(f"Expected {expected} completed questions, got {len(results)}")

    if set(selected_benchmarks) == set(BENCHMARK_ORDER) and args.limit_per_dataset == 0:
        main_bins, totals = summarize_main(results)
        gt_rows, exclusions = summarize_gt_k(results)
        atomic_csv(args.output_dir / "fixed_fragmentation_selection.csv", main_bins)
        atomic_csv(args.output_dir / "selection_totals.csv", totals)
        atomic_csv(args.output_dir / "gt_k_selection.csv", gt_rows)
        atomic_csv(args.output_dir / "gt_k_exclusions.csv", exclusions)
        report_path = args.output_dir / "fixed_fragmentation_selection_report.md"
        write_report(report_path, fixed_rows, main_bins, totals, gt_rows, exclusions, args.nli_pairs)
        atomic_json(
            args.output_dir / "run_summary.json",
            {
                "status": "complete",
                "questions": len(results),
                "fixed_bin_counts": dict(Counter(row["frag_bin"] for row in results)),
                "counterfactual_available": False,
                "reward_variant": "all_output_density_plus_answer_entropy_temperature_no_counterfactual",
                "report": str(report_path),
            },
        )
    else:
        atomic_json(
            args.output_dir / "smoke_summary.json",
            {"status": "partial_smoke", "benchmarks": selected_benchmarks, "questions": len(results)},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
