#!/usr/bin/env python3
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "verl"))

from verl.utils.reward_score.sqa3d_ttrv import (  # noqa: E402
    compute_score,
    evaluate_sqa3d_official_freeform,
    evaluate_sqa3d_strict_freeform,
    evaluate_sqa3d_em,
    map_to_sqa3d_answer,
    normalize_sqa3d_answer,
    summarize_sqa3d_validation,
    ttrv_score_sqa3d,
)


assert normalize_sqa3d_answer("The chair is brown.") == "chair brown"
assert normalize_sqa3d_answer("A brown chair!") == "brown chair"

with tempfile.TemporaryDirectory() as tmpdir:
    answer_dict_path = Path(tmpdir) / "answer_dict.json"
    answer_dict_path.write_text(json.dumps({"brown": 0, "black": 1, "behind table": 2}), encoding="utf-8")
    official_answer_dict_path = Path(tmpdir) / "official_answer_dict.json"
    official_answer_dict_path.write_text(
        json.dumps([{"brown": 0, "black": 1}, {"0": "brown", "1": "black"}]),
        encoding="utf-8",
    )

    assert map_to_sqa3d_answer("The chair is brown.", {"brown": 0}) == "brown"
    assert map_to_sqa3d_answer("near the behind table area", {"behind table": 2}) == "behind table"
    assert map_to_sqa3d_answer("not in the answer space", {"brown": 0}) == "unknown"
    assert map_to_sqa3d_answer("Yes, you can see the chair.", {"yes": 0, "chair": 1}) == "chair"
    assert map_to_sqa3d_answer(
        "There are three washing machines on my right.",
        {"three": 0, "washing machines": 1, "right": 2},
        question_type="how",
    ) == "washing machines"
    assert map_to_sqa3d_answer("The door is closed.", {"door": 0, "closed": 1}) == "closed"
    assert map_to_sqa3d_answer("The chair is on my left side.", {"on": 0, "left": 1}) == "left"
    answer_prior = Counter({"right": 200, "to right": 1, "closed": 60, "door": 28, "three": 53, "washing machines": 4})
    assert map_to_sqa3d_answer("I will go to the right.", {"right": 0, "to right": 1}, answer_prior) == "right"
    assert map_to_sqa3d_answer("The door is closed.", {"door": 0, "closed": 1}, answer_prior) == "closed"
    assert map_to_sqa3d_answer(
        "There are three washing machines.",
        {"three": 0, "washing machines": 1},
        answer_prior,
    ) == "three"
    assert map_to_sqa3d_answer("3", {"three": 0}) == "three"

    assert evaluate_sqa3d_em("brown", ["brown"], {"brown": 0})
    assert evaluate_sqa3d_em("The chair is brown.", ["brown"], {"brown": 0})
    assert evaluate_sqa3d_em("There are three rows.", ["three"], {"three": 0})
    assert evaluate_sqa3d_em("The door is closed.\nI am standing there.", ["closed"], {"door": 0, "closed": 1})
    assert not evaluate_sqa3d_em("black", ["brown"], {"brown": 0, "black": 1})
    assert evaluate_sqa3d_em("I will go to the right.", ["right"], {"right": 0, "to right": 1}, answer_prior)
    assert not evaluate_sqa3d_em("I am facing the desk on my left or right side.", ["left"], {"left": 0, "right side": 1})
    assert not evaluate_sqa3d_em("True or false?", ["false"], {"true": 0, "false": 1})
    assert evaluate_sqa3d_official_freeform("I will go to the right.", "right")
    assert evaluate_sqa3d_official_freeform("True or false?", "false")
    assert not evaluate_sqa3d_strict_freeform("I will go to the right.", "right")
    assert evaluate_sqa3d_strict_freeform("3", "three")

    rollouts = ["brown", "The chair is brown.", "brown", "black"]
    rewards = ttrv_score_sqa3d(rollouts, alpha=0.5, answer_dict_path=str(answer_dict_path))
    entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(2)
    assert abs(rewards[0] - (0.75 - 0.5 * entropy)) < 1e-5
    assert abs(rewards[-1] - (0.25 - 0.5 * entropy)) < 1e-5
    assert abs(ttrv_score_sqa3d(["brown"], answer_dict_path=str(official_answer_dict_path))[0] - 1.0) < 1e-5

    train_outputs = compute_score(
        data_sources=["sqa3d"] * 4,
        solution_strs=rollouts,
        ground_truths=["brown"] * 4,
        extra_infos=[{"index": 0, "question_type": "what"}] * 4,
        alpha=0.5,
        answer_dict_path=str(answer_dict_path),
    )
    assert train_outputs[0]["score"] > train_outputs[-1]["score"]
    assert train_outputs[0]["acc"] == 1.0
    assert train_outputs[-1]["acc"] == 0.0
    assert train_outputs[1]["acc"] == 1.0
    assert train_outputs[1]["strict_acc"] == 0.0
    assert train_outputs[0]["answer_type"] == "what"
    assert math.isclose(train_outputs[0]["unique_pred_ratio"], 0.5)
    assert math.isclose(train_outputs[0]["group_pass"], 1.0)
    assert math.isclose(train_outputs[0]["group_em_mean"], 0.75)

    hybrid_outputs = compute_score(
        data_sources=["sqa3d"] * 4,
        solution_strs=rollouts,
        ground_truths=["brown"] * 4,
        extra_infos=[{"index": 0, "question_type": "what"}] * 4,
        alpha=0.5,
        answer_dict_path=str(answer_dict_path),
        reward_strategy="hybrid",
        em_bonus=1.0,
    )
    assert math.isclose(hybrid_outputs[0]["score"], train_outputs[0]["score"] + 1.0)
    assert math.isclose(hybrid_outputs[-1]["score"], train_outputs[-1]["score"])

    eval_outputs = compute_score(
        data_sources=["sqa3d", "sqa3d"],
        solution_strs=["brown", "not mapped"],
        ground_truths=["brown", "brown"],
        extra_infos=[{"index": 0}, {"index": 1}],
        alpha=0.5,
        answer_dict_path=str(answer_dict_path),
    )
    assert eval_outputs[0]["score"] == 1.0
    assert eval_outputs[1]["score"] == 0.0
    assert eval_outputs[1]["unknown"] == 1.0

    summary = summarize_sqa3d_validation(
        data_sources=["sqa3d"] * 8,
        group_keys=["g1"] * 4 + ["g2"] * 4,
        infos_dict={
            "acc": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "strict_acc": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "closed_acc": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "pred": ["brown", "unknown", "brown", "black", "unknown", "yes", "yes", "yes"],
            "answer_type": ["what"] * 4 + ["is"] * 4,
            "unknown": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "closed_unknown": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
    )
    assert math.isclose(summary["val-sqa3d/sqa3d/em/overall/baseline"], 0.5)
    assert math.isclose(summary["val-sqa3d/sqa3d/em/overall/sample-level"], 1.0)
    assert math.isclose(summary["val-sqa3d/sqa3d/em/overall/dataset-level"], 1.0)
    assert math.isclose(summary["val-sqa3d/sqa3d/em/what/baseline"], 1.0)
    assert math.isclose(summary["val-sqa3d/sqa3d/em/is/baseline"], 0.0)
    assert math.isclose(summary["val-sqa3d/sqa3d/strict_em/overall/baseline"], 0.5)
    assert math.isclose(summary["val-sqa3d/sqa3d/strict_em/overall/sample-level"], 0.5)
    assert math.isclose(summary["val-sqa3d/sqa3d/closed_em/overall/baseline"], 0.5)
    assert math.isclose(summary["val-sqa3d/sqa3d/unknown/overall/baseline"], 0.5)
    assert math.isclose(summary["val-sqa3d/sqa3d/unknown/overall/sample-level"], 0.0)
    assert math.isclose(summary["val-sqa3d/sqa3d/unknown/overall/dataset-level"], 0.0)

print("All SQA3D reward tests passed!")
