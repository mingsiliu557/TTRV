#!/usr/bin/env python3
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "verl"))

from verl.trainer.ppo.metric_utils import process_validation_metrics


def main():
    data_sources = ["modelnet40_freeform"] * 40
    sample_inputs = ["What is this?"] * 40
    group_keys = [f"modelnet40_freeform::{idx}" for idx in (0, 1) for _ in range(20)]

    acc = [1.0] * 10 + [0.0] * 10 + [1.0] * 20
    pred = ["chair"] * 10 + ["unknown"] * 10 + ["table"] * 20
    infos = {"acc": acc, "pred": pred}

    prompt_grouped = process_validation_metrics(data_sources, sample_inputs, infos)
    prompt_acc = prompt_grouped["modelnet40_freeform"]["acc"]
    assert "mean@40" in prompt_acc, prompt_acc.keys()
    assert "mean@20" not in prompt_acc, prompt_acc.keys()

    index_grouped = process_validation_metrics(data_sources, sample_inputs, infos, group_keys=group_keys)
    index_acc = index_grouped["modelnet40_freeform"]["acc"]
    assert "mean@20" in index_acc, index_acc.keys()
    assert "mean@40" not in index_acc, index_acc.keys()
    assert "best@20/mean" in index_acc, index_acc.keys()
    assert "maj@20/mean" in index_acc, index_acc.keys()
    assert math.isclose(index_acc["mean@20"], 0.75), index_acc["mean@20"]

    print("Validation grouping tests passed!")


if __name__ == "__main__":
    main()
