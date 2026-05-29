#!/usr/bin/env python3
import math

from verl.utils.reward_score.modelnet40_freeform import compute_score, map_to_modelnet40_class, ttrv_score_freeform


assert map_to_modelnet40_class("a chair") == "8"
assert map_to_modelnet40_class("A wooden chair.") == "8"
assert map_to_modelnet40_class("This is a chair") == "8"
assert map_to_modelnet40_class("chair with armrests") == "8"

assert map_to_modelnet40_class("a night_stand") == "23"
assert map_to_modelnet40_class("a night stand") == "23"
assert map_to_modelnet40_class("nightstand") == "23"

assert map_to_modelnet40_class("3D object") == "unknown"
assert map_to_modelnet40_class("something") == "unknown"

rollouts = ["a chair", "chair", "wooden chair", "a stool", "a piece of furniture"]
rewards = ttrv_score_freeform(rollouts, alpha=0.5)
entropy = -(0.6 * math.log(0.6) + 0.2 * math.log(0.2) + 0.2 * math.log(0.2))
normalized_entropy = entropy / math.log(3)
expected_chair_reward = 0.6 - 0.5 * normalized_entropy
expected_other_reward = 0.2 - 0.5 * normalized_entropy

assert abs(rewards[0] - expected_chair_reward) < 1e-5
assert abs(rewards[3] - expected_other_reward) < 1e-5
assert rewards[4] == -1.0

unknown_rollouts = ["a vivid blue cartoon character"] * 32
unknown_rewards = ttrv_score_freeform(unknown_rollouts, unknown_reward=-1.0)
assert all(reward == -1.0 for reward in unknown_rewards)

mixed_rollouts = ["a chair"] * 16 + ["a vivid blue cartoon character"] * 16
mixed_rewards = ttrv_score_freeform(mixed_rollouts, alpha=0.75, unknown_reward=-1.0)
assert mixed_rewards[0] > mixed_rewards[-1]
assert mixed_rewards[-1] == -1.0

train_outputs = compute_score(
    data_sources=["modelnet40_freeform"] * 32,
    solution_strs=unknown_rollouts,
    ground_truths=["chair"] * 32,
    extra_infos=[{"index": 0, "label_id": 8}] * 32,
    alpha=0.75,
    unknown_reward=-1.0,
)
assert all(output["score"] == -1.0 for output in train_outputs)
assert all(output["unknown"] == 1.0 for output in train_outputs)

eval_outputs = compute_score(
    data_sources=["modelnet40_freeform", "modelnet40_freeform"],
    solution_strs=["a chair", "a vivid blue cartoon character"],
    ground_truths=["chair", "chair"],
    extra_infos=[{"index": 0, "label_id": 8}, {"index": 1, "label_id": 8}],
    alpha=0.75,
    unknown_reward=-1.0,
)
assert eval_outputs[0]["score"] == 1.0
assert eval_outputs[0]["acc"] == 1.0
assert eval_outputs[1]["score"] == 0.0
assert eval_outputs[1]["acc"] == 0.0

print("All reward tests passed!")
