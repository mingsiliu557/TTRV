"""GPU-first contract gates. Failures are blockers, never xfailed or suppressed."""
import numpy as np
import pytest
import torch

from verl import DataProto
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import test_time_train_metrics as reward_metrics
from verl.workers.reward_manager.ttrl import TTRLRewardManager


@pytest.fixture(autouse=True)
def require_gpu():
    assert torch.cuda.is_available(), "This contract suite must run in the allocated GPU shell"


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids, **kwargs):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return "yes" if ids and ids[0] == 1 else "no" if ids and ids[0] == 2 else ""


def batch(lengths, label="yes"):
    width = max(max(lengths), 1)
    response = torch.zeros(len(lengths), width, dtype=torch.long, device="cuda")
    mask = torch.zeros_like(response)
    for i, n in enumerate(lengths):
        response[i, :n] = 1
        mask[i, :n] = 1
    prompts = torch.ones(len(lengths), 2, dtype=torch.long, device="cuda")
    return DataProto.from_dict(dict(prompts=prompts, responses=response,
                                    attention_mask=torch.cat((torch.ones_like(prompts), mask), -1)),
                               dict(data_source=np.array(["MME-YN-TTT"] * len(lengths), dtype=object),
                                    reward_model=np.array([dict(ground_truth=label)] * len(lengths), dtype=object),
                                    extra_info=np.array([dict(index=str(i)) for i in range(len(lengths))], dtype=object)))


def test_yesno_dispatch_and_invalid():
    assert auto_extract("yesno", ["Yes, it is.", "I think yes", "yes and no", ""]) == ["yes", "yes", "unknown", "unknown"]
    rewards, _ = auto_verify("yesno", ["maybe", "yes", "no"], ["maybe", "yes", "no"])
    assert rewards == [0.0, 1.0, 1.0]
    manager = TTRLRewardManager(Tokenizer(), 0)
    assert manager._data_source_to_task("MME-YN-TTT") == "yesno"
    assert not manager._selection_correct("unknown", "unknown", "MME-YN-TTT")


@pytest.mark.parametrize("style", ["frequency_entropy", "density_cluster_answer_entropy"])
def test_label_permutation_does_not_change_rewards(style):
    torch.manual_seed(42)
    # Existing density implementation accepts numpy features; this is not a CPU model.
    features = torch.randn(32, 16, device="cuda").cpu().numpy()
    responses = ["Yes, it is."] * 17 + ["no"] * 14 + ["maybe"]
    def compute(label):
        return reward_metrics(responses, [label] * 32, task="yesno", reward_style=style,
                              response_embeddings=features, return_details=True)[0]
    np.testing.assert_allclose(compute("yes"), compute("no"))


def test_yesno_canonical_evidence():
    from verl.trainer.ppo.ray_trainer import _canonicalize_density_evidence_text
    template = "Candidate option: {candidate}\nVisual evidence needed:"
    assert _canonicalize_density_evidence_text("Is it red?", "Yes, it is.", template, "yesno") == "Candidate option: yes\nVisual evidence needed:"
    assert _canonicalize_density_evidence_text("Is it red?", "maybe", template, "yesno") == "Candidate option: unknown\nVisual evidence needed:"


@pytest.mark.parametrize("lengths", [[1, 4, 7], [1, 4, 0]])
def test_eval_reward_uses_each_response_end(lengths):
    data = batch(lengths)
    result = TTRLRewardManager(Tokenizer(), 0, mode="eval")(data, return_dict=True)["reward_tensor"]
    expected = torch.zeros_like(result)
    for i, length in enumerate(lengths):
        if length:
            expected[i, length - 1] = 1
    torch.testing.assert_close(result, expected, msg="I11: evaluation reward must land on each response's own final token; empty rows must stay zero")


def test_train_32_to_16_reward_alignment():
    lengths = ([1, 4, 7] * 22)[:64]
    data = batch(lengths)
    result = TTRLRewardManager(Tokenizer(), 0, mode="train", n_votes_per_prompt=32, n_samples_per_prompt=16)(data, return_dict=True)["reward_tensor"]
    assert result.shape == (32, 7)
    expected = torch.zeros_like(result)
    for group in range(2):
        for row in range(16):
            expected[group * 16 + row, lengths[group * 32 + row] - 1] = 1
    torch.testing.assert_close(result, expected, msg="I11/I05: retained first-16 rewards must preserve their response lengths and group order")
