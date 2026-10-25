import math

import torch

from verl.utils.torch_functional import (
    self_certainty_from_logits,
    self_certainty_from_logits_with_chunking,
)


def _direct_reverse_kl(logits: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return (-math.log(vocab_size) - log_probs.mean(dim=-1)).clamp_min(0.0)


def test_self_certainty_uniform_logits_is_zero():
    logits = torch.zeros(3, 11)
    actual = self_certainty_from_logits(logits)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=1e-6, rtol=0.0)


def test_self_certainty_matches_direct_reverse_kl():
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 17, dtype=torch.bfloat16)
    expected = _direct_reverse_kl(logits)
    actual = self_certainty_from_logits(logits)
    chunked = self_certainty_from_logits_with_chunking(logits, chunk_size=3)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(chunked, expected, atol=1e-6, rtol=1e-6)


def test_self_certainty_is_not_bounded_by_log_vocab():
    logits = torch.zeros(7)
    logits[0] = 100.0
    actual = self_certainty_from_logits(logits)
    expected = _direct_reverse_kl(logits)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual.item() > math.log(logits.numel())


def test_self_certainty_chunk_size_must_be_positive():
    try:
        self_certainty_from_logits_with_chunking(torch.zeros(2, 3), chunk_size=0)
    except ValueError as exc:
        assert "chunk_size must be positive" in str(exc)
    else:
        raise AssertionError("expected a ValueError for chunk_size=0")


def test_intuitor_reward_masks_tokens_but_not_invalid_candidates():
    import numpy as np

    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import apply_intuitor_self_certainty_reward

    self_certainties = torch.tensor(
        [
            [1.0, 3.0, 0.0, 0.0],
            [2.0, 4.0, 6.0, 0.0],
            [5.0, 7.0, 0.0, 0.0],
            [8.0, 10.0, 12.0, 14.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    data = DataProto.from_dict(
        tensors={
            "responses": torch.zeros(4, 4, dtype=torch.long),
            "self_certainties": self_certainties,
            "response_mask": response_mask,
            "token_level_scores": torch.full((4, 4), 99.0),
            "answer_valid": torch.tensor([True, False, True, True]),
            "answer_correct": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        },
        non_tensors={"uid": np.array(["p0", "p0", "p1", "p1"], dtype=object)},
    )

    data, metrics = apply_intuitor_self_certainty_reward(data, vocab_size=10)

    expected_rewards = torch.tensor([2.0, 4.0, 6.0, 11.0])
    torch.testing.assert_close(data.batch["token_level_scores"].sum(-1), expected_rewards)
    assert data.batch["token_level_scores"][1, 2].item() == 4.0
    assert torch.count_nonzero(data.batch["token_level_scores"]).item() == 4
    assert metrics["train/intuitor_parser_valid_ratio"] == 0.75
    assert metrics["train/intuitor_sc_active_group_ratio"] == 1.0
