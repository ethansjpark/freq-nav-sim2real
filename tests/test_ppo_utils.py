"""Tests for PPO utilities (GAE, RolloutBuffer, ppo_update)."""
from __future__ import annotations

import torch
import pytest

from src.train.ppo_utils import RolloutBuffer, _compute_gae_py, compute_gae, ppo_update
from src.models.encoder import VisualEncoder
from src.models.policy import Policy


class TestRolloutBuffer:
    def test_add_and_len(self):
        buf = RolloutBuffer()
        assert len(buf) == 0
        for _ in range(5):
            buf.add(
                obs=torch.rand(1, 3, 8, 8),
                action=torch.tensor([0]),
                logprob=torch.tensor([-1.0]),
                reward=torch.tensor([0.5]),
                done=torch.tensor([0.0]),
                value=torch.tensor([1.0]),
            )
        assert len(buf) == 5

    def test_as_tensors_shapes(self):
        buf = RolloutBuffer()
        for _ in range(4):
            buf.add(
                obs=torch.rand(1, 3, 8, 8),
                action=torch.tensor([1]),
                logprob=torch.tensor([-0.5]),
                reward=torch.tensor([0.0]),
                done=torch.tensor([0.0]),
                value=torch.tensor([[0.3]]),
            )
        batch = buf.as_tensors()
        assert batch["obs"].shape == (4, 3, 8, 8)
        assert batch["actions"].shape == (4,)
        assert batch["rewards"].shape == (4,)
        assert batch["values"].shape == (4,)

    def test_clear(self):
        buf = RolloutBuffer()
        buf.add(
            obs=torch.rand(1, 3, 8, 8),
            action=torch.tensor([0]),
            logprob=torch.tensor([-1.0]),
            reward=torch.tensor([0.5]),
            done=torch.tensor([0.0]),
            value=torch.tensor([1.0]),
        )
        buf.clear()
        assert len(buf) == 0


class TestComputeGAE:
    def _make_inputs(self, n=8):
        rewards = torch.rand(n)
        dones = torch.zeros(n)
        dones[-1] = 1.0
        values = torch.rand(n)
        next_value = torch.rand(1)
        return rewards, dones, values, next_value

    def test_output_keys(self):
        result = _compute_gae_py(*self._make_inputs(), gamma=0.99, gae_lambda=0.95)
        assert "advantages" in result
        assert "returns" in result

    def test_output_shapes(self):
        rewards, dones, values, nv = self._make_inputs(n=16)
        result = _compute_gae_py(rewards, dones, values, nv, 0.99, 0.95)
        assert result["advantages"].shape == (16,)
        assert result["returns"].shape == (16,)

    def test_returns_equal_advantages_plus_values(self):
        rewards, dones, values, nv = self._make_inputs()
        result = _compute_gae_py(rewards, dones, values, nv, 0.99, 0.95)
        torch.testing.assert_close(
            result["returns"], result["advantages"] + values, atol=1e-5, rtol=1e-5
        )

    def test_dispatch_matches_python(self):
        rewards, dones, values, nv = self._make_inputs()
        py_result = _compute_gae_py(rewards, dones, values, nv, 0.99, 0.95)
        dispatch_result = compute_gae(rewards, dones, values, nv, 0.99, 0.95)
        torch.testing.assert_close(
            py_result["advantages"], dispatch_result["advantages"], atol=1e-5, rtol=1e-5
        )

    def test_zero_gamma_no_bootstrapping(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.zeros(3)
        values = torch.tensor([0.5, 0.5, 0.5])
        nv = torch.tensor([0.5])
        result = _compute_gae_py(rewards, dones, values, nv, gamma=0.0, gae_lambda=0.95)
        expected_adv = rewards - values
        torch.testing.assert_close(result["advantages"], expected_adv, atol=1e-5, rtol=1e-5)

    def test_all_done_no_bootstrapping(self):
        rewards = torch.tensor([1.0, 2.0])
        dones = torch.ones(2)
        values = torch.tensor([0.5, 0.5])
        nv = torch.tensor([10.0])
        result = _compute_gae_py(rewards, dones, values, nv, gamma=0.99, gae_lambda=0.95)
        expected_adv = rewards - values
        torch.testing.assert_close(result["advantages"], expected_adv, atol=1e-5, rtol=1e-5)


class TestPPOUpdate:
    def test_returns_metrics(self):
        encoder = VisualEncoder(feature_dim=32)
        policy = Policy(encoder, feature_dim=32, action_dim=3)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        n = 16
        obs = torch.rand(n, 3, 64, 64)
        actions = torch.randint(0, 3, (n,))
        logprobs = torch.randn(n)
        rewards = torch.rand(n)
        dones = torch.zeros(n)
        values = torch.rand(n)
        advantages = torch.randn(n)
        returns = advantages + values

        batch = {
            "obs": obs,
            "actions": actions,
            "logprobs": logprobs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }

        metrics = ppo_update(
            policy=policy,
            optimizer=optimizer,
            batch=batch,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            update_epochs=1,
            minibatch_size=8,
            max_grad_norm=0.5,
        )

        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert metrics["entropy"] > 0
