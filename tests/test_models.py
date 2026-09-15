"""Tests for visual encoder and policy network."""
from __future__ import annotations

import torch
import pytest

from src.models.encoder import VisualEncoder
from src.models.policy import Policy


class TestVisualEncoder:
    def test_output_shape(self):
        enc = VisualEncoder(feature_dim=512)
        x = torch.rand(2, 3, 64, 64)
        out = enc(x)
        assert out.shape == (2, 512)

    def test_custom_feature_dim(self):
        enc = VisualEncoder(feature_dim=128)
        x = torch.rand(1, 3, 64, 64)
        out = enc(x)
        assert out.shape == (1, 128)

    def test_different_image_sizes(self):
        enc = VisualEncoder(feature_dim=256)
        for size in [64, 128, 224]:
            x = torch.rand(1, 3, size, size)
            out = enc(x)
            assert out.shape == (1, 256)

    def test_rejects_non_4d(self):
        enc = VisualEncoder()
        with pytest.raises(ValueError, match="Expected BCHW"):
            enc(torch.rand(3, 64, 64))

    def test_gradient_flows(self):
        enc = VisualEncoder(feature_dim=64)
        x = torch.rand(1, 3, 64, 64, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestPolicy:
    @pytest.fixture
    def policy(self):
        enc = VisualEncoder(feature_dim=64)
        return Policy(enc, feature_dim=64, action_dim=3)

    def test_forward_shapes(self, policy):
        obs = torch.rand(4, 3, 64, 64)
        logits, value = policy(obs)
        assert logits.shape == (4, 3)
        assert value.shape == (4, 1)

    def test_act_shapes(self, policy):
        obs = torch.rand(1, 3, 64, 64)
        action, logprob, value = policy.act(obs)
        assert action.shape == (1,)
        assert logprob.shape == (1,)
        assert value.shape == (1, 1)

    def test_act_valid_actions(self, policy):
        obs = torch.rand(8, 3, 64, 64)
        action, _, _ = policy.act(obs)
        assert (action >= 0).all() and (action < 3).all()

    def test_evaluate_actions(self, policy):
        obs = torch.rand(4, 3, 64, 64)
        actions = torch.tensor([0, 1, 2, 0])
        logprob, entropy, value = policy.evaluate_actions(obs, actions)
        assert logprob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4, 1)
        assert (entropy >= 0).all()

    def test_gradient_flows(self, policy):
        obs = torch.rand(2, 3, 64, 64)
        logits, value = policy(obs)
        loss = logits.sum() + value.sum()
        loss.backward()
        for p in policy.parameters():
            assert p.grad is not None
