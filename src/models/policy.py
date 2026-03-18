from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class Policy(nn.Module):
    """
    Actor-critic policy for discrete PointNav actions:
    0=forward, 1=left, 2=right.
    """

    def __init__(self, encoder: nn.Module, feature_dim: int = 512, action_dim: int = 3):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # PPO-friendly orthogonal initialization.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        # Smaller policy logits init helps early training stability.
        actor_last = self.actor[-1]
        critic_last = self.critic[-1]
        if isinstance(actor_last, nn.Linear):
            nn.init.orthogonal_(actor_last.weight, gain=0.01)
            nn.init.zeros_(actor_last.bias)
        if isinstance(critic_last, nn.Linear):
            nn.init.orthogonal_(critic_last.weight, gain=1.0)
            nn.init.zeros_(critic_last.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        return self.actor(z), self.critic(z)

    def get_dist_and_value(self, obs: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        logits, value = self(obs)
        return torch.distributions.Categorical(logits=logits), value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value
