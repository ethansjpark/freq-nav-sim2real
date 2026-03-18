from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    value: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.data: List[Transition] = []

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.data.append(
            Transition(
                obs=obs.detach(),
                action=action.detach(),
                logprob=logprob.detach(),
                reward=reward.detach(),
                done=done.detach(),
                value=value.detach(),
            )
        )

    def __len__(self) -> int:
        return len(self.data)

    def as_tensors(self) -> Dict[str, torch.Tensor]:
        obs = torch.cat([t.obs for t in self.data], dim=0)
        actions = torch.cat([t.action for t in self.data], dim=0)
        logprobs = torch.cat([t.logprob for t in self.data], dim=0)
        rewards = torch.cat([t.reward for t in self.data], dim=0)
        dones = torch.cat([t.done for t in self.data], dim=0)
        values = torch.cat([t.value for t in self.data], dim=0).squeeze(-1)
        return {
            "obs": obs,
            "actions": actions,
            "logprobs": logprobs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
        }

    def clear(self) -> None:
        self.data.clear()


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, torch.Tensor]:
    t = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(1, device=rewards.device)
    next_val = next_value.reshape(1)

    for step in reversed(range(t)):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_val * non_terminal - values[step]
        last_adv = delta + gamma * gae_lambda * non_terminal * last_adv
        advantages[step] = last_adv
        next_val = values[step].reshape(1)

    returns = advantages + values
    return {"advantages": advantages, "returns": returns}


def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    update_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    obs = batch["obs"]
    actions = batch["actions"].long()
    old_logprobs = batch["logprobs"]
    returns = batch["returns"]
    advantages = batch["advantages"]

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = obs.shape[0]
    metrics: Dict[str, float] = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(update_epochs):
        idx = torch.randperm(n, device=obs.device)
        for start in range(0, n, minibatch_size):
            mb_idx = idx[start : start + minibatch_size]
            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logprobs = old_logprobs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]

            logits, values = policy(mb_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            values = values.squeeze(-1)

            ratio = (new_logprobs - mb_old_logprobs).exp()
            pg_loss_1 = -mb_advantages * ratio
            pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

            value_loss = F.mse_loss(values, mb_returns)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            metrics["loss"] += float(loss.detach().item())
            metrics["policy_loss"] += float(policy_loss.detach().item())
            metrics["value_loss"] += float(value_loss.detach().item())
            metrics["entropy"] += float(entropy.detach().item())
            updates += 1

    if updates > 0:
        for key in metrics:
            metrics[key] /= float(updates)
    return metrics
