"""连续两段报价的随机策略、组回报和 GRPO 更新。"""

from copy import deepcopy
from dataclasses import dataclass
from math import log
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F

from ..errors import InputValidationError
from ..models import LearningConfig
from .agent import _initialize_linear, _periods_payload, _read_checkpoint_payload
from .learning import ACTION_SIZE, OBSERVATION_SIZE


class GaussianActor(nn.Module):
    def __init__(self, initial_std: float) -> None:
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, ACTION_SIZE),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), log(initial_std)))
        self.apply(_initialize_linear)

    def distribution(self, states):
        return Normal(self.mean(states), self.log_std.exp())


def action_log_prob(distribution, latent):
    # Softsign 的导数为 (1 + |z|)^-2；二维动作使用联合概率。
    return (distribution.log_prob(latent) + 2 * torch.log1p(latent.abs())).sum(-1)


def group_advantages(rewards, dones, gamma, threshold):
    returns = torch.zeros_like(rewards, dtype=torch.float64)
    future = torch.zeros(rewards.shape[0], dtype=torch.float64)
    for step in reversed(range(rewards.shape[1])):
        future = rewards[:, step] + gamma * (~dones[:, step]) * future
        returns[:, step] = future
    deviation = returns.std(dim=0, correction=0)
    active = deviation > threshold
    advantages = (returns - returns.mean(dim=0)) / deviation.clamp_min(threshold)
    return advantages.float(), active


@dataclass
class GroupBatch:
    states: torch.Tensor
    latent: torch.Tensor
    old_log_prob: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    can_bid: torch.Tensor


def _collect_trajectory(policy, rng, runner, device):
    samples = []

    def action(state):
        with torch.no_grad():
            observation = torch.tensor(state, dtype=torch.float32, device=device)
            distribution = policy.distribution(observation)
            latent = distribution.mean + distribution.stddev * torch.randn(
                ACTION_SIZE, generator=rng, device=device
            )
            chosen = F.softsign(latent)
            samples.append(tuple(value.cpu() for value in (
                observation, latent, action_log_prob(distribution, latent),
                distribution.mean, distribution.stddev,
            )))
            return tuple(chosen.cpu().tolist())

    result = runner.run_episode(action)
    transitions = result.learning_transitions
    if len(samples) != len(transitions):
        raise RuntimeError("GRPO actions and market transitions are misaligned.")
    fields = [torch.stack(values) for values in zip(*samples)]
    can_bid = [
        any(item.offer.offer_segment in {"learning_minimum", "learning_flexible"}
            and item.offer.offered_energy_mwh > 0 for item in market.offers)
        for market in result.market_results
    ]
    return (*fields,
        torch.tensor([t.reward for t in transitions], dtype=torch.float64),
        torch.tensor([t.done for t in transitions]), torch.tensor(can_bid),
    ), result


def collect_group(agent, runner, on_episode=None):
    """同一旧策略运行完整组；仅返回训练张量和回合指标所需结果。"""
    policy = deepcopy(agent.actor).eval().requires_grad_(False)
    trajectories, results = [], []
    for _ in range(agent.config.group_size):
        trajectory, result = _collect_trajectory(policy, agent.rng, runner, agent.device)
        trajectories.append(trajectory)
        if on_episode is None:
            results.append(result)
        else:
            on_episode(result)
    return GroupBatch(*(torch.stack(values) for values in zip(*trajectories))), results


class GRPOAgent:
    def __init__(self, config: LearningConfig, *, seed=None, run=0):
        self.config, self.seed, self.run = config, seed, run
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise InputValidationError("GRPO device requires an available GPU.")
        if seed is not None:
            torch.manual_seed(seed)
        self.actor = GaussianActor(config.initial_std).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed if seed is not None else torch.seed())
        self.interaction_steps = 0
        self.gradient_updates = 0

    def deterministic_action(self, state):
        with torch.no_grad():
            observation = torch.tensor(state, dtype=torch.float32, device=self.device)
            return tuple(F.softsign(self.actor.mean(observation)).cpu().tolist())

    def update(self, batch: GroupBatch):
        cfg = self.config
        advantages, active = group_advantages(
            batch.rewards, batch.dones, cfg.gamma, cfg.advantage_threshold
        )
        mask = (batch.can_bid & active.unsqueeze(0)).flatten()
        states, latent, old_log_prob, old_mean, old_std = (
            values.flatten(0, 1)[mask].to(self.device) for values in (
                batch.states, batch.latent, batch.old_log_prob, batch.old_mean, batch.old_std
            )
        )
        advantages = advantages.flatten()[mask].to(self.device)
        kl, clipped, loss_value, grad_norm = 0.0, 0.0, 0.0, 0.0
        count = len(advantages)
        updates = 0
        stop_reason = "update_epochs" if count else "no_valid_samples"
        for _ in range(cfg.update_epochs if count else 0):
            indices = torch.randperm(count, generator=self.rng, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for index in indices.split(cfg.minibatch_size):
                distribution = self.actor.distribution(states[index])
                ratio = (action_log_prob(distribution, latent[index]) - old_log_prob[index]).exp()
                surrogate = torch.minimum(
                    ratio * advantages[index],
                    ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages[index],
                )
                # 分批累积整组均值梯度；尾批也按实际样本数加权。
                loss = -surrogate.sum() / count
                loss.backward()
                loss_value += loss.item()
            grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm).item()
            self.optimizer.step()
            self.gradient_updates += 1
            updates += 1
            with torch.no_grad():
                self.actor.log_std.clamp_(log(cfg.min_std), log(cfg.max_std))
                distribution = self.actor.distribution(states)
                kl = kl_divergence(Normal(old_mean, old_std), distribution).sum(-1).mean().item()
                ratio = (action_log_prob(distribution, latent) - old_log_prob).exp()
                clipped = ((ratio - 1).abs() > cfg.clip_epsilon).float().mean().item()
            if kl > cfg.target_kl:
                stop_reason = "target_kl"
                break
        self.optimizer.zero_grad(set_to_none=True)
        self.interaction_steps += batch.rewards.numel()
        return {
            "interaction_steps": self.interaction_steps,
            "gradient_updates": self.gradient_updates,
            "updates_this_group": updates, "eligible_samples": count,
            "sample_coverage": float(updates > 0), "stop_reason": stop_reason,
            "learning_rate": self.optimizer.param_groups[0]["lr"], "grad_norm": grad_norm,
            "loss": loss_value, "kl": kl, "clip_fraction": clipped,
            "policy_std": self.actor.log_std.exp().mean().item(),
            "low_variance_fraction": (~active).float().mean().item(),
        }

    def save_checkpoint(self, path: Path, *, episode, validation_total_reward):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": 3, "algorithm": "grpo",
            "state_version": "v5-observation-16", "distribution": "softsign-normal-v1",
            "advantage_version": "group-time-return-v1",
            "observation_size": OBSERVATION_SIZE, "action_size": ACTION_SIZE,
            "periods": _periods_payload(self.config), "max_bid_price": self.config.max_bid_price,
            "episode": episode, "validation_total_reward": validation_total_reward,
            "seed": self.seed, "run": self.run,
            "interaction_steps": self.interaction_steps, "gradient_updates": self.gradient_updates,
            "actor": self.actor.state_dict(),
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path: Path):
        payload = _read_checkpoint_payload(path, self.device, algorithm="grpo")
        if (payload["periods"] != _periods_payload(self.config)
                or payload["max_bid_price"] != self.config.max_bid_price):
            raise InputValidationError("GRPO checkpoint periods or max_bid_price differ from config.")
        self.actor.load_state_dict(payload["actor"])
        return payload
