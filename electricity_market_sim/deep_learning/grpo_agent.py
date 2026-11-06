"""连续两段报价的随机策略、串并行轨迹采样、组回报和 GRPO 更新。"""

from dataclasses import dataclass
from functools import partial
from math import log

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F

from ..errors import InputValidationError
from ..models import LearningConfig
from .network_utils import _deterministic_action, _initialize_linear
from .learning import ACTION_SIZE, OBSERVATION_SIZE
from .learning_metrics import _metrics


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

    def forward(self, states):
        return F.softsign(self.mean(states))


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

    result = runner(action)
    transitions = result.learning_transitions
    fields = [torch.stack(values) for values in zip(*samples)]
    can_bid = [
        any(item.offer.offer_segment in {"learning_minimum", "learning_flexible"}
            for item in market.offers)
        for market in result.market_results
    ]
    return (*fields,
        torch.tensor([t.reward for t in transitions], dtype=torch.float64),
        torch.tensor([t.done for t in transitions]), torch.tensor(can_bid),
    ), result


def collect_group(agent, runner, on_episode):
    """同一旧策略运行完整组，通过回调处理回合结果，只返回训练张量。"""
    trajectories = []
    for _ in range(agent.config.group_size):
        trajectory, result = _collect_trajectory(agent.actor, agent.rng, runner, agent.device)
        trajectories.append(trajectory)
        on_episode(result)
    return GroupBatch(*(torch.stack(values) for values in zip(*trajectories)))


def _sample_trajectory(job):
    config, weights, rng_state, runner, run, episode = job
    torch.set_num_threads(1)
    policy = GaussianActor(config.initial_std).eval().requires_grad_(False)
    policy.load_state_dict(weights)
    rng = torch.Generator().set_state(rng_state)
    trajectory, result = _collect_trajectory(policy, rng, runner, torch.device("cpu"))
    metrics = _metrics(result, run=run, phase="train", episode=episode, gamma=config.gamma)
    return trajectory, metrics


def collect_parallel_group(agent, runner, pool, steps_per_episode, first_episode):
    """CPU 多进程采样；组内轨迹顺序和随机数序列与串行保持一致。"""
    config = agent.config
    weights = {name: value.detach().clone() for name, value in agent.actor.state_dict().items()}
    jobs = []
    for index in range(config.group_size):
        jobs.append((config, weights, agent.rng.get_state(), runner,
                     agent.run, first_episode + index + 1))
        # 与串行逐小时 randn(2) 的调用方式一致；预留每条轨迹的随机数。
        for _ in range(steps_per_episode):
            torch.randn(ACTION_SIZE, generator=agent.rng, device=agent.device)
    trajectories, metrics = zip(*pool.map(_sample_trajectory, jobs))
    batch = GroupBatch(*(torch.stack(values) for values in zip(*trajectories)))
    return batch, list(metrics)


class GRPOAgent:
    def __init__(self, config: LearningConfig, *, seed: int, run: int):
        self.config, self.seed, self.run = config, seed, run
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise InputValidationError("GRPO device requires an available GPU.")
        torch.manual_seed(seed)
        self.actor = GaussianActor(config.initial_std).to(self.device)
        self.deterministic_action = partial(_deterministic_action, self.actor, self.device)
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)

    def update(self, batch: GroupBatch) -> None:
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
        count = len(advantages)
        
        for _ in range(cfg.update_epochs if count else 0):
            indices = torch.randperm(count, generator=self.rng, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            for index in indices.split(cfg.minibatch_size):
                distribution = self.actor.distribution(states[index])
                ratio = (action_log_prob(distribution, latent[index]) - old_log_prob[index]).exp()
                surrogate = torch.minimum(
                    ratio * advantages[index],
                    ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages[index],
                )
                # 分批累积整组均值梯度；尾批也按实际样本数加权。
                loss = -surrogate.sum() / count
                loss.backward() # 反向传播
                
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.optimizer.step() # 更新参数
            
            with torch.no_grad():
                # 限制报价动作的随机波动范围，并计算新旧策略的差异。
                self.actor.log_std.clamp_(log(cfg.min_std), log(cfg.max_std))
                distribution = self.actor.distribution(states)
                kl = kl_divergence(Normal(old_mean, old_std), distribution).sum(-1).mean().item()
            # 策略变化超过阈值时，提前结束本组更新。
            if kl > cfg.target_kl:
                break
            
        # 更新结束后清除梯度，将参数的 grad 设为 None。
        self.optimizer.zero_grad(set_to_none=True)
