"""CPU 多进程采样；组内轨迹顺序和随机数序列与串行保持一致。"""

import torch

from .grpo_agent import GaussianActor, GroupBatch, _collect_trajectory
from .learning import ACTION_SIZE
from .learning_metrics import _metrics


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
    """主进程分配随机流，工作进程只返回训练张量和指标。"""
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
