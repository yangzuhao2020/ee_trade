"""GRPO 完整轨迹组采样与验证选模。"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from multiprocessing import get_context
from math import cos, pi

from ..errors import InputValidationError
from ..simulation import simulate_learning_episode
from .checkpoints import load_policy, save_checkpoint
from .grpo_agent import GRPOAgent, collect_group, collect_parallel_group
from .learning_metrics import _metrics
from .test import _evaluate_actor_and_baseline
from .training import _training_progress


def train_single_run(input_path, output_path, 
                     checkpoint_root, scenario, *,
                     config, run, seed, 
                     steps_per_episode, training_steps):
    """完成一次 GRPO 训练，并用验证集选出最佳策略。"""
    # 将交互步数预算换算为完整轨迹组数。
    group_steps = config.group_size * steps_per_episode
    # 训练交互总步数：优先使用传入值，未传入时使用配置值。
    budget = config.training_steps if training_steps is None else training_steps
    if budget <= 0 or budget % group_steps:
        raise InputValidationError(
            f"GRPO training_steps 必须是完整组交互数 {group_steps} 的正整数倍。"
        )
    groups = budget // group_steps
    agent = GRPOAgent(config, seed=seed, run=run)
    best = checkpoint_root / "best.pt"
    best_reward = float("-inf")
    best.unlink(missing_ok=True)

    runner = partial(simulate_learning_episode, input_path, scenario=scenario, period="train")
    validation = partial(simulate_learning_episode, input_path, scenario=scenario, period="validation")
    metrics = []
    # CPU 可并行采样，其他设备使用单进程。
    workers = min(config.sampling_workers, config.group_size) if agent.device.type == "cpu" else 1
    print(f"GRPO 训练轮次：{groups}（每轮 {config.group_size} 条轨迹，{group_steps} 次交互）")
    print(f"GRPO 采样进程：{workers}")
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
            if workers > 1 else nullcontext(None))
    
    with pool as executor:
        for group in _training_progress(groups, run):
            episode = (group - 1) * config.group_size

            def record(result):
                nonlocal episode
                episode += 1
                metrics.append(_metrics(result, run=run, phase="train", episode=episode,
                                        gamma=config.gamma))

            # 2. 收集一组完整轨迹及各回合指标。
            if executor is None:
                batch = collect_group(agent, runner, on_episode=record)
            else:
                batch, episode_metrics = collect_parallel_group(
                    agent, runner, executor, steps_per_episode, episode
                )
                metrics.extend(episode_metrics)
                episode += config.group_size

            # 按训练进度余弦衰减学习率，再更新策略。
            fraction = (group - 1) / max(groups - 1, 1)
            lr_ratio = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + cos(pi * fraction)) / 2
            for parameters in agent.optimizer.param_groups:
                parameters["lr"] = config.learning_rate * lr_ratio

            agent.update(batch)
            
            # 按组间隔验证，最后一组始终验证；只保存验证奖励最高的策略。
            if group % config.validation_interval != 0 and group != groups:
                continue
            result = validation(agent.deterministic_action)
            score = _metrics(result, run=run, phase="validation", episode=episode, gamma=config.gamma)
            metrics.append(score)
            if score.total_reward > best_reward:
                best_reward = score.total_reward
                save_checkpoint(agent, best, episode=episode, validation_total_reward=best_reward)

    # 加载最佳策略，与基准策略比较并输出验证结果。
    action_provider, payload = load_policy(config, best)
    result, comparison = _evaluate_actor_and_baseline(
        action_provider=action_provider,
        input_path=input_path, output_path=output_path / "validation",
        scenario=scenario, period="validation", episode=payload["episode"],
        gamma=config.gamma, run=run,
    )
    metrics.extend(comparison)
    return result, metrics, best, best_reward
