"""GRPO 完整轨迹组采样与验证选模。"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
from math import cos, pi
from time import monotonic

from ..errors import InputValidationError
from ..simulation import LearningEpisodeRunner
from .grpo_agent import GRPOAgent, collect_group
from .grpo_sampling import collect_parallel_group
from .learning_metrics import _metrics, _write_metrics, _write_records
from .test import _evaluate_actor_and_baseline


def train_single_run(input_path, output_path, checkpoint_root, scenario, *,
                     config, run, seed, steps_per_episode, training_steps):
    from .training import _training_progress

    group_steps = config.group_size * steps_per_episode
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

    runner = LearningEpisodeRunner(input_path, scenario=scenario, period="train")
    validation = LearningEpisodeRunner(input_path, scenario=scenario, period="validation")
    metrics, diagnostics, progress = [], [], []
    started = monotonic()
    validation_seconds = 0.0
    workers = min(config.sampling_workers, config.group_size) if agent.device.type == "cpu" else 1
    print(f"GRPO 训练轮次：{groups}（每轮 {config.group_size} 条轨迹，{group_steps} 次交互）")
    print(f"GRPO 采样进程：{workers}")
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
            if workers > 1 else nullcontext(None))
    with pool as executor:
        for group in _training_progress(1, groups, run):
            episode = (group - 1) * config.group_size

            def record(result):
                nonlocal episode
                episode += 1
                metrics.append(_metrics(result, run=run, phase="train", episode=episode,
                                        gamma=config.gamma))

            if executor is None:
                batch, _ = collect_group(agent, runner, on_episode=record)
            else:
                batch, episode_metrics = collect_parallel_group(
                    agent, runner, executor, steps_per_episode, episode
                )
                metrics.extend(episode_metrics)
                episode += config.group_size
            fraction = (group - 1) / max(groups - 1, 1)
            lr_ratio = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + cos(pi * fraction)) / 2
            for parameters in agent.optimizer.param_groups:
                parameters["lr"] = config.learning_rate * lr_ratio
            diagnostics.append({"group": group, **agent.update(batch)})
            # 每个完整组后验证，只保存验证奖励最高的策略。
            validation_start = monotonic()
            result = validation.run_episode(agent.deterministic_action)
            validation_seconds += monotonic() - validation_start
            score = _metrics(result, run=run, phase="validation", episode=episode, gamma=config.gamma)
            metrics.append(score)
            progress.append({
                "episode": episode, "interaction_steps": agent.interaction_steps,
                "elapsed_seconds": monotonic() - started, "validation_seconds": validation_seconds,
                "total_reward": score.total_reward,
            })
            if score.total_reward > best_reward:
                best_reward = score.total_reward
                agent.save_checkpoint(best, episode=episode, validation_total_reward=best_reward)

    payload = agent.load_checkpoint(best)
    result, comparison = _evaluate_actor_and_baseline(
        agent=agent, input_path=input_path, output_path=output_path / "validation",
        scenario=scenario, period="validation", episode=payload["episode"],
        gamma=config.gamma, run=run,
    )
    metrics.extend(comparison)
    _write_metrics(output_path / "learning_metrics.csv", metrics)
    _write_records(output_path / "grpo_diagnostics.csv", diagnostics)
    _write_records(output_path / "learning_progress.csv", progress)
    return result, metrics, None, best, best_reward, agent.seed
