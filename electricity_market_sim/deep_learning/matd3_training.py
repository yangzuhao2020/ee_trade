"""MATD3 回合训练、经验收集与验证选模。"""

from functools import partial

from ..errors import InputValidationError
from ..simulation import simulate_learning_episode
from .checkpoints import load_policy, save_checkpoint
from .learning_metrics import _EpisodeMetrics, _metrics
from .matd3_agent import MATD3Agent
from .test import _evaluate_actor_and_baseline
from .training import _training_progress


def train_single_run(input_path, output_path, 
                     checkpoint_root, scenario, *,
                     config, run, seed, steps_per_episode,
                     training_episodes, training_steps):
    """完成一次 MATD3 训练，收集经验、更新网络，并按验证奖励选模。"""
    # 确定训练回合数；设置步数预算时按完整回合换算。
    episode_count = (
        config.training_episodes if training_episodes is None else training_episodes
    )
    budget = config.training_steps if training_steps is None else training_steps
    if budget:
        if budget % steps_per_episode:
            raise InputValidationError(f"MATD3 training_steps 必须是 {steps_per_episode} 的正整数倍。")
        episode_count = budget // steps_per_episode
        
    # 扣除初始经验收集和经验池预热，计算探索噪声衰减所用步数。
    regular_episode_count = max(0, episode_count - config.initial_experience_episodes)
    initial_transition_count = (
        min(episode_count, config.initial_experience_episodes) * steps_per_episode
    )
    regular_transition_count = regular_episode_count * steps_per_episode
    warmup_transitions = max(0, config.batch_size - initial_transition_count - 1)
    eligible_regular_steps = max(0, regular_transition_count - warmup_transitions)
    
    agent = MATD3Agent(
        config,
        total_regular_steps=eligible_regular_steps,
        seed=seed,
        run=run,
    )
    best_path = checkpoint_root / "best.pt"
    best_path.unlink(missing_ok=True)

    # 只有训练时段的经验进入经验池；验证时段只运行确定性 Actor。
    train_episode = partial(simulate_learning_episode, input_path, scenario=scenario, period="train")
    validation_episode = partial(simulate_learning_episode, input_path, scenario=scenario, period="validation")
    metrics: list[_EpisodeMetrics] = []
    best_total_reward = float("-inf")
    
    # 初始回合只收集经验，后续回合通过 observe() 按间隔更新网络。
    for episode in _training_progress(episode_count, run):
        initial_experience = episode <= config.initial_experience_episodes
        training_result = train_episode(
            lambda state, initial=initial_experience: agent.exploration_action(
                state, initial_experience=initial
            ),
            transition_consumer=lambda transition, enabled=not initial_experience: (
                agent.observe(transition, train=enabled)
            ),
        )
        metrics.append(
            _metrics(
                training_result,
                run=run,
                phase="initial_experience" if initial_experience else "train",
                episode=episode,
                gamma=config.gamma,
            )
        )
        # 按回合间隔验证，最后一回合始终验证。
        is_validation_due = episode % config.validation_interval == 0
        is_last_episode = episode == episode_count
        should_validate = is_validation_due or is_last_episode
        
        if should_validate:
            validation_result = validation_episode(agent.deterministic_action)
            validation_metrics = _metrics(
                validation_result,
                run=run,
                phase="validation",
                episode=episode,
                gamma=config.gamma,
            )
            metrics.append(validation_metrics)
            # 按验证时段总奖励选模型；打平时保留更早的检查点。
            if (
                not initial_experience
                and validation_metrics.total_reward > best_total_reward
            ):
                best_total_reward = validation_metrics.total_reward
                save_checkpoint(
                    agent,
                    best_path,
                    episode=episode,
                    validation_total_reward=best_total_reward,
                )

    if best_total_reward == float("-inf"):
        raise InputValidationError(
            "训练未生成最佳模型检查点；请增加训练回合数 training_episodes，"
            "或减少初始经验收集回合数 episodes_collecting_initial_experience。"
        )

    # 加载验证表现最佳的策略，与基准策略比较。
    action_provider, best_payload = load_policy(config, best_path)
    validation_result, comparison = _evaluate_actor_and_baseline(
        action_provider=action_provider,
        input_path=input_path,
        output_path=output_path / "validation",
        scenario=scenario,
        period="validation",
        episode=best_payload["episode"],
        gamma=config.gamma,
        run=run,
    )
    metrics.extend(comparison)
    return (
        validation_result,
        metrics,
        best_path,
        best_total_reward,
    )

