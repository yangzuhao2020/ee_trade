"""第五版学习策略的网络、经验池、参数更新，以及检查点和运行清单持久化。"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..errors import InputValidationError
from .learning import ACTION_SIZE, OBSERVATION_SIZE
from ..models import LearningConfig, LearningTransition
from ..utils import format_timestamp

_CHECKPOINT_VERSION = 2 # 检查点版本
_STATE_VERSION = "v5-observation-16"
_RUN_MANIFEST = "runs.json"


def _periods_payload(config: LearningConfig) -> dict[str, list[str]]:
    """以可写入检查点的字符串形式记录三个交付时段。"""
    return {
        name: [format_timestamp(period.start), format_timestamp(period.end)]
        for name, period in config.periods
    }


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBSERVATION_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, ACTION_SIZE)
        self.apply(_initialize_linear)

    def forward(self, state: Tensor) -> Tensor:
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        return F.softsign(self.fc3(state))


class _TwinCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q1 = self._network()
        self.q2 = self._network()
        self.apply(_initialize_linear)

    @staticmethod
    def _network() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(OBSERVATION_SIZE + ACTION_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat((state, action), dim=1)
        return self.q1(inputs), self.q2(inputs)

    def q1_value(self, state: Tensor, action: Tensor) -> Tensor:
        return self.q1(torch.cat((state, action), dim=1))


def _initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def _read_checkpoint_payload(path: Path, device: torch.device | str, *, algorithm="matd3") -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if (
        payload.get("checkpoint_version") != (3 if algorithm == "grpo" else _CHECKPOINT_VERSION)
        or payload.get("algorithm") != algorithm
        or payload.get("state_version") != _STATE_VERSION
        or payload.get("observation_size") != OBSERVATION_SIZE
        or payload.get("action_size") != ACTION_SIZE
    ):
        raise InputValidationError(
            f"Learning checkpoint {path} is incompatible with Version 5."
        )
    if algorithm == "grpo" and payload.get("distribution") != "softsign-normal-v1":
        raise InputValidationError(f"Incompatible GRPO checkpoint: {path}")
    return payload


class _ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.states = np.empty((capacity, OBSERVATION_SIZE), dtype=np.float32)
        self.actions = np.empty((capacity, ACTION_SIZE), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, OBSERVATION_SIZE), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def append(self, transition: LearningTransition) -> None:
        index = self.position
        self.states[index] = transition.state
        self.actions[index] = transition.action
        self.rewards[index, 0] = transition.reward
        self.next_states[index] = transition.next_state
        self.dones[index, 0] = float(transition.done)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(values[indices], device=device)
            for values in (
                self.states,
                self.actions,
                self.rewards,
                self.next_states,
                self.dones,
            )
        )


class MATD3Agent:
    """第五版 MATD3 的单智能体实现。"""
    def __init__(
        self,
        config: LearningConfig,
        *,
        total_regular_steps: int,
        seed: int | None = None,
        run: int = 0,
    ) -> None:
        self.device = torch.device(config.device)
        self.config = config
        self.seed = seed
        self.run = run
        self.total_regular_steps = total_regular_steps
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.actor = _Actor().to(self.device)
        self.actor_target = _Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        self.critics = _TwinCritic().to(self.device)
        self.critics_target = _TwinCritic().to(self.device)
        self.critics_target.load_state_dict(self.critics.state_dict())
        self.critics_target.eval()
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critics.parameters(), lr=config.learning_rate
        )
        self.replay = _ReplayBuffer(config.replay_buffer_size, self.rng)
        self.regular_steps = 0
        self.gradient_updates = 0
        self.noise_factor = config.noise_dt

    def deterministic_action(self, state: tuple[float, ...]) -> tuple[float, float]:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        return float(action[0]), float(action[1])

    def exploration_action(
        self, state: tuple[float, ...], *, initial_experience: bool
    ) -> tuple[float, float]:
        if initial_experience:
            action = np.full(ACTION_SIZE, state[-1], dtype=np.float64)
            action += self.rng.normal(
                0.0, self.config.exploration_noise_std, size=ACTION_SIZE
            )
        else:
            action = np.asarray(self.deterministic_action(state), dtype=np.float64)
            standard_deviation = (
                self.config.noise_sigma * self.config.noise_scale * self.noise_factor
            )
            action += self.rng.normal(0.0, standard_deviation, size=ACTION_SIZE)
        action = np.clip(action, -1.0, 1.0)
        return float(action[0]), float(action[1])

    def observe(self, transition: LearningTransition, *, train: bool) -> None:
        self.replay.append(transition)
        if not train or self.replay.size < self.config.batch_size:
            return
        self.regular_steps += 1
        if self.regular_steps % self.config.train_frequency_steps != 0:
            return
        progress = self.regular_steps / self.total_regular_steps
        self.noise_factor = self.config.noise_dt * (1.0 - progress)
        for _ in range(self.config.gradient_steps):
            self._gradient_step()

    def _gradient_step(self) -> None:
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size, self.device
        )
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.target_policy_noise
            noise.clamp_(-self.config.target_noise_clip, self.config.target_noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critics_target(next_states, next_actions)
            target_q = rewards + self.config.gamma * (1.0 - dones) * torch.minimum(
                target_q1, target_q2
            )

        current_q1, current_q2 = self.critics(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.gradient_updates += 1
        if self.gradient_updates % self.config.policy_delay != 0:
            return
        for parameter in self.critics.parameters():
            parameter.requires_grad_(False)
        actor_loss = -self.critics.q1_value(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        for parameter in self.critics.parameters():
            parameter.requires_grad_(True)
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critics, self.critics_target)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(source_parameter, alpha=self.config.tau)

    def save_checkpoint(
        self,
        path: Path,
        *,
        episode: int,
        validation_total_reward: float | None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "algorithm": "matd3",
            "state_version": _STATE_VERSION,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "episode": episode,
            "validation_total_reward": validation_total_reward,
            "periods": _periods_payload(self.config),
            "max_bid_price": self.config.max_bid_price,
            "seed": self.seed,
            "run": self.run,
            "actor": self.actor.state_dict(),
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        payload = _read_checkpoint_payload(path, self.device)
        if payload.get("max_bid_price") != self.config.max_bid_price:
            raise InputValidationError(
                f"Learning checkpoint {path} was trained with max_bid_price="
                f"{payload.get('max_bid_price')!r}, but learning_config.max_bid_price="
                f"{self.config.max_bid_price!r}."
            )
        if payload.get("periods") != _periods_payload(self.config):
            raise InputValidationError(
                f"Learning checkpoint {path} 的训练、验证、测试时段与当前 "
                "learning_config.periods 不一致。"
            )
        self.actor.load_state_dict(payload["actor"])
        return payload


def _write_run_manifest(
    checkpoint_root: Path,
    config: LearningConfig,
    runs: list[dict[str, Any]],
    selected_run: int,
) -> None:
    """记录每个独立运行的最佳检查点，供独立的测试命令读取。"""
    payload = {
        "algorithm": config.algorithm,
        "periods": _periods_payload(config),
        "selected_run": selected_run,
        "runs": runs,
    }
    (checkpoint_root / _RUN_MANIFEST).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _read_run_manifest(path: Path, config: LearningConfig) -> dict[str, Any]:
    """读取训练写出的运行清单，并核对其数据划分与当前配置一致。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["algorithm"] != config.algorithm:
        raise InputValidationError("Run manifest algorithm differs from config; use a separate output directory.")
    if payload.get("periods") != _periods_payload(config):
        raise InputValidationError(
            f"Learning run manifest {path} 的训练、验证、测试时段与当前 "
            "learning_config.periods 不一致。"
        )
    return payload


def _default_checkpoint_root(config: LearningConfig, output_path: Path) -> Path:
    if config.trained_policies_save_path is not None:
        return Path(config.trained_policies_save_path)
    return output_path / "checkpoints"


def create_agent(config: LearningConfig):
    if config.algorithm == "grpo":
        from .grpo_agent import GRPOAgent
        return GRPOAgent(config)
    return MATD3Agent(config, total_regular_steps=1)
