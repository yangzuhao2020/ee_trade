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
from ..models import LearningConfig, LearningTransition, MarketSettings
from ..utils import format_timestamp

_CHECKPOINT_VERSION = 2


_STATE_VERSION = "v5-observation-16"


_RUN_MANIFEST = "runs.json"


def _periods_payload(config: LearningConfig) -> dict[str, list[str]]:
    """以可写入检查点的字符串形式记录三个交付时段。"""
    return {
        name: [format_timestamp(period.start), format_timestamp(period.end)]
        for name, period in config.periods or ()
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


def _read_checkpoint_payload(path: Path, device: torch.device | str) -> dict[str, Any]:
    if not path.is_file():
        raise InputValidationError(f"Learning checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise InputValidationError(
            f"Cannot load learning checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise InputValidationError(
            f"Learning checkpoint {path} is incompatible with Version 5."
        )
    if (
        payload.get("checkpoint_version") != _CHECKPOINT_VERSION
        or payload.get("state_version") != _STATE_VERSION
        or payload.get("observation_size") != OBSERVATION_SIZE
        or payload.get("action_size") != ACTION_SIZE
    ):
        raise InputValidationError(
            f"Learning checkpoint {path} is incompatible with Version 5."
        )
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
        if self.size < batch_size:
            raise RuntimeError("Replay buffer does not contain a complete batch.")
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

    def state_dict(self) -> dict[str, Any]:
        if self.size < self.capacity:
            indices = np.arange(self.size)
        else:
            indices = np.concatenate(
                (np.arange(self.position, self.capacity), np.arange(self.position))
            )
        return {
            "capacity": self.capacity,
            "states": torch.from_numpy(self.states[indices].copy()),
            "actions": torch.from_numpy(self.actions[indices].copy()),
            "rewards": torch.from_numpy(self.rewards[indices].copy()),
            "next_states": torch.from_numpy(self.next_states[indices].copy()),
            "dones": torch.from_numpy(self.dones[indices].copy()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stored_size = int(state["states"].shape[0])
        if stored_size > self.capacity:
            raise InputValidationError(
                "Checkpoint replay buffer is larger than the configured capacity."
            )
        expected_shapes = {
            "states": (stored_size, OBSERVATION_SIZE),
            "actions": (stored_size, ACTION_SIZE),
            "rewards": (stored_size, 1),
            "next_states": (stored_size, OBSERVATION_SIZE),
            "dones": (stored_size, 1),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, shape in expected_shapes.items():
            value = state.get(name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise InputValidationError(
                    f"Checkpoint replay field {name!r} has an invalid shape."
                )
            arrays[name] = value.cpu().numpy()
        self.states[:stored_size] = arrays["states"]
        self.actions[:stored_size] = arrays["actions"]
        self.rewards[:stored_size] = arrays["rewards"]
        self.next_states[:stored_size] = arrays["next_states"]
        self.dones[:stored_size] = arrays["dones"]
        self.size = stored_size
        self.position = stored_size % self.capacity


class MATD3Agent:
    """The Version 5 single-agent specialization of MATD3/TD3."""

    def __init__(
        self,
        config: LearningConfig,
        *,
        total_regular_steps: int,
        seed: int | None = None,
        run: int = 0,
    ) -> None:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise InputValidationError(
                f"learning_config.device={config.device!r} requires an available GPU."
            )
        try:
            self.device = torch.device(config.device)
        except (RuntimeError, ValueError) as exc:
            raise InputValidationError(
                f"Invalid learning_config.device: {config.device!r}."
            ) from exc
        self.config = config
        self.seed = seed
        self.run = run
        self.total_regular_steps = max(1, total_regular_steps)
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
        progress = min(1.0, self.regular_steps / self.total_regular_steps)
        self.noise_factor = max(0.0, self.config.noise_dt * (1.0 - progress))
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
                source.parameters(), target.parameters(), strict=True
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
            "state_version": _STATE_VERSION,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "episode": episode,
            "validation_total_reward": validation_total_reward,
            "periods": _periods_payload(self.config),
            "max_bid_price": self.config.max_bid_price,
            "seed": self.seed,
            "run": self.run,
            "regular_steps": self.regular_steps,
            "gradient_updates": self.gradient_updates,
            "noise_factor": self.noise_factor,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critics": self.critics.state_dict(),
            "critics_target": self.critics_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": json.dumps(self.rng.bit_generator.state),
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)

    def load_checkpoint(self, path: Path, *, load_optimizers: bool) -> dict[str, Any]:
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
        try:
            self.actor.load_state_dict(payload["actor"])
            self.actor_target.load_state_dict(payload["actor_target"])
            self.critics.load_state_dict(payload["critics"])
            self.critics_target.load_state_dict(payload["critics_target"])
            if load_optimizers:
                self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
                self.replay.load_state_dict(payload["replay"])
                self.regular_steps = int(payload["regular_steps"])
                self.gradient_updates = int(payload["gradient_updates"])
                self.noise_factor = float(payload["noise_factor"])
                saved_seed = payload.get("seed")
                if saved_seed is not None:
                    if not isinstance(saved_seed, int) or isinstance(saved_seed, bool):
                        raise TypeError("Checkpoint seed must be an integer or null.")
                    self.seed = saved_seed
                saved_run = payload.get("run")
                if isinstance(saved_run, int) and not isinstance(saved_run, bool):
                    self.run = saved_run
                torch.set_rng_state(payload["torch_rng_state"].cpu())
                self.rng.bit_generator.state = json.loads(payload["numpy_rng_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise InputValidationError(
                f"Learning checkpoint {path} is incomplete or incompatible."
            ) from exc
        return payload


def _checkpoint_from_setting(path: str | None, fallback: Path, name: str) -> Path:
    if path is None:
        return fallback / name
    configured = Path(path)
    return configured if configured.suffix == ".pt" else configured / name


def _learning_config(settings: MarketSettings) -> LearningConfig:
    config = settings.learning_config
    if config is None or not config.learning_mode:
        raise InputValidationError(
            "Learning training/evaluation requires learning_config.learning_mode: true."
        )
    return config


def _write_run_manifest(
    checkpoint_root: Path,
    config: LearningConfig,
    runs: list[dict[str, Any]],
    selected_run: int,
) -> None:
    """记录每个独立运行的最佳检查点，供独立的测试命令读取。"""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "periods": _periods_payload(config),
        "selected_run": selected_run,
        "runs": runs,
    }
    (checkpoint_root / _RUN_MANIFEST).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _read_run_manifest(path: Path, config: LearningConfig) -> dict[str, Any]:
    """读取训练写出的运行清单，并核对其数据划分与当前配置一致。"""

    if not path.is_file():
        raise InputValidationError(
            f"Learning run manifest does not exist: {path}; please train first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs = payload["runs"]
        for entry in runs:
            if (
                not isinstance(entry["run"], int)
                or isinstance(entry["run"], bool)
                or not isinstance(entry["best_checkpoint"], str)
                or not isinstance(entry["validation_total_reward"], (int, float))
                or isinstance(entry["validation_total_reward"], bool)
            ):
                raise TypeError
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InputValidationError(f"Learning run manifest {path} is invalid.") from exc
    if not runs:
        raise InputValidationError(f"Learning run manifest {path} lists no runs.")
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
