"""Single-agent MATD3 training and evaluation for Version 5 scenarios."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import load_market_settings
from .errors import InputValidationError
from .learning import ACTION_SIZE, OBSERVATION_SIZE
from .learning_metrics import LEARNING_METRIC_DECIMALS, LEARNING_METRIC_FIELDS
from .models import LearningConfig, LearningTransition, MarketSettings, SimulationResult
from .reporting import write_results
from .simulation import LearningEpisodeRunner, market_openings

_CHECKPOINT_VERSION = 1
_STATE_VERSION = "v5-observation-38"
_DEFAULT_INDEPENDENT_RUNS = 3


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
        load_base_mw: float,
        validation_discounted_reward: float | None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "state_version": _STATE_VERSION,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "episode": episode,
            "load_base_mw": load_base_mw,
            "validation_discounted_reward": validation_discounted_reward,
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


@dataclass(frozen=True)
class _EpisodeMetrics:
    run: int
    phase: str
    episode: int
    total_reward: float
    discounted_reward: float
    total_profit_eur: float
    accepted_energy_mwh: float
    minimum_segment_average_bid_eur_per_mwh: float
    flexible_segment_average_bid_eur_per_mwh: float
    minimum_segment_acceptance_ratio: float
    flexible_segment_acceptance_ratio: float


def _metrics(
    result: SimulationResult, *, run: int, phase: str, episode: int, gamma: float
) -> _EpisodeMetrics:
    total_reward = 0.0
    discounted_reward = 0.0
    total_profit = 0.0
    accepted_energy = 0.0
    segment_totals = {
        "learning_minimum": {"offered": 0.0, "accepted": 0.0, "price": 0.0},
        "learning_flexible": {"offered": 0.0, "accepted": 0.0, "price": 0.0},
    }
    durations = {
        market.delivery_start: market.duration_hours for market in result.market_results
    }
    for index, step in enumerate(result.learning_steps):
        total_reward += step.reward
        discounted_reward += gamma**index * step.reward
        total_profit += step.profit_eur
        accepted_energy += step.accepted_power_mw * durations[step.delivery_start]
    for market in result.market_results:
        for cleared in market.offers:
            segment = cleared.offer.offer_segment
            if segment not in segment_totals:
                continue
            totals = segment_totals[segment]
            offered = cleared.offer.offered_energy_mwh
            totals["offered"] += offered
            totals["accepted"] += cleared.accepted_energy_mwh
            totals["price"] += cleared.offer.bid_price_eur_per_mwh * offered

    def average_price(segment: str) -> float:
        totals = segment_totals[segment]
        return totals["price"] / totals["offered"] if totals["offered"] else 0.0

    def acceptance_ratio(segment: str) -> float:
        totals = segment_totals[segment]
        return totals["accepted"] / totals["offered"] if totals["offered"] else 0.0

    return _EpisodeMetrics(
        run=run,
        phase=phase,
        episode=episode,
        total_reward=total_reward,
        discounted_reward=discounted_reward,
        total_profit_eur=total_profit,
        accepted_energy_mwh=accepted_energy,
        minimum_segment_average_bid_eur_per_mwh=average_price("learning_minimum"),
        flexible_segment_average_bid_eur_per_mwh=average_price("learning_flexible"),
        minimum_segment_acceptance_ratio=acceptance_ratio("learning_minimum"),
        flexible_segment_acceptance_ratio=acceptance_ratio("learning_flexible"),
    )


def _write_metrics(path: Path, metrics: list[_EpisodeMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["run", "phase", "episode", *LEARNING_METRIC_FIELDS]
        )
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "run": item.run,
                    "phase": item.phase,
                    "episode": item.episode,
                    **{
                        name: f"{getattr(item, name):.{decimals}f}"
                        for name, decimals in LEARNING_METRIC_DECIMALS.items()
                    },
                }
            )


def _write_run_summary(path: Path, metrics: list[_EpisodeMetrics]) -> None:
    """Write final evaluation means and population standard deviations."""

    evaluations = [item for item in metrics if item.phase in {"evaluation", "baseline"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["phase", "run_count"]
    for name in LEARNING_METRIC_FIELDS:
        fieldnames.extend((f"{name}_mean", f"{name}_std"))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for phase in ("evaluation", "baseline"):
            phase_metrics = [item for item in evaluations if item.phase == phase]
            if not phase_metrics:
                continue
            row: dict[str, str | int] = {
                "phase": phase,
                "run_count": len(phase_metrics),
            }
            for name in LEARNING_METRIC_FIELDS:
                values = [float(getattr(item, name)) for item in phase_metrics]
                row[f"{name}_mean"] = f"{fmean(values):.12f}"
                row[f"{name}_std"] = f"{pstdev(values):.12f}"
            writer.writerow(row)


def _checkpoint_from_setting(path: str | None, fallback: Path, name: str) -> Path:
    if path is None:
        return fallback / name
    configured = Path(path)
    return configured if configured.suffix == ".pt" else configured / name


def _restore_best_checkpoint(
    *,
    restored_path: Path,
    destination: Path,
    expected_reward: float,
    expected_seed: int | None,
    expected_load_base_mw: float,
) -> None:
    """Copy the genuine companion best checkpoint without changing the Agent."""

    candidates = [destination]
    companion = (
        restored_path
        if restored_path.name == "best.pt"
        else restored_path.parent / "best.pt"
    )
    if companion not in candidates:
        candidates.append(companion)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = _read_checkpoint_payload(candidate, "cpu")
        saved_reward = payload.get("validation_discounted_reward")
        try:
            matches = (
                saved_reward is not None and float(saved_reward) == expected_reward
            )
            matches = (
                matches
                and float(payload["load_base_mw"]) == expected_load_base_mw
                and (expected_seed is None or payload.get("seed") == expected_seed)
            )
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            continue
        if candidate.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
        return
    raise InputValidationError(
        "Resume checkpoint records a historical best reward but no matching "
        f"best.pt is available beside {restored_path}."
    )


def _default_evaluation_checkpoint(config: LearningConfig, output_path: Path) -> Path:
    if config.trained_policies_load_path is not None:
        return _checkpoint_from_setting(
            config.trained_policies_load_path,
            output_path / "checkpoints",
            "best.pt",
        )
    if config.trained_policies_save_path is not None:
        return Path(config.trained_policies_save_path) / "best.pt"
    return output_path / "checkpoints" / "best.pt"


def _learning_config(settings: MarketSettings) -> LearningConfig:
    config = settings.learning_config
    if config is None or not config.learning_mode:
        raise InputValidationError(
            "Learning training/evaluation requires learning_config.learning_mode: true."
        )
    return config


def _independent_run_seed(base_seed: int | None, run: int) -> int:
    """Offset the scenario seed per run, or draw a recorded seed when it is null."""

    if base_seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    return base_seed + run - 1


def _marginal_cost_action(state: tuple[float, ...]) -> tuple[float, float]:
    """Quote both Version 5 segments at the observed marginal cost."""

    normalized_marginal_cost = state[-1]
    return normalized_marginal_cost, normalized_marginal_cost


def _evaluate_actor_and_baseline(
    *,
    agent: MATD3Agent,
    input_path: Path,
    output_path: Path,
    scenario: str,
    load_base_mw: float,
    episode: int,
    gamma: float,
    run: int,
) -> tuple[SimulationResult, list[_EpisodeMetrics]]:
    """Run deterministic Actor and marginal-cost baseline on fresh markets."""

    actor_runner = LearningEpisodeRunner(
        input_path, scenario=scenario, load_base_mw=load_base_mw
    )
    actor_result = actor_runner.run_episode(agent.deterministic_action)
    baseline_runner = LearningEpisodeRunner(
        input_path, scenario=scenario, load_base_mw=load_base_mw
    )
    baseline_result = baseline_runner.run_episode(
        _marginal_cost_action, enforce_action_bounds=False
    )

    write_results(output_path, actor_result)
    write_results(output_path / "baseline", baseline_result)
    comparison = [
        _metrics(
            actor_result,
            run=run,
            phase="evaluation",
            episode=episode,
            gamma=gamma,
        ),
        _metrics(
            baseline_result,
            run=run,
            phase="baseline",
            episode=episode,
            gamma=gamma,
        ),
    ]
    _write_metrics(output_path / "learning_evaluation_metrics.csv", comparison)
    return actor_result, comparison


def _train_single_run(
    input_path: Path,
    output_path: Path,
    checkpoint_root: Path,
    scenario: str = "base",
    *,
    run: int,
    seed: int,
    checkpoint_path: str | Path | None = None,
    training_episodes: int | None = None,
) -> tuple[SimulationResult, list[_EpisodeMetrics], Path, Path, float]:
    """Train and evaluate one statistically independent MATD3 run."""

    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = _learning_config(settings)
    episode_count = (
        config.training_episodes if training_episodes is None else training_episodes
    )
    if episode_count <= 0:
        raise InputValidationError("Training episodes must be positive.")
    steps_per_episode = sum(
        len(opening.products) for opening in market_openings(settings)
    )
    regular_episode_count = max(0, episode_count - config.initial_experience_episodes)
    initial_transition_count = (
        min(episode_count, config.initial_experience_episodes) * steps_per_episode
    )
    regular_transition_count = regular_episode_count * steps_per_episode
    warmup_transitions = max(0, config.batch_size - initial_transition_count - 1)
    eligible_regular_steps = max(0, regular_transition_count - warmup_transitions)
    agent = MATD3Agent(
        config,
        total_regular_steps=max(1, eligible_regular_steps),
        seed=seed,
        run=run,
    )
    latest_path = checkpoint_root / "latest.pt"
    best_path = checkpoint_root / "best.pt"

    restored: dict[str, Any] | None = None
    restored_path: Path | None = None
    requested_checkpoint = (
        Path(checkpoint_path) if checkpoint_path is not None else None
    )
    if requested_checkpoint is not None:
        restored_path = requested_checkpoint
        restored = agent.load_checkpoint(requested_checkpoint, load_optimizers=True)
    elif config.continue_learning:
        load_path = _checkpoint_from_setting(
            config.trained_policies_load_path,
            checkpoint_root,
            "latest.pt",
        )
        restored_path = load_path
        restored = agent.load_checkpoint(load_path, load_optimizers=True)

    runner = LearningEpisodeRunner(
        input_path,
        scenario=scenario,
        load_base_mw=(
            float(restored["load_base_mw"]) if restored is not None else None
        ),
    )
    metrics: list[_EpisodeMetrics] = []
    best_discounted_reward = float("-inf")
    first_episode = 1
    if restored is not None:
        first_episode = int(restored["episode"]) + 1
    if (
        restored is not None
        and restored.get("validation_discounted_reward") is not None
    ):
        best_discounted_reward = float(restored["validation_discounted_reward"])
        assert restored_path is not None
        _restore_best_checkpoint(
            restored_path=restored_path,
            destination=best_path,
            expected_reward=best_discounted_reward,
            expected_seed=(
                int(restored["seed"]) if restored.get("seed") is not None else None
            ),
            expected_load_base_mw=float(restored["load_base_mw"]),
        )
    if best_discounted_reward == float("-inf"):
        # No historical best is being continued: drop a stale best.pt left by
        # an unrelated earlier training so it cannot pass as this run's best.
        best_path.unlink(missing_ok=True)

    for episode in range(first_episode, episode_count + 1):
        initial_experience = episode <= config.initial_experience_episodes
        training_result = runner.run_episode(
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
        assert runner.load_base_mw is not None
        should_validate = (
            episode % config.validation_interval == 0 or episode == episode_count
        )
        if should_validate:
            validation_result = runner.run_episode(agent.deterministic_action)
            validation_metrics = _metrics(
                validation_result,
                run=run,
                phase="validation",
                episode=episode,
                gamma=config.gamma,
            )
            metrics.append(validation_metrics)
            if (
                not initial_experience
                and validation_metrics.discounted_reward > best_discounted_reward
            ):
                best_discounted_reward = validation_metrics.discounted_reward
                agent.save_checkpoint(
                    best_path,
                    episode=episode,
                    load_base_mw=runner.load_base_mw,
                    validation_discounted_reward=best_discounted_reward,
                )
        agent.save_checkpoint(
            latest_path,
            episode=episode,
            load_base_mw=runner.load_base_mw,
            validation_discounted_reward=(
                None
                if best_discounted_reward == float("-inf")
                else best_discounted_reward
            ),
        )

    if best_discounted_reward == float("-inf") or not best_path.is_file():
        raise InputValidationError(
            "Training did not produce a best checkpoint; increase training_episodes "
            "or reduce episodes_collecting_initial_experience."
        )

    best_payload = agent.load_checkpoint(best_path, load_optimizers=False)
    evaluation_result, comparison = _evaluate_actor_and_baseline(
        agent=agent,
        input_path=input_path,
        output_path=output_path,
        scenario=scenario,
        load_base_mw=float(best_payload["load_base_mw"]),
        episode=int(best_payload["episode"]),
        gamma=config.gamma,
        run=run,
    )
    metrics.extend(comparison)
    _write_metrics(output_path / "learning_metrics.csv", metrics)
    return (
        evaluation_result,
        metrics,
        latest_path,
        best_path,
        best_discounted_reward,
    )


def train_learning_scenario(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
    *,
    checkpoint_path: str | Path | None = None,
    training_episodes: int | None = None,
    independent_runs: int | None = None,
) -> SimulationResult:
    """Train independent MATD3 runs and evaluate the best validated Actor."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = _learning_config(settings)
    if independent_runs is None:
        run_count = (
            1
            if checkpoint_path is not None or config.continue_learning
            else _DEFAULT_INDEPENDENT_RUNS
        )
    else:
        run_count = independent_runs
    if run_count != 1 and run_count < _DEFAULT_INDEPENDENT_RUNS:
        raise InputValidationError(
            "Independent learning runs must be 1 for resume/debug or at least 3 "
            "for Version 5 reporting."
        )
    if run_count > 1 and (checkpoint_path is not None or config.continue_learning):
        raise InputValidationError(
            "Resume training operates on one checkpoint; use independent_runs=1."
        )

    checkpoint_root = (
        Path(config.trained_policies_save_path)
        if config.trained_policies_save_path is not None
        else output_path / "checkpoints"
    )
    all_metrics: list[_EpisodeMetrics] = []
    selected_best_path: Path | None = None
    selected_best_reward = float("-inf")
    final_result: SimulationResult | None = None

    for run in range(1, run_count + 1):
        run_output = (
            output_path if run_count == 1 else output_path / "runs" / f"run_{run:02d}"
        )
        run_checkpoints = (
            checkpoint_root if run_count == 1 else checkpoint_root / f"run_{run:02d}"
        )
        result, metrics, latest_path, best_path, best_reward = _train_single_run(
            input_path,
            run_output,
            run_checkpoints,
            scenario,
            run=run,
            seed=_independent_run_seed(settings.seed, run),
            checkpoint_path=checkpoint_path,
            training_episodes=training_episodes,
        )
        final_result = result
        all_metrics.extend(metrics)
        if run_count > 1:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_path, checkpoint_root / "latest.pt")
        if selected_best_path is None or best_reward > selected_best_reward:
            selected_best_reward = best_reward
            selected_best_path = best_path

    assert final_result is not None and selected_best_path is not None
    global_best_path = checkpoint_root / "best.pt"
    if selected_best_path != global_best_path:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_best_path, global_best_path)
        final_result = evaluate_learning_scenario(
            input_path,
            output_path,
            scenario=scenario,
            checkpoint_path=global_best_path,
        )

    _write_metrics(output_path / "learning_metrics.csv", all_metrics)
    _write_run_summary(output_path / "learning_run_summary.csv", all_metrics)
    return final_result


def evaluate_learning_scenario(
    input_dir: str | Path,
    output_dir: str | Path,
    scenario: str = "base",
    *,
    checkpoint_path: str | Path | None = None,
) -> SimulationResult:
    """Compare the best Actor with an update-free marginal-cost baseline."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    settings = load_market_settings(input_path / "config.yaml", scenario=scenario)
    config = _learning_config(settings)
    agent = MATD3Agent(
        config,
        total_regular_steps=max(
            1,
            config.training_episodes
            * sum(len(opening.products) for opening in market_openings(settings)),
        ),
    )
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else _default_evaluation_checkpoint(config, output_path)
    )
    payload = agent.load_checkpoint(checkpoint, load_optimizers=False)
    saved_run = payload.get("run")
    run = (
        saved_run
        if isinstance(saved_run, int) and not isinstance(saved_run, bool)
        else 0
    )
    result, _ = _evaluate_actor_and_baseline(
        agent=agent,
        input_path=input_path,
        output_path=output_path,
        scenario=scenario,
        load_base_mw=float(payload["load_base_mw"]),
        episode=int(payload["episode"]),
        gamma=config.gamma,
        run=run,
    )
    return result
