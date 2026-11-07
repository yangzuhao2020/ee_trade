"""MATD3 与 GRPO 共用的检查点保存、策略加载及运行清单读写。"""

import json
from functools import partial
from pathlib import Path
from typing import Any

import torch

from ..errors import InputValidationError
from ..models import LearningConfig
from ..utils import format_timestamp
from .learning import ACTION_SIZE, OBSERVATION_SIZE
from .matd3_agent import _Actor
from .network_utils import _deterministic_action

_CHECKPOINT_VERSION = 2 # 检查点版本
_STATE_VERSION = "v5-observation-16"
_RUN_MANIFEST = "runs.json"


def _periods_payload(config: LearningConfig):
    """将训练、验证和测试时段的起止时间转为字符串，供检查点保存。"""
    return {
        name: [format_timestamp(start), format_timestamp(end)]
        for name, (start, end) in config.periods.items()
    }


def save_checkpoint(agent, path, episode, validation_total_reward):
    config = agent.config
    grpo = config.algorithm == "grpo"
    payload = {
        "checkpoint_version": 3 if grpo else _CHECKPOINT_VERSION,
        "algorithm": config.algorithm,
        "state_version": _STATE_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "episode": episode,
        "validation_total_reward": validation_total_reward,
        "periods": _periods_payload(config),
        "max_bid_price": config.max_bid_price,
        "seed": agent.seed,
        "run": agent.run,
        "actor": agent.actor.state_dict(),
    }
    if grpo:
        payload.update(
            distribution="softsign-normal-v1",
            advantage_version="group-time-return-v1",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _write_run_manifest(
    checkpoint_root: Path,
    config: LearningConfig,
    runs: list[dict[str, Any]],
    selected_run: int,
):
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
        raise InputValidationError("运行清单中的算法与当前配置不一致，请为不同算法使用独立的输出目录。")
    if payload.get("periods") != _periods_payload(config):
        raise InputValidationError(
            f"运行清单 {path} 的训练、验证、测试时段与当前 "
            "learning_config.periods 不一致。"
        )
    if (
        type(payload.get("selected_run")) is not int
        or payload["selected_run"] not in {entry["run"] for entry in payload["runs"]}
    ):
        raise InputValidationError("运行清单中的 selected_run 必须是整数，且对应 runs 中已有的运行编号。")
    return payload


def _default_checkpoint_root(config: LearningConfig, output_path: Path) -> Path:
    if config.trained_policies_save_path is not None:
        return Path(config.trained_policies_save_path)
    return output_path / "checkpoints"


def load_policy(config: LearningConfig, path: Path):
    """读取并校验检查点，返回确定性 Actor 策略和元数据。"""
    device = torch.device(config.device)
    algorithm = config.algorithm
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
    if payload.get("max_bid_price") != config.max_bid_price:
        raise InputValidationError(
            f"Learning checkpoint {path} was trained with max_bid_price="
            f"{payload.get('max_bid_price')!r}, but learning_config.max_bid_price="
            f"{config.max_bid_price!r}."
        )
    if payload.get("periods") != _periods_payload(config):
        raise InputValidationError(
            f"Learning checkpoint {path} 的训练、验证、测试时段与当前 "
            "learning_config.periods 不一致。"
        )
    for field in ("run", "episode"):
        if type(payload.get(field)) is not int:
            raise InputValidationError(f"Learning checkpoint {path} must include an integer {field}.")
    if algorithm == "grpo":
        from .grpo_agent import GaussianActor
        actor = GaussianActor(config.initial_std)
    else:
        actor = _Actor()
    actor.to(device).eval()
    actor.load_state_dict(payload["actor"])

    return partial(_deterministic_action, actor, device), payload
