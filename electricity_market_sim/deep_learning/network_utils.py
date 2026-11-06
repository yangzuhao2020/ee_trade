"""MATD3 与 GRPO 共用的网络初始化和确定性动作计算。"""

import torch
from torch import nn


def _initialize_linear(module: nn.Module):
    # 线性层权重使用 Xavier 均匀初始化，偏置置零。
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


@torch.no_grad()
def _deterministic_action(
    actor: nn.Module, device: torch.device, state: tuple[float, ...]
) -> tuple[float, float]:
    observation = torch.as_tensor(state, dtype=torch.float32, device=device)
    return tuple(actor(observation).cpu().tolist())
