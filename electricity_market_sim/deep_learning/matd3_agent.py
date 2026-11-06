"""MATD3 网络、经验回放、动作生成及参数更新。"""

from functools import partial

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..models import LearningConfig, LearningTransition
from .learning import ACTION_SIZE, OBSERVATION_SIZE
from .network_utils import _deterministic_action, _initialize_linear


class _Actor(nn.Module):
    def __init__(self):
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
    def __init__(self):
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

    def forward(self, state: Tensor, action: Tensor):
        inputs = torch.cat((state, action), dim=1)
        return self.q1(inputs), self.q2(inputs)


class _ReplayBuffer:
    def __init__(self, capacity, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.values = np.empty(
            (capacity, 2 * OBSERVATION_SIZE + ACTION_SIZE + 2), dtype=np.float32
        )
        self.position = 0
        self.size = 0

    def append(self, transition: LearningTransition):
        self.values[self.position] = (
            *transition.state, *transition.action, transition.reward,
            *transition.next_state, transition.done,
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device: torch.device):
        indices = self.rng.integers(0, self.size, size=batch_size)
        return torch.as_tensor(self.values[indices], device=device).split(
            (OBSERVATION_SIZE, ACTION_SIZE, 1, OBSERVATION_SIZE, 1), dim=1
        )


class MATD3Agent:
    """第五版 MATD3 的单智能体实现。"""
    def __init__(
        self,
        config: LearningConfig,
        *,
        total_regular_steps,
        seed,
        run,
    ):
        self.device = torch.device(config.device)
        self.config = config
        self.seed = seed
        self.run = run
        self.total_regular_steps = total_regular_steps
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.actor = _Actor().to(self.device)
        # 生成无探索噪声的报价函数。
        self.deterministic_action = partial(_deterministic_action, self.actor, self.device)
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

    def exploration_action(self, state, *, initial_experience):
        if initial_experience:
            # 初始阶段以归一化边际成本作为两段报价的基准。
            action = np.full(ACTION_SIZE, state[-1], dtype=np.float64)
            standard_deviation = self.config.exploration_noise_std
        else:
            # 后续使用策略网络报价，探索噪声随训练进度衰减。
            action = np.asarray(self.deterministic_action(state), dtype=np.float64)
            standard_deviation = (
                self.config.noise_sigma * self.config.noise_scale * self.noise_factor
            )
        # 加入高斯噪声，探索不同报价。
        action += self.rng.normal(0.0, standard_deviation, size=ACTION_SIZE)
        return float(action[0]), float(action[1])

    def observe(self, transition, *, train):
        # 保存本小时的训练记录；收集阶段或样本不足时不更新网络。
        self.replay.append(transition)
        if not train or self.replay.size < self.config.batch_size:
            return
        self.regular_steps += 1
        # 达到设定的交互步数间隔后再训练。
        if self.regular_steps % self.config.train_frequency_steps != 0:
            return
        # 按训练进度衰减探索噪声，再执行指定次数的参数更新。
        progress = self.regular_steps / self.total_regular_steps
        self.noise_factor = self.config.noise_dt * (1.0 - progress)
        for _ in range(self.config.gradient_steps):
            self._gradient_step()

    def _gradient_step(self):
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size, self.device
        )
        # 计算价值网络的训练目标，此处不计算梯度。
        with torch.no_grad():
            # 给下一小时的目标报价动作加入受限噪声，平滑收益估计。
            noise = torch.randn_like(actions) * self.config.target_policy_noise
            noise.clamp_(-self.config.target_noise_clip, self.config.target_noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            # 两个目标价值网络估计后续累计奖励，取较小值减少高估。
            target_q1, target_q2 = self.critics_target(next_states, next_actions)
            # 当前奖励加折扣后的后续奖励；回合结束时不计后续奖励。
            target_q = rewards + self.config.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)

        current_q1, current_q2 = self.critics(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward() # TODO: 反向传播
        nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.gradient_updates += 1
        if self.gradient_updates % self.config.policy_delay != 0:
            return
        for parameter in self.critics.parameters():
            parameter.requires_grad_(False)

        actor_actions = self.actor(states)
        critic_inputs = torch.cat((states, actor_actions), dim=1)
        actor_loss = -self.critics.q1(critic_inputs).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward() # TODO: 反向传播
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        for parameter in self.critics.parameters():
            parameter.requires_grad_(True)
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critics, self.critics_target)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        # 目标参数 = (1 - tau) × 原目标参数 + tau × 当前训练网络参数。
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(source_parameter, alpha=self.config.tau)
