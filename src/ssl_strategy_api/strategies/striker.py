import os
from typing import List, Mapping, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch import Tensor
from torch.distributions import Normal

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState


class StrikerStrategy:

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        total_episodes: int = 500000,
        num_envs: int = 1,
    ):
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_probs: List[Tensor] = []
        self.values: List[Tensor] = []
        self.rewards: List[Tensor] = []

        self.actor = Actor(state_dim=15, action_dim=5).to(self.device)
        self.critic = Critic(state_dim=15).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.actor_scheduler = LinearLR(
            self.actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_episodes
        )
        self.critic_scheduler = LinearLR(
            self.critic_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_episodes
        )

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def reset(self, seed=None) -> None:
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

    def record_rewards(self, step_rewards: np.ndarray) -> None:
        """Stores a vector of step rewards from all parallel environments."""
        self.rewards.append(torch.tensor(step_rewards, dtype=torch.float32, device=self.device))

    def record_reward(self, reward: float) -> None:
        """Single-environment fallback."""
        self.rewards.append(torch.tensor([reward], dtype=torch.float32, device=self.device))

    def decide_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Processes (num_envs, 15) observations in one batched GPU forward pass and returns (num_envs, 5) actions."""
        batch_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)

        dist = self.actor(batch_tensor)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        val = self.critic(batch_tensor).squeeze(-1)

        self.log_probs.append(log_prob)
        self.values.append(val)

        return action.detach().cpu().numpy()

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        """Single-environment evaluation interface."""
        if not config.controlled_robot_ids or config.team != TeamColor.BLUE:
            return {}

        state_t = self.observe(state, config)
        if state_t is None:
            return {}

        dist = self.actor(state_t.to(self.device))
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1).squeeze()
        val = self.critic(state_t.to(self.device)).squeeze()

        self.log_probs.append(log_prob)
        self.values.append(val)

        controlled_id = config.controlled_robot_ids[0]
        return self.create_intents(action, controlled_id)

    def finish(self) -> None:
        if len(self.values) <= 1 or len(self.rewards) == 0:
            self.reset()
            return

        values_t = torch.stack(self.values)
        log_probs_t = torch.stack(self.log_probs)
        rewards_t = torch.stack(self.rewards)

        returns, advantages = self.compute_gae_and_returns(rewards_t, values_t)

        actor_loss = -(log_probs_t * advantages.detach()).mean()
        critic_loss = (returns.detach() - values_t).pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        self.reset()

    def observe(self, state: WorldState, config: TeamConfig) -> Optional[Tensor]:
        striker = state.robots_blue.get(config.controlled_robot_ids[0])
        goalkeeper = state.robots_yellow.get(0)
        ball = state.ball
        if striker is None or goalkeeper is None or ball is None:
            return None

        return torch.tensor([
            striker.x_m, striker.y_m, striker.heading_rad, striker.vx_mps, striker.vy_mps,
            striker.angular_velocity_rad_s, float(striker.infrared or 0),
            goalkeeper.x_m, goalkeeper.y_m, goalkeeper.vx_mps, goalkeeper.vy_mps,
            ball.x_m, ball.y_m, ball.vx_mps, ball.vy_mps
        ], dtype=torch.float32).unsqueeze(0)

    def create_intents(self, action_t: Tensor, robot_id: int) -> Mapping[int, RobotIntent]:
        action = action_t.detach().cpu().numpy().flatten()
        return {
            robot_id: RobotIntent(
                vx_mps=float(np.clip(action[0], -2.0, 2.0)),
                vy_mps=float(np.clip(action[1], -2.0, 2.0)),
                angular_velocity_rad_s=float(np.clip(action[2], -3.0, 3.0)),
                kick_speed_mps=float(np.clip(action[3], 0.0, 5.0)),
                dribbler=bool(action[4] > 0.0),
            )
        }

    def compute_gae_and_returns(
        self, rewards: Tensor, values: Tensor, gamma=0.99, gae_lambda=0.95
    ) -> Tuple[Tensor, Tensor]:
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
            values = values.unsqueeze(1)
            is_1d = True
        else:
            is_1d = False

        T, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.zeros(num_envs, device=self.device)

        values_detached = values.detach()

        for t in reversed(range(T)):
            v_next = 0.0 if t == T - 1 else values_detached[t + 1]
            done_mask = 0.0 if t == T - 1 else 1.0

            td_error = rewards[t] + gamma * v_next * done_mask - values_detached[t]
            gae = td_error + gae_lambda * gamma * done_mask * gae
            advantages[t] = gae

        returns = advantages + values

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if is_1d:
            returns = returns.squeeze(1)
            advantages = advantages.squeeze(1)

        return returns, advantages

    def save_checkpoint(self, filepath: str, episode: int, metrics: dict = None) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "role": "striker",
            "episode": episode,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "metrics": metrics or {},
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, eval_mode: bool = False) -> int:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        if eval_mode:
            self.actor.eval()
            self.critic.eval()
        else:
            self.actor.train()
            self.critic.train()

        return checkpoint.get("episode", 0)


class Actor(nn.Module):
    def __init__(self, state_dim=15, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, state: Tensor) -> Normal:
        mean = self.mean(self.net(state))
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)