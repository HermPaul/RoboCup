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

STD_CHECKPOINT = "src/ssl_strategy_api/strategies/weights/striker/latest.pt"

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
        self.dones: List[Tensor] = []

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

        if checkpoint_path is None:
            self.load_checkpoint(STD_CHECKPOINT)
        elif checkpoint_path != "new":
            self.load_checkpoint(checkpoint_path)

    def reset(self, seed=None) -> None:
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def record_step(self, step_rewards: np.ndarray, step_dones: np.ndarray) -> None:
        self.rewards.append(torch.tensor(step_rewards, dtype=torch.float32, device=self.device))
        self.dones.append(torch.tensor(step_dones, dtype=torch.float32, device=self.device))

    def decide_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        batch_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)

        dist = self.actor(batch_tensor)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        val = self.critic(batch_tensor).squeeze(-1)

        self.log_probs.append(log_prob)
        self.values.append(val)

        return action.detach().cpu().numpy()

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
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

    def finish_rollout(self, last_obs_batch: np.ndarray) -> None:
        if len(self.values) <= 1 or len(self.rewards) == 0:
            self.reset()
            return

        with torch.no_grad():
            last_tensor = torch.tensor(last_obs_batch, dtype=torch.float32, device=self.device)
            next_value = self.critic(last_tensor).squeeze(-1)

        values_t = torch.stack(self.values)
        log_probs_t = torch.stack(self.log_probs)
        rewards_t = torch.stack(self.rewards)
        dones_t = torch.stack(self.dones)

        returns, advantages = self.compute_gae_and_returns(
            rewards_t, values_t, dones_t, next_value
        )

        actor_loss = -(log_probs_t * advantages.detach()).mean()
        critic_loss = (returns.detach() - values_t).pow(2).mean()

        # 1. Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # 2. Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # 3. Step schedulers after optimizer.step()
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        self.reset()

    def observe(self, state: WorldState, config: TeamConfig) -> Optional[Tensor]:
        striker = state.robots_blue.get(config.controlled_robot_ids[0])
        goalkeeper = state.robots_yellow.get(0)
        ball = state.ball
        if striker is None or goalkeeper is None or ball is None:
            return None

        # Relative spatial calculations
        dx_ball = ball.x_m - striker.x_m
        dy_ball = ball.y_m - striker.y_m
        dist_ball = float(np.hypot(dx_ball, dy_ball))

        angle_ball = float(np.arctan2(dy_ball, dx_ball) - striker.heading_rad)
        angle_ball = float(np.arctan2(np.sin(angle_ball), np.cos(angle_ball)))

        goal_x = 4.5 * config.attack_direction_x
        dx_goal = goal_x - striker.x_m
        dy_goal = 0.0 - striker.y_m
        dist_goal = float(np.hypot(dx_goal, dy_goal))

        dx_gk = goalkeeper.x_m - striker.x_m
        dy_gk = goalkeeper.y_m - striker.y_m

        return torch.tensor([
            dx_ball, dy_ball, dist_ball, angle_ball,
            striker.heading_rad, striker.vx_mps, striker.vy_mps, striker.angular_velocity_rad_s,
            float(striker.infrared or 0),
            dx_goal, dy_goal, dist_goal,
            dx_gk, dy_gk, float(np.hypot(goalkeeper.vx_mps, goalkeeper.vy_mps))
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
            self,
            rewards: Tensor,
            values: Tensor,
            dones: Tensor,
            next_value: Tensor,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
    ) -> Tuple[Tensor, Tensor]:
        T, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.zeros(num_envs, device=self.device)

        values_detached = values.detach()

        for t in reversed(range(T)):
            v_next = next_value if t == T - 1 else values_detached[t + 1]
            non_terminal = 1.0 - dones[t].float()

            td_error = rewards[t] + gamma * v_next * non_terminal - values_detached[t]
            gae = td_error + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae

        returns = advantages + values

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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