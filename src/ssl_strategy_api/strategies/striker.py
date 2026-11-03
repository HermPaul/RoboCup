import os
from typing import Mapping, Optional
import numpy as np
from rsoccer_gym.Render import ball
from rsoccer_gym.Render import ball
from rsoccer_gym.Render.robot import Robot
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Normal

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState


class StrikerStrategy:

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.log_probs = []
        self.values = []
        self.rewards = []  # Stores per-step rewards

        self.actor = Actor(state_dim=12, action_dim=5)
        self.critic = Critic(state_dim=12)
        #self.actor, self.critic = ActorCriticNet(state_dim=12, action_dim=5)
       # self.network = ActorCriticNet(state_dim=12, action_dim=5)
# If you need to access them specifically later, do something like:
        #self.actor = self.network.actor
        #self.critic = self.network.critic
        

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def reset(self, seed=None) -> None:
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

    def record_reward(self, reward: float) -> None:
        """Stores the intermediate step reward returned by env.step()."""
        self.rewards.append(reward)

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids or config.team != TeamColor.BLUE:
            return {}

        state_t = self.observe(state, config)
        if state_t is None:
            return {}

        dist = self.actor(state_t)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1).squeeze()
        val = self.critic(state_t).squeeze()

        self.log_probs.append(log_prob)
        self.values.append(val)

        controlled_id = config.controlled_robot_ids[0]

        # Get actual robot and ball objects
        striker = state.robots_blue.get(controlled_id)
        ball = state.ball
        return self.create_intents(action, controlled_id, striker, ball)

    def finish(self, final_reward: float = 0.0) -> None:
        """Trains the model using accumulated step-by-step trajectory rewards."""
        if len(self.values) <= 1 or len(self.rewards) == 0:
            self.reset()
            return

        values_t = torch.stack(self.values).squeeze()
        log_probs_t = torch.stack(self.log_probs).squeeze()

        # Compute GAE across the list of per-step rewards
        returns, advantages = self.compute_gae_and_returns(self.rewards, values_t)

        actor_loss = -(log_probs_t * advantages.detach()).mean()
        critic_loss = (returns.detach() - values_t).pow(2).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        self.reset()

    def observe(self, state: WorldState, config: TeamConfig) -> Tensor | None:
        """
            converts the world state to a 12-dimensional array used as input to the actor-critic network
        """

        striker = state.robots_blue.get(config.controlled_robot_ids[0])
        goalkeeper = state.robots_yellow.get(0)
        ball = state.ball
        if striker is None or goalkeeper is None or ball is None:
            return None

        return torch.tensor([
            striker.x_m, striker.y_m, striker.heading_rad, float(striker.infrared or 0),
            goalkeeper.x_m, goalkeeper.y_m, goalkeeper.heading_rad, float(goalkeeper.infrared or 0),
            ball.x_m, ball.y_m, ball.vx_mps, ball.vy_mps
        ], dtype=torch.float32).unsqueeze(0)

    def create_intents(self, action_t: torch.Tensor, robot_id: int, striker: Robot, ball: ball.Ball) -> Mapping[int, RobotIntent]:
        action = action_t.detach().numpy().flatten()
        dx = ball.x_m - striker.x_m
        dy = ball.y_m - striker.y_m

        distance = float(np.hypot(dx, dy))

        # Direction toward opponent goal
        goal_direction = np.arctan2(
            -striker.y_m,
            1.0 - striker.x_m
        )

        angle_error = goal_direction - striker.heading_rad

        angle_error = np.arctan2(
            np.sin(angle_error),
            np.cos(angle_error)
        )

        # Shoot when close enough to the ball
        # and approximately facing the opponent goal
        should_kick = (
            distance < 0.50
            and abs(angle_error) < 0.5
        )
        return {
            robot_id: RobotIntent(
                vx_mps=float(np.clip(action[0], -2.0, 2.0)),
                vy_mps=float(np.clip(action[1], -2.0, 2.0)),
                angular_velocity_rad_s=float(
                    np.clip(action[2], -3.0, 3.0)
                ),
                kick_speed_mps=4.0 if should_kick else 0.0,
                dribbler=distance < 0.50,
            )
        }


    def compute_gae_and_returns(
        self, rewards: list, values: Tensor, gamma=0.99, gae_lambda=0.95
    ) -> tuple[Tensor, Tensor]:
        advantages = []
        gae = 0.0

        values_detached = values.detach()

        for t in reversed(range(len(values_detached))):
            v_next = 0.0 if t == len(values_detached) - 1 else values_detached[t + 1].item()
            done_mask = 0.0 if (t == len(values_detached) - 1) else 1.0

            td_error = rewards[t] + gamma * v_next * done_mask - values_detached[t].item()
            gae = td_error + gae_lambda * gamma * done_mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        return returns, advantages

    def save_checkpoint(
            self,
            filepath: str,
            episode: int,
            metrics: dict = None
    ) -> None:
        """Saves models, optimizers, and training metadata to a single file."""
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
        """Loads weights and optimizer states. Returns the episode number."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

        checkpoint = torch.load(filepath, map_location="cpu")

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
    def __init__(self, state_dim=12, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, action_dim)
        with torch.no_grad():
            self.mean.bias[3] = 1.0
        # Reduced initial exploration noise (e^-0.5 ~ 0.6) to prevent directional chatter
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, state: Tensor) -> Normal:
        mean = self.mean(self.net(state))
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim=12):
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
    