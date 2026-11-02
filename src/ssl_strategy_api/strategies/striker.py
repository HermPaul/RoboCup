import math
from typing import Mapping
import torch
import torch.nn as nn
import numpy as np

from ssl_strategy_api import RobotState
from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim: int = 12, action_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

class StrikerStrategy:
    """Student placeholder for Exercise 1.

    Replace this strategy with your own implementation.

    The default behavior below is intentionally simple and deliberately not optimal:
    it demonstrates how to access the controlled robot and return a valid mapping.
    """
    def __init__(self, model_path: str = "model.pth"):
        self.model = ActorCriticNet()
        self.action_dim = 3  # vx, vy, angular_velocity # TODO add kicker and 
        self.state_dim = 12  # state_vector
        self.model.eval()

    def reset(self, seed=None) -> None:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        """Simple starter example for students.

        You must change this strategy to implement your own striker behavior.

        This example shows the basic structure:
        - check whether any robots are controlled
        - select the robot for the blue team
        - get its state from the world
        - return a mapping for that robot id

        In this exercise, the blue team is the striker and uses robot id 0 by default.
        """
        if not config.controlled_robot_ids:
            return {}

        if config.team != TeamColor.BLUE:
            return {}

        robot_id = config.controlled_robot_ids[0]
        robots = state.robots_blue
        robot = robots.get(robot_id)
        if robot is None:
            return {}

        obs = self.observe(state, config, robot)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Forward pass through the model to get action and value
        with torch.no_grad():
            action_logits, _ = self.model(obs_tensor)
            action = action_logits.squeeze(0).numpy()  # Remove batch dimension
        vx = float(action[0])
        vy = float(action[1])
        v_theta = float(action[2])

        # Clip velocities to safe operating ranges for rSoccer simulation
        vx = max(-1.5, min(1.5, vx))
        vy = max(-1.5, min(1.5, vy))
        v_theta = max(-3.0, min(3.0, v_theta))

        return {
            robot_id: RobotIntent(
                vx_mps=vx,
                vy_mps=vy,
                angular_velocity_rad_s=v_theta,
                kick_speed_mps=5 if robot.infrared else 0,
            )
        }


    def observe(self, state: WorldState, config: TeamConfig, striker: RobotState) -> np.ndarray:

        rx = striker.x
        ry = striker.y
        r_vx = striker.v_x
        r_vy = striker.v_y
        r_heading = striker.theta
        infrared = striker.infrared

        ball = state.ball
        bx = ball.x
        by = ball.y
        b_vx = ball.v_x
        b_vy = ball.v_y

        state_vector = np.array([
            rx, ry, r_vx, r_vy,
            bx, by, b_vx, b_vy,
            bx - rx, by - ry, # distance robot ball
            r_heading, 
            infrared,
        ], dtype=np.float32)

        return state_vector


