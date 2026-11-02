import gymnasium as gym
import torch
from ssl_strategy_api.run_demos.run_rsoccer_exercise1 import Exercise1Env
from ssl_strategy_api.strategies.striker import StrikerStrategy  # Deine Striker-Strategie
from rsoccer_gym.Entities import Robot  # Falls du rSoccer-Roboter direkt übergibst

def main():

    env = Exercise1Env(render_mode="human")
    
    striker = StrikerStrategy(model_path="striker_model.pth")
    
    print("Starte Trainings-Schleife...")

    episode_rewards = []

    for episode in range(1, 6):  # 5 Episoden als Beispiel
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # 3. Hole die Intents/Aktionen von deiner Strategie
            # (Je nachdem, ob deine decide() Methode direkt rSoccer-Roboter oder RobotIntents zurückgibt)
            actions = striker.decide(env.frame, env.blue_config)
            commands = []
            for robot_id, intent in actions.items():
                commands.append(
                    Robot(
                        yellow=False,  # <--- HIER kommt das yellow-Attribut hin!
                        id=robot_id,
                        v_x=intent.vx_mps,
                        v_y=intent.vy_mps,
                        v_theta=intent.angular_velocity_rad_s,
                        dribbler=False
                    )
                )
            obs, reward, terminated, truncated, info = env.step(commands)
            
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {episode}: Total Reward: {total_reward}")

    env.close()

    model_path = "striker_model.pth"
    torch.save({
        "model_state_dict": striker.model.state_dict(),
        "episode_rewards": episode_rewards
    }, model_path)
    
    print(f"Training beendet! Modell unter '{model_path}' gespeichert.")

if __name__ == "__main__":
    main()