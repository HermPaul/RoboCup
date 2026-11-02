from ssl_strategy_api.adapters.rsoccer import to_rsoccer_commands
from ssl_strategy_api.run_demos.run_rsoccer_exercise1_eval import Exercise1EvalEnv
from ssl_strategy_api.strategies.striker import StrikerStrategy
from ssl_strategy_api.strategies.idle_goalkeeper_rl import IdleGoalkeeperStrategy
from ssl_strategy_api.strategies.goalkeeper import GoalkeeperStrategy
from ssl_strategy_api.strategies.goalkeeper_rl import QTableGoalkeeperStrategy
from collections import Counter
import statistics

STEPS = 3600   # 90 s / 25 ms


def run_test_episode(striker, goalie, seed):
    env = Exercise1EvalEnv(spawner_seed=seed)          # without render_mode = no window
    state, _ = env.reset(seed=seed)
    striker.reset()
    stats = Counter()
    half_length = env.field.length / 2
    
    for _ in range(STEPS):
        commands = to_rsoccer_commands(state, env.blue_config, striker.decide(state, env.blue_config))
        commands += to_rsoccer_commands(state, env.yellow_config, goalie.decide(state, env.yellow_config))
        state, *_ = env.step(commands)

        reason = env.ball_spawner.respawn_when_goal_or_out_or_idle(
            state.ball, env.field, attack_direction_x=1)
        
        if reason is None:
            continue

        stats[reason] += 1

        if reason == "out-of-bounds":
            stats[classify_out_reason(state, env.blue_config, half_length)] += 1
        
        env.respawn_ball_keep_robots()         # new ball, Robot stays at position
        state = env._frame_to_observations()

    env.close()
    return stats

def classify_out_reason(state, config, half_length):
    ball = state.ball
    robot_id = config.controlled_robot_ids[0]
    robot = state.robots_blue.get(robot_id)
    dx = robot.x_m - ball.x_m
    dy = robot.y_m - ball.y_m
    distance_robo_ball = (dx * dx + dy * dy) ** 0.5

    if robot is None:
        return "other out"

    if distance_robo_ball < 0.2:
        return "robot with ball out"
    if ball.x_m > half_length - 0.1:
        return "missed the goal wide to the left" if ball.y_m > 0 else "missed the goal wide to the right"
    return "other out"


if __name__ == "__main__":
    goalies = {"idle": IdleGoalkeeperStrategy, "rule-based": GoalkeeperStrategy, "q-table": QTableGoalkeeperStrategy}
    for name, Goalie in goalies.items():
        total = Counter()
        phases = Counter()
        goals = []
        for seed in range(1, 51):
            print(f"\r{name}: Spiel {seed}/50", end="", flush=True)
            striker = StrikerStrategy(debug=False)
            s = run_test_episode(striker, Goalie(), seed)
            phases += striker.phase_steps
            goals.append(s["goal"])
            total += s
        
        mean = sum(goals) / len(goals)
        sd = statistics.stdev(goals)
        print(f"\r{name}: {mean:.1f} ± {sd / len(goals) ** 0.5:.1f} Tore", dict(total),)
        sum_steps = sum(phases.values())
        print(" " + ", ".join(f"{phase} {steps / sum_steps:.0%}" for phase, steps in phases.most_common()))
        print(" " + ", ".join(f"{phase} {steps}" for phase, steps in phases.most_common()))
