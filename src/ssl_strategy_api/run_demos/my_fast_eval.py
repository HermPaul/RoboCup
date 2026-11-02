from ssl_strategy_api.adapters.rsoccer import to_rsoccer_commands
from ssl_strategy_api.run_demos.run_rsoccer_exercise1_eval import Exercise1EvalEnv
from ssl_strategy_api.strategies.striker import StrikerStrategy
from ssl_strategy_api.strategies.idle_goalkeeper_rl import IdleGoalkeeperStrategy
from ssl_strategy_api.strategies.goalkeeper import GoalkeeperStrategy
from ssl_strategy_api.strategies.goalkeeper_rl import QTableGoalkeeperStrategy

STEPS = 3600   # 90 s / 25 ms


def count_goals(striker, goalie, seed):
    env = Exercise1EvalEnv(spawner_seed=seed)          # ohne render_mode = kein Fenster
    state, _ = env.reset(seed=seed)
    striker.reset()
    goals = 0
    for _ in range(STEPS):
        commands = to_rsoccer_commands(state, env.blue_config, striker.decide(state, env.blue_config))
        commands += to_rsoccer_commands(state, env.yellow_config, goalie.decide(state, env.yellow_config))
        state, *_ = env.step(commands)

        reason = env.ball_spawner.respawn_when_goal_or_out_or_idle(
            state.ball, env.field, attack_direction_x=1)
        if reason == "goal":
            goals += 1
        if reason is not None:                     # Tor, Aus oder Ball liegt im Strafraum
            env.respawn_ball_keep_robots()         # neuer Ball, Roboter bleibt stehen
            state = env._frame_to_observations()
    env.close()
    return goals


if __name__ == "__main__":
    goalies = {"idle": IdleGoalkeeperStrategy, "rule-based": GoalkeeperStrategy, "q-table": QTableGoalkeeperStrategy}
    for name, Goalie in goalies.items():
        results = [count_goals(StrikerStrategy(debug=False), Goalie(), seed) for seed in range(1, 6)]
        print(name, results, "Mean:", sum(results) / len(results))