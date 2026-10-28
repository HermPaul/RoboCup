# SSL Student Strategy API

This package provides a simulator-independent SSL strategy framework, where strategy commands use physical units (for example, linear velocities in meters per second and angular velocities in radians per second). It has no runtime dependencies and requires Python 3.10 or newer.

First create a folder (for example `robocup_project`), then clone the rSoccer repository, then clone this `ssl_strategy_api` repository. After that, create `rsoccer_venv` inside the `rSoccer` repository, activate it, and install both packages:

```sh
mkdir -p robocup_project
cd robocup_project

git clone <RSOCCER_REPO_URL>
git clone <SSL_STRATEGY_API_REPO_URL>

cd rSoccer
python3.10 -m venv rsoccer_venv
source rsoccer_venv/bin/activate
python -m pip install -e .
python -m pip install --no-deps -e ../ssl_strategy_api
```

Replace `<RSOCCER_REPO_URL>` and `<SSL_STRATEGY_API_REPO_URL>` with the real repository URLs.

Note: use Python 3.10 or higher; ideally use python version 3.10 explicitly when creating the virtual environment.

`--no-deps` is safe for the shared API because it has no third-party dependencies; `-e` makes the venv import the package directly from this workspace, so local edits are immediately available.

## Strategy Interface

`WorldState` contains the ball, blue and yellow robot mappings, field geometry, and optional frame metadata. Positions are in meters, linear velocities are in m/s, headings are radians, and angular velocities are in rad/s. Velocity values may be absent; `velocity_source` distinguishes measured, estimated, and unavailable planar velocity. rSoccer provides you already with a velocity observation for the ball and robots. Though when you use other simulators (e.g. grSim) the velocities may need to be estimated by you.

Strategies implement the function `decide(state, config)`. This function gets the current state of the game (e.g. robot positions, ball position) and config (e.g. which team are we handling? for example YELLOW_TEAM, which robots are we controlling? e.g. controlled_robot_ids= (0,1)) as arguments and return a mapping from controlled robot IDs to `RobotIntent` (which is an action for each robot). Intents use field-frame velocities in physical units. Set `kick_speed_mps` to a value from 0 to 5; zero means no kick. The dribbler remains a boolean. For example, `RobotIntent(kick_speed_mps=2.5)` requests a 2.5 m/s kick.

`MotionLimits()` centrally caps planar speed at 2.0 m/s by vector magnitude and angular speed at 3.0 rad/s. For example, a request of `(vx_mps=2, vy_mps=2)` has magnitude about 2.83 m/s, so it is scaled to about `(1.41, 1.41)` while preserving direction. A task may configure lower limits, but not higher ones. Runners use this shared default rather than redefining limits at each command call. The rSoccer command adapter also rotates field-frame velocity into robot-relative command coordinates.

## Example Goalkeeper Environment

Initially, we provide a complete goalkeeper baseline in two versions:

- Rule-based goalkeeper: `src/ssl_strategy_api/strategies/goalkeeper.py`
- RL-based goalkeeper (Q-table style): `src/ssl_strategy_api/strategies/goalkeeper_rl.py`

You can run an example environment that uses the rule-based goalkeeper while a ball is repeatedly shot against it, so you can observe the baseline behavior:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_goalkeeper
```

## Run Exercise 1

Run Exercise 1 with:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1
```

In the default setup, blue controls robot id 0 with the striker strategy (student side), and yellow uses an idle goalkeeper strategy.

Choosing the goalkeeper strategy

In the Exercise 1 runner, the yellow strategy is selected through the `--yellow-strategy` argument.
By default, it uses the idle goalkeeper:

```text
default="ssl_strategy_api.strategies.idle_goalkeeper_rl:IdleGoalkeeperStrategy"
```

You can switch to other provided goalkeeper baselines by changing that default in the parser:

- Rule-based goalkeeper:

```text
default="ssl_strategy_api.strategies.goalkeeper:GoalkeeperStrategy"
```

- RL-based goalkeeper (Q-table):

```text
default="ssl_strategy_api.strategies.goalkeeper_rl:QTableGoalkeeperStrategy"
```

You can also choose the goalkeeper at runtime without editing code:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1 --yellow-strategy ssl_strategy_api.strategies.goalkeeper:GoalkeeperStrategy

python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1 --yellow-strategy ssl_strategy_api.strategies.goalkeeper_rl:QTableGoalkeeperStrategy
```

Rendering and duration options

If you do not want the simulator window to open, run in headless mode:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1 --no-render
```

This is useful for longer runs, batch evaluations, or RL training loops where visual rendering is unnecessary and can slow execution.

You can also control the maximum run duration:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1 --duration-seconds 300
```

The default duration is 90 seconds.

## Run Exercise 1 Evaluation

Run the Exercise 1 evaluation file with:

```sh
python -m ssl_strategy_api.run_demos.run_rsoccer_exercise1_eval
```


### Exercise 1 vs Exercise 1 Eval

`run_rsoccer_exercise1.py` and `run_rsoccer_exercise1_eval.py` are very similar, but they are used for different purposes.

- `run_rsoccer_exercise1.py` is episode-based. The episode stops when the ball goes out of the field bounds, a goal is scored, or the ball is idle in the opponent goal area. Then a new episode is started and the blue robot along with the ball are respawned in random positions in the field. This is useful for RL training episodes and quick checks of rule-based strategies.
- `run_rsoccer_exercise1_eval.py` is the final evaluation runner. The episode does not stop on those events. Instead, when the ball goes out of the field bounds, a goal is scored, or the ball is idle in the opponent goal area, the ball is immediately respawned at a random valid position on the field, while robot positions remain unchanged.
