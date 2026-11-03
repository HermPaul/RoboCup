"""
Progress of our rule-based striker (RoboCup Assignment 1).

Data: results.txt, plus step 12 from our chat (only idle was measured there).
Run:  python plot_progress.py
Out:  three PNGs in ./plots next to this file, then the plot windows open.

To add a new test: copy the last entry in STEPS and paste your new numbers.
"""

import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HZ = 40                      # sim steps per second (100 steps = 2.5 s)
GOALIES = ["idle", "rule-based", "q-table"]
GOALIE_COLORS = {"idle": "#2a78d6", "rule-based": "#eb6834", "q-table": "#1baf7a"}  # colorblind-safe
REVERTED_COLOR = "#C44E52"


def five_seeds(runs):
    """Early tests: goals per game for 5 seeds."""
    return {"mean": statistics.mean(runs),
            "sem": statistics.stdev(runs) / math.sqrt(len(runs))}


def fifty_seeds(mean, sem, reasons, phases=None, phase_goals=None):
    """Later tests: mean ± standard error as printed by the eval (50 seeds)."""
    return {"mean": mean, "sem": sem, "reasons": reasons, "phases": phases,
            "phase_goals": phase_goals or reasons["goal"]}


def ph(APPROACH, CARRY, TURNING, SHOOTING):
    return {"APPROACH": APPROACH, "CARRY": CARRY, "TURNING": TURNING, "SHOOTING": SHOOTING}


L, R = "missed the goal wide to the left", "missed the goal wide to the right"
OOB, BALL_OUT, OTHER = "out-of-bounds", "robot with ball out", "other out"
AREA = "idle-in-opponent-goal-area"

STEPS = [
    {"label": "baseline", "kept": True, "seeds": 5, "res": {
        "idle":       five_seeds([7, 11, 12, 13, 12]),
        "rule-based": five_seeds([11, 17, 12, 13, 12]),
        "q-table":    five_seeds([3, 4, 3, 9, 5])}},
    {"label": "aim closer to center", "kept": True, "seeds": 5, "res": {
        "idle":       five_seeds([10, 14, 13, 17, 12]),
        "rule-based": five_seeds([12, 15, 16, 15, 13]),
        "q-table":    five_seeds([8, 6, 9, 11, 7])}},
    {"label": "empty goal: aim at center", "kept": True, "seeds": 5, "res": {
        "idle":       five_seeds([19, 17, 20, 19, 18]),
        "rule-based": five_seeds([12, 15, 16, 15, 13]),
        "q-table":    five_seeds([8, 6, 8, 11, 7])}},
    {"label": "shoot only when aligned", "kept": True, "seeds": 5, "res": {
        "idle":       five_seeds([17, 17, 18, 19, 18]),
        "rule-based": five_seeds([16, 17, 17, 16, 17]),
        "q-table":    five_seeds([16, 17, 18, 15, 16])}},
    {"label": "stay steady 0.125 s", "kept": True, "seeds": 5, "res": {
        "idle":       five_seeds([17, 17, 17, 18, 17]),
        "rule-based": five_seeds([16, 15, 15, 16, 16]),
        "q-table":    five_seeds([16, 17, 17, 18, 14])}},
    # Same code as step 5, now 50 seeds. Phase times come from the 5-seed phase check.
    {"label": "same code, 50 seeds", "kept": True, "seeds": 50, "res": {
        "idle": fifty_seeds(16.8, 0.4, {"goal": 842, OOB: 2, BALL_OUT: 2, AREA: 2},
                            ph(7756, 5947, 4210, 87), phase_goals=86),
        "rule-based": fifty_seeds(15.7, 0.4, {"goal": 785, OOB: 17, L: 5, R: 8, BALL_OUT: 3,
                                              AREA: 7, OTHER: 1},
                                  ph(7893, 5943, 4075, 89), phase_goals=79),
        "q-table": fifty_seeds(16.4, 0.4, {"goal": 822, OOB: 10, BALL_OUT: 3, L: 2, AREA: 6, R: 5},
                               ph(8015, 5867, 4031, 87), phase_goals=81)}},
    {"label": "half brake distance", "kept": False, "seeds": 50, "res": {
        "idle": fifty_seeds(16.5, 0.1, {"goal": 823, OOB: 41, BALL_OUT: 41, AREA: 2},
                            ph(74672, 65299, 39206, 823)),
        "rule-based": fifty_seeds(15.5, 0.1, {"goal": 774, OOB: 54, BALL_OUT: 43, R: 9, AREA: 5, L: 2},
                                  ph(74588, 64200, 40396, 816)),
        "q-table": fifty_seeds(15.7, 0.2, {"goal": 783, AREA: 6, OOB: 48, BALL_OUT: 42, R: 1, L: 5},
                               ph(74346, 66242, 38594, 818))}},
    {"label": "direct shot on good spawn", "kept": False, "seeds": 50, "res": {
        "idle": fifty_seeds(17.5, 0.4, {"goal": 876, OOB: 19, BALL_OUT: 19, AREA: 9},
                            ph(89533, 38354, 51229, 884)),
        "rule-based": fifty_seeds(8.7, 0.4, {"goal": 435, AREA: 252, OOB: 20, L: 3, BALL_OUT: 12,
                                             R: 4, OTHER: 1},
                                  ph(92446, 37652, 49035, 867)),
        "q-table": fifty_seeds(7.4, 0.5, {AREA: 343, "goal": 368, OOB: 17, OTHER: 1, L: 3, BALL_OUT: 13},
                               ph(95149, 36369, 47668, 814))}},
    {"label": "direct shot only vs idle", "kept": True, "seeds": 50, "res": {
        "idle": fifty_seeds(17.0, 0.4, {"goal": 852, OOB: 4, R: 3, BALL_OUT: 1},
                            ph(77758, 61228, 40158, 856)),
        "rule-based": fifty_seeds(15.9, 0.4, {"goal": 795, OOB: 13, L: 6, R: 6, AREA: 5, BALL_OUT: 1},
                                  ph(77798, 60461, 40890, 851)),
        "q-table": fifty_seeds(16.3, 0.4, {"goal": 816, AREA: 12, OOB: 8, BALL_OUT: 2, L: 3, R: 3},
                               ph(78248, 62585, 38315, 852))}},
    {"label": "looser steady check", "kept": True, "seeds": 50, "res": {
        "idle": fifty_seeds(17.5, 0.4, {"goal": 876, OOB: 2, L: 1, AREA: 4, BALL_OUT: 1},
                            ph(80236, 62712, 36174, 878)),
        "rule-based": fifty_seeds(16.5, 0.4, {"goal": 826, OOB: 16, R: 11, AREA: 4, L: 4, BALL_OUT: 1},
                                  ph(80348, 61698, 37085, 869)),
        "q-table": fifty_seeds(16.5, 0.4, {"goal": 823, OOB: 19, L: 11, AREA: 16, R: 4, OTHER: 1,
                                           BALL_OUT: 3},
                               ph(81397, 62520, 35201, 882))}},
    {"label": "faster turning", "kept": True, "seeds": 50, "res": {
        "idle": fifty_seeds(18.8, 0.5, {"goal": 941, AREA: 7, OOB: 15, BALL_OUT: 13, OTHER: 2},
                            ph(95302, 42905, 40845, 948)),
        "rule-based": fifty_seeds(16.8, 0.5, {"goal": 842, OOB: 10, L: 3, AREA: 9, R: 5, BALL_OUT: 2},
                                  ph(84584, 65252, 29280, 884)),
        "q-table": fifty_seeds(17.1, 0.4, {"goal": 856, OOB: 25, R: 9, AREA: 9, BALL_OUT: 3, L: 12,
                                           OTHER: 1},
                               ph(84673, 64816, 29592, 919))}},
    # From our chat: only idle was measured.
    {"label": "bigger direct zone (idle only)", "kept": True, "seeds": 50, "res": {
        "idle": fifty_seeds(19.9, 0.4, {"goal": 993, OOB: 28, BALL_OUT: 27, AREA: 11, OTHER: 1},
                            ph(109781, 19333, 49883, 1003))}},
    {"label": "early shot while carrying", "kept": False, "seeds": 50, "res": {
        "idle": fifty_seeds(19.6, 0.4, {"goal": 980, OOB: 40, L: 3, BALL_OUT: 28, AREA: 13, R: 8,
                                        OTHER: 1},
                            ph(110829, 19036, 49156, 979)),
        "rule-based": fifty_seeds(15.8, 0.4, {"goal": 791, OOB: 35, L: 12, AREA: 36, R: 21, BALL_OUT: 2},
                                  ph(87104, 65431, 26669, 796)),
        "q-table": fifty_seeds(16.5, 0.4, {"goal": 823, OOB: 49, R: 25, L: 19, AREA: 18, BALL_OUT: 5},
                               ph(88274, 64525, 26391, 810))}},
    {"label": "no early shot, turn 0.4 m earlier", "kept": True, "seeds": 50, "res": {
        "idle": fifty_seeds(19.9, 0.4, {"goal": 997, AREA: 12, OOB: 26, BALL_OUT: 25, OTHER: 1},
                            ph(109386, 18486, 51122, 1006)),
        "rule-based": fifty_seeds(16.9, 0.5, {"goal": 845, OOB: 17, R: 12, AREA: 6, L: 4, BALL_OUT: 1},
                                  ph(82604, 61906, 34594, 896)),
        "q-table": fifty_seeds(17.1, 0.5, {"goal": 857, OOB: 12, R: 4, AREA: 7, L: 7, BALL_OUT: 1},
                               ph(85323, 59495, 34274, 908))}},
]

# "out-of-bounds" is the sum of the finer out reasons, so it is not plotted on its own.
FAIL_TYPES = {
    "stuck in goal area (mostly saved)": [AREA],
    "missed wide": [L, R],
    "out with ball (lost control)": [BALL_OUT],
    "other out": [OTHER],
}
FAIL_COLORS = ["#e34948", "#6250d6", "#eda100", "#a3a29c"]

# SHOOTING is one step per shot, too small to see, so it is left out.
PHASES = ["APPROACH", "CARRY", "TURNING"]
PHASE_COLORS = ["#e87ba4", "#008300", "#6250d6"]


def rows_for(goalie, need):
    """(step number, step, result) for every step that has `need` for this goalie."""
    return [(i, s, s["res"][goalie]) for i, s in enumerate(STEPS, 1)
            if goalie in s["res"] and s["res"][goalie].get(need) is not None]


def color_reverted_ticks(ax, steps):
    for tick, s in zip(ax.get_xticklabels(), steps):
        if not s["kept"]:
            tick.set_color(REVERTED_COLOR)
            tick.set_fontweight("bold")


def plot_progress(out_dir):
    fig, ax = plt.subplots(figsize=(13, 7))
    n = len(STEPS)

    first_50 = next(i for i, s in enumerate(STEPS, 1) if s["seeds"] == 50)
    ax.axvspan(0.5, first_50 - 0.5, color="0.93", zorder=0)
    xt = ax.get_xaxis_transform()
    ax.text((first_50) / 2, 0.98, "5 seeds per goalie", transform=xt,
            ha="center", va="top", color="0.4")
    ax.text((first_50 + n) / 2, 0.98, "50 seeds per goalie", transform=xt,
            ha="center", va="top", color="0.4")

    for i, s in enumerate(STEPS, 1):
        if not s["kept"]:
            ax.axvspan(i - 0.45, i + 0.45, color=REVERTED_COLOR, alpha=0.1, zorder=0)

    finals = []
    for g in GOALIES:
        rows = rows_for(g, "mean")
        xs = [i for i, _, _ in rows]
        ys = [r["mean"] for _, _, r in rows]
        es = [r["sem"] for _, _, r in rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=6, capsize=3, lw=2,
                    color=GOALIE_COLORS[g], label=g)
        finals.append((ys[-1], xs[-1], g))

    # final values, spread apart so they don't overlap
    last_y = None
    for y, x, g in sorted(finals, reverse=True):
        label_y = y if last_y is None else min(y, last_y - 0.9)
        ax.text(x + 0.2, label_y, f"{y:.1f}", va="center",
                color=GOALIE_COLORS[g], fontweight="bold")
        last_y = label_y

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([f"{i}  {s['label']}" for i, s in enumerate(STEPS, 1)],
                       rotation=35, ha="right", rotation_mode="anchor")
    color_reverted_ticks(ax, STEPS)
    ax.set_xlim(0.5, n + 0.9)
    ax.set_ylim(0, 22)
    ax.set_ylabel("goals per game (90 s), mean ± standard error")
    ax.set_title("Striker progress, step by step")
    ax.grid(axis="y", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=REVERTED_COLOR, alpha=0.2))
    labels.append("tried and reverted")
    ax.legend(handles, labels, title="goalie", loc="lower left", bbox_to_anchor=(0.6, 0.03))

    fig.tight_layout()
    fig.savefig(out_dir / "1_progress.png", dpi=200)


def plot_failures(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, g in zip(axes, GOALIES):
        rows = rows_for(g, "reasons")
        pos = range(len(rows))
        bottom = [0.0] * len(rows)
        for (name, keys), color in zip(FAIL_TYPES.items(), FAIL_COLORS):
            vals = [sum(r["reasons"].get(k, 0) for k in keys) / s["seeds"] for _, s, r in rows]
            ax.bar(pos, vals, bottom=bottom, color=color, label=name, edgecolor="white", linewidth=1)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(pos)
        ax.set_xticklabels([str(i) for i, _, _ in rows])
        color_reverted_ticks(ax, [s for _, s, _ in rows])
        ax.set_title(f"vs {g}")
        ax.set_xlabel("test step (red = reverted)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("failed attempts per game")
    axes[0].legend(loc="upper left", fontsize=10)
    fig.suptitle("What went wrong (50-seed runs)")
    fig.tight_layout()
    fig.savefig(out_dir / "2_failures.png", dpi=200)


def plot_phases(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, g in zip(axes, GOALIES):
        rows = rows_for(g, "phases")
        pos = range(len(rows))
        bottom = [0.0] * len(rows)
        for phase, color in zip(PHASES, PHASE_COLORS):
            vals = [r["phases"][phase] / r["phase_goals"] / HZ for _, _, r in rows]
            ax.bar(pos, vals, bottom=bottom, color=color, label=phase.lower(), edgecolor="white", linewidth=1)
            bottom = [b + v for b, v in zip(bottom, vals)]
        for p, total in zip(pos, bottom):
            ax.text(p, total + 0.05, f"{total:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(i) for i, _, _ in rows])
        color_reverted_ticks(ax, [s for _, s, _ in rows])
        ax.set_title(f"vs {g}")
        ax.set_xlabel("test step (red = reverted)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("seconds per goal")
    axes[1].legend(loc="upper right", fontsize=10)
    fig.suptitle("Where the time goes (time per goal, split by phase)")
    fig.tight_layout()
    fig.savefig(out_dir / "3_phases.png", dpi=200)


def main():
    out_dir = Path(__file__).resolve().with_name("plots")
    out_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

    plot_progress(out_dir)
    plot_failures(out_dir)
    plot_phases(out_dir)

    print(f"Saved 3 plots to {out_dir}")
    plt.show()


if __name__ == "__main__":
    main()
