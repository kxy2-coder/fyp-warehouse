# =============================================================================
# plot_training.py — Visualise RL Training Results
# =============================================================================
# Reads training_log.csv (produced by rl_env.py) and plots:
#   1. Overall reward over episodes
#   2. Picks per hour over episodes
#   3. Average travel distance per agent over episodes
#   4. Congestion rate over episodes
#   5. Layout parameters over episodes
#
# Each plot shows both raw values (light dots) and a rolling average
# (solid line) to make trends visible through the noise.
#
# USAGE:
#   python rl_env.py                    # train first (creates training_log.csv)
#   python plot_training.py             # then plot
#   python plot_training.py --window 50 # change rolling average window size
# =============================================================================
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend — saves to file only
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})


def load_log(filepath="training_log.csv"):
    """Load the training log CSV. Exit with a message if not found."""
    if not os.path.exists(filepath):
        print(f"\n  ERROR: '{filepath}' not found.")
        print(f"  Run 'python rl_env.py' first to generate training data.\n")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} episodes from '{filepath}'")
    return df


def load_eval_log(filepath="evaluations.npz"):
    """
    Load the EvalCallback evaluation log (evaluations.npz).
    Returns (timesteps, mean_rewards) arrays, or (None, None) if not found.
    """
    if not os.path.exists(filepath):
        return None, None
    data = np.load(filepath)
    timesteps    = data["timesteps"]       # training timestep at each eval
    mean_rewards = data["results"].mean(axis=1)   # mean reward per eval

    # Remove invalid layout points (reward = -1.0) caused by policy
    # occasionally pushing parameters into infeasible combinations during eval
    valid_mask   = mean_rewards >= 0.0
    n_removed    = np.sum(~valid_mask)
    if n_removed > 0:
        print(f"  Removed {n_removed} invalid eval points (reward < 0)")
    timesteps    = timesteps[valid_mask]
    mean_rewards = mean_rewards[valid_mask]

    print(f"  Loaded {len(timesteps)} evaluation checkpoints from '{filepath}'")
    return timesteps, mean_rewards


def plot_metric(ax, df, column, ylabel, color, window, higher_is_better=True):
    """
    Plot a single metric: raw values as faint dots, rolling average as line.
    """
    episodes = df["episode"]
    values   = df[column]
    rolling  = values.rolling(window=window, min_periods=1).mean()

    ax.scatter(episodes, values, alpha=0.15, s=8, color=color, label="Per episode")
    ax.plot(episodes, rolling, color=color, linewidth=2,
            label=f"Rolling avg (window={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right" if higher_is_better else "upper right",
              fontsize=9)

    # Show start and end values of the rolling average
    start_val = rolling.iloc[min(window, len(rolling) - 1)]
    end_val   = rolling.iloc[-1]
    ax.set_title(f"{ylabel}\n"
                 f"Start: {start_val:.3f}  →  End: {end_val:.3f}",
                 fontsize=11)


def plot_parameters(ax, df, window):
    """
    Plot layout parameter values over episodes to see what the agent changed.
    """
    params = [
        ("aisle_width",        "Aisle width",         "#534AB7"),
        ("centre_aisle_width", "Centre aisle width",  "#1D9E75"),
        ("depot_count",        "Depot count",         "#D85A30"),
        ("shelf_start_row",    "Shelf start row",     "#378ADD"),
        ("shelf_end_row",      "Shelf end row",       "#D4537E"),
        ("cross_aisle_row",    "Cross-aisle row",     "#E8A020"),
        ("cross_aisle_on",     "Cross-aisle on",      "#8B6FBA"),
    ]

    episodes = df["episode"]

    for col, label, color in params:
        if col in df.columns:
            rolling = df[col].rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, rolling, linewidth=1.5, label=label, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Parameter value")
    ax.set_title("Layout parameters over training (rolling avg)", fontsize=11)
    ax.legend(loc="best", fontsize=8)


def main():
    parser = argparse.ArgumentParser(
        description="Plot RL training results from training_log.csv")
    parser.add_argument("--file", type=str, default="training_log.csv",
                        help="Path to training log CSV (default: training_log.csv)")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling average window size (default: 20)")
    args = parser.parse_args()

    print("=" * 55)
    print("  RL TRAINING VISUALISATION")
    print("=" * 55)

    df = load_log(args.file)

    # ── Create figure with 5 subplots ─────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Warehouse Layout RL Training Results",
                 fontsize=14, fontweight="bold", y=0.98)

    # Plot 1: Overall reward (training only)
    plot_metric(axes[0, 0], df, "reward",
                "Reward (0-1)", "#534AB7", args.window, higher_is_better=True)

    # Plot 2: Picks per hour (raw, not normalised)
    plot_metric(axes[0, 1], df, "picks_per_hour",
                "Picks per hour", "#1D9E75", args.window, higher_is_better=True)

    # Plot 3: Distance per agent (raw, not normalised)
    plot_metric(axes[1, 0], df, "dist_per_agent",
                "Avg distance per agent (metres)", "#D85A30", args.window,
                higher_is_better=False)

    # Plot 4: Congestion rate (raw, not normalised)
    plot_metric(axes[1, 1], df, "congestion_rate",
                "Congestion rate", "#D4537E", args.window,
                higher_is_better=False)

    # Plot 5: Layout parameters
    plot_parameters(axes[2, 0], df, args.window)

    # Plot 6: Normalised KPIs (to see how each contributes to reward)
    episodes = df["episode"]
    w = args.window
    axes[2, 1].plot(episodes,
                     df["norm_picks"].rolling(window=w, min_periods=1).mean(),
                     linewidth=1.5, label="Norm picks/hr", color="#1D9E75")
    axes[2, 1].plot(episodes,
                     df["norm_distance"].rolling(window=w, min_periods=1).mean(),
                     linewidth=1.5, label="Norm distance", color="#D85A30")
    axes[2, 1].plot(episodes,
                     df["norm_congestion"].rolling(window=w, min_periods=1).mean(),
                     linewidth=1.5, label="Norm congestion", color="#D4537E")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_ylabel("Normalised value (0-1)")
    axes[2, 1].set_title("Normalised KPI contributions (rolling avg)", fontsize=11)
    axes[2, 1].legend(loc="best", fontsize=9)
    axes[2, 1].set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # ── Save and show ─────────────────────────────────────────────────────
    output_file = "training_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to '{output_file}'")
    print("  Opening plot window...\n")
    plt.show()


if __name__ == "__main__":
    main()