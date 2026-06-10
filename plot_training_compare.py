# =============================================================================
# plot_training_compare.py — Compare reward curves across operating conditions
# =============================================================================
# Reads multiple training_log.csv files from Final_Training_Results/Agents=N,Lambda=X/
# and plots the rolling-average reward for each lambda on a single figure,
# letting you see how training quality varies with arrival rate at a fixed
# agent count.
#
# Usage:
#   python plot_training_compare.py --agents 4
#   python plot_training_compare.py --agents 6 --window 30
#   python plot_training_compare.py --agents 8 --root Final_Training_Results
# =============================================================================

import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})

# Lambda values to plot for each agent count (matches Final_Training_Results folder names)
LAMBDA_SETS = {
    4: [0.02, 0.04, 0.06, 0.08, 0.10],
    6: [0.04, 0.06, 0.08, 0.10, 0.12],
    8: [0.06, 0.08, 0.10, 0.12, 0.14],
}

# Fixed lambda → colour mapping so the same arrival rate is the same colour
# across all three figures (4, 6, 8 agents). 7 unique lambdas total.
LAMBDA_COLOURS = {
    0.02: "#534AB7",   # purple
    0.04: "#1D9E75",   # green
    0.06: "#D85A30",   # orange
    0.08: "#D4537E",   # pink
    0.10: "#E8A020",   # yellow
    0.12: "#378ADD",   # blue
    0.14: "#8B6FBA",   # lavender
}


def _candidate_folders(root, agents, lam):
    """
    Return possible folder paths for an (agents, lambda) combo. The
    Final_Training_Results folder uses inconsistent capitalisation ("Lambda" vs "lambda")
    and lambda values like 0.1 (no trailing zero), so we try several variants.
    """
    lam_strs = [f"{lam:g}", f"{lam:.2f}"]              # 0.1 and 0.10
    case_variants = ["Lambda", "lambda"]
    folders = []
    for lstr in lam_strs:
        for cvar in case_variants:
            folders.append(os.path.join(root, f"Agents={agents},{cvar}={lstr}"))
    return folders


def _load_log(root, agents, lam):
    """Find and load training_log.csv for the given (agents, lambda)."""
    for folder in _candidate_folders(root, agents, lam):
        path = os.path.join(folder, "training_log.csv")
        if os.path.exists(path):
            return pd.read_csv(path), folder
    return None, None


def plot_agents(agents, root="Final_Training_Results", window=20,
                out_file=None, lambdas=None):
    """
    Plot rolling-average reward for every lambda value at a given agent count.
    """
    if lambdas is None:
        lambdas = LAMBDA_SETS.get(agents)
        if lambdas is None:
            print(f"ERROR: no default lambda set for {agents} agents. "
                  f"Pass --lambdas explicitly.")
            sys.exit(1)

    if out_file is None:
        out_file = f"reward_compare_agents{agents}.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Reward over Training — {agents} Agents",
                 fontsize=13, fontweight="bold")

    n_loaded = 0
    for i, lam in enumerate(lambdas):
        df, folder = _load_log(root, agents, lam)
        if df is None:
            print(f"  Skipped λ={lam} — training_log.csv not found")
            continue

        rolling   = df["reward"].rolling(window=window, min_periods=1).mean()
        # Look up the colour for this lambda. round() handles floating-point
        # representation issues (e.g. 0.1 stored as 0.10000000000000001).
        colour    = LAMBDA_COLOURS.get(round(lam, 2), "#888888")
        # Start = first reliable rolling value (after the window fills), or
        # the first value if the run is shorter than the window.
        start_idx = min(window, len(rolling) - 1)
        start_val = rolling.iloc[start_idx]
        end_val   = rolling.iloc[-1]
        ax.plot(df["episode"], rolling,
                color=colour, linewidth=1.8,
                label=f"λ = {lam:g}  "
                      f"(start: {start_val:.3f} → end: {end_val:.3f})")
        print(f"  Loaded {len(df):>4} episodes for λ={lam:g}  → {folder}")
        n_loaded += 1

    if n_loaded == 0:
        print("ERROR: no training logs loaded — check --root path")
        sys.exit(1)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (rolling mean, window = {})".format(window))
    ax.margins(y=0.05)
    ax.legend(loc="lower right", fontsize=10, title="Arrival rate")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"\n  Comparison plot saved → {out_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare training reward curves across lambda values "
                    "for a fixed agent count")
    parser.add_argument("--agents", type=int, required=True,
                        choices=[4, 6, 8],
                        help="Number of agents (4, 6 or 8)")
    parser.add_argument("--root",   type=str, default="Final_Training_Results",
                        help="Root folder containing Agents=N,Lambda=X subdirs "
                             "(default: Final_Training_Results)")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling-average window size (default: 20)")
    parser.add_argument("--out",    type=str, default=None,
                        help="Output PNG filename (default: "
                             "reward_compare_agents<N>.png)")
    args = parser.parse_args()

    plot_agents(agents=args.agents, root=args.root,
                window=args.window, out_file=args.out)


if __name__ == "__main__":
    main()
