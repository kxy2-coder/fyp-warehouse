# =============================================================================
# plot_agent_traces.py — Per-Agent Performance Traces
# =============================================================================
# Reads agent_traces.csv and produces a 2x2 figure saved as agent_traces.png:
#   Panel 1: Picks completed (cumulative) per agent over shift time
#   Panel 2: Distance travelled (cumulative) per agent over shift time
#   Panel 3: Idle ticks per 10-min bin per agent (non-cumulative)
#   Panel 4: Blocked events (cumulative) per agent over shift time
#
# Usage:
#   python plot_agent_traces.py                        # reads agent_traces.csv
#   python plot_agent_traces.py --file my_traces.csv   # custom filename
# =============================================================================

import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict


AGENT_COLOURS = [
    "#4A90D9",   # blue
    "#5CB85C",   # green
    "#E8A838",   # amber
    "#E85858",   # red
    "#9B59B6",   # purple
    "#1ABC9C",   # teal
    "#E67E22",   # orange
    "#34495E",   # dark grey
]


def load_traces(filename):
    """
    Read agent_traces.csv and return a dict:
        agent_id → { field → list of values ordered by tick }
    Also returns the list of ticks (x-axis, in hours).
    """
    rows = []
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "tick":           int(row["tick"]),
                "agent_id":       int(row["agent_id"]),
                "picks":          float(row["picks"]),
                "distance":       float(row["distance"]),
                "idle_ticks":     float(row["idle_ticks"]),
                "blocked_events": float(row["blocked_events"]),
                "fatigue":        float(row["fatigue"]),
            })

    # Group by agent_id, sort by tick
    agents = defaultdict(list)
    for row in rows:
        agents[row["agent_id"]].append(row)
    for aid in agents:
        agents[aid].sort(key=lambda r: r["tick"])

    return agents


def plot_from_traces(filename="agent_traces.csv", out_file="agent_traces.png"):
    """
    Build and save the per-agent trace plot from a CSV file.
    Called directly by main.py when --plot-traces is set.
    """
    agents = load_traces(filename)
    agent_ids = sorted(agents.keys())
    n_agents  = len(agent_ids)

    # Convert ticks to hours for all agents (same ticks for all)
    ticks_hrs = [r["tick"] / 3600 for r in agents[agent_ids[0]]]

    # Convert cumulative series to per-bin rates
    def per_bin(records, field):
        vals = [r[field] for r in records]
        return [vals[i] - vals[i-1] if i > 0 else vals[0] - 0
                for i in range(len(vals))]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Per-Agent Performance Traces  —  {n_agents} agents, 8-hour shift\n"
        f"(averaged across runs, 10-min bins)",
        fontsize=12, fontweight="bold",
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for idx, aid in enumerate(agent_ids):
        records  = agents[aid]
        col      = AGENT_COLOURS[idx % len(AGENT_COLOURS)]
        label    = f"Agent {aid}"

        picks_bin    = per_bin(records, "picks")
        dist_bin     = per_bin(records, "distance")
        idle_bin     = per_bin(records, "idle_ticks")
        blocked_bin  = per_bin(records, "blocked_events")

        ax1.plot(ticks_hrs, picks_bin,   color=col, linewidth=1.2, label=label, alpha=0.85)
        ax2.plot(ticks_hrs, dist_bin,    color=col, linewidth=1.2, label=label, alpha=0.85)
        ax3.plot(ticks_hrs, idle_bin,    color=col, linewidth=1.2, label=label, alpha=0.85)
        ax4.plot(ticks_hrs, blocked_bin, color=col, linewidth=1.2, label=label, alpha=0.85)

    for ax, title, ylabel, zero_based in [
        (ax1, "Picks per 10-min Bin",          "Picks",       False),
        (ax2, "Distance per 10-min Bin",        "Cells",       False),
        (ax3, "Idle Ticks per 10-min Bin",      "Ticks idle",  True),
        (ax4, "Blocked Events per 10-min Bin",  "Collisions",  True),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Shift time (hours)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(9))
        ax.set_xlim(0, max(ticks_hrs) + 0.1)
        if zero_based:
            ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"  Agent traces plot saved → {out_file}")
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Per-agent trace visualisation")
    parser.add_argument("--file", default="agent_traces.csv",
                        help="CSV file to read (default: agent_traces.csv)")
    parser.add_argument("--out",  default="agent_traces.png",
                        help="Output PNG filename (default: agent_traces.png)")
    args = parser.parse_args()
    plot_from_traces(filename=args.file, out_file=args.out)


if __name__ == "__main__":
    main()
