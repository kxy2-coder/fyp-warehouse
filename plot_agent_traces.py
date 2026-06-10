# =============================================================================
# plot_agent_traces.py — Per-Agent Performance Traces
# =============================================================================
# Reads agent_traces.csv and produces a 2x2 figure saved as agent_traces.png:
#   Panel 1: Picks per 10-min bin      — throughput over the shift
#   Panel 2: Distance per 10-min bin   — metres walked per window
#   Panel 3: Avg walking speed per bin — m/s while the agent is actively walking
#                                        reference lines: free speed (1.2 m/s)
#                                        and congestion threshold (0.24 m/s).
#                                        line gaps = agent not walking (idle/picking)
#   Panel 4: Idle ticks per 10-min bin — time spent waiting for jobs
#                                        high = agents starved of work
#                                        low  = agents always busy
#
# Together these separate two failure modes:
#   speed near threshold + low idle  → layout too cramped, agents blocked
#   speed near free      + high idle → layout has dead zones / too few items
#
# Usage:
#   python plot_agent_traces.py                        # reads agent_traces.csv
#   python plot_agent_traces.py --file my_traces.csv   # custom filename
# =============================================================================

import argparse
import csv
import numpy as np
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
                "tick":       int(row["tick"]),
                "agent_id":   int(row["agent_id"]),
                "picks":      float(row["picks"]),
                "distance":   float(row["distance"]),
                "idle_ticks": float(row["idle_ticks"]),
                # slow_ticks replaces blocked_events from Sim_V2 — fall back
                # gracefully so old CSVs still load without crashing.
                "slow_ticks": float(row.get("slow_ticks", row.get("blocked_events", 0))),
                "walk_ticks": float(row.get("walk_ticks", 0)),
            })

    # Group by agent_id, sort by tick
    agents = defaultdict(list)
    for row in rows:
        agents[row["agent_id"]].append(row)
    for aid in agents:
        agents[aid].sort(key=lambda r: r["tick"])

    return agents


def _load_speed_trace(speed_file):
    """
    Load speed_trace.csv into a dict: agent_id (1-based) → np.array of speed.
    Empty cells (agent not walking that tick) become np.nan.
    Also returns the ticks-in-hours array.
    """
    ticks, speed_dict = [], {}
    with open(speed_file, newline="") as f:
        reader = csv.DictReader(f)
        agent_cols = [k for k in reader.fieldnames if k.startswith("agent_")]
        for k in agent_cols:
            speed_dict[int(k.split("_")[1])] = []
        for row in reader:
            ticks.append(int(row["tick"]) / 3600.0)
            for k in agent_cols:
                val = row[k]
                speed_dict[int(k.split("_")[1])].append(
                    float(val) if val != "" else np.nan
                )
    return np.array(ticks), {aid: np.array(v) for aid, v in speed_dict.items()}


def plot_from_traces(filename="agent_traces.csv", out_file="agent_traces.png"):
    """
    Build and save the per-agent trace plot from a CSV file.
    Produces a 2x2 figure: picks, distance, idle time, congestion (all 10-min bins).
    Walking speed is plotted separately by plot_speed_trace().
    """
    agents = load_traces(filename)
    agent_ids = sorted(agents.keys())
    n_agents  = len(agent_ids)

    ticks_hrs = [r["tick"] / 3600 for r in agents[agent_ids[0]]]

    def per_bin(records, field):
        vals = [r[field] for r in records]
        return [vals[i] - vals[i-1] if i > 0 else vals[0]
                for i in range(len(vals))]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Per-Agent Performance Traces: {n_agents} agents (per 10-min bins)",
        fontsize=12, fontweight="bold",
    )

    # Collect per-agent series so we can compute proper y-limits afterwards
    all_picks, all_dist, all_idle, all_cong = [], [], [], []

    for idx, aid in enumerate(agent_ids):
        records   = agents[aid]
        col       = AGENT_COLOURS[idx % len(AGENT_COLOURS)]
        label     = f"Agent {aid}"
        picks_bin = per_bin(records, "picks")
        dist_bin  = per_bin(records, "distance")
        idle_bin  = per_bin(records, "idle_ticks")
        cong_bin  = per_bin(records, "slow_ticks")
        all_picks.extend(picks_bin)
        all_dist.extend(dist_bin)
        all_idle.extend(idle_bin)
        all_cong.extend(cong_bin)
        ax1.plot(ticks_hrs, picks_bin, color=col, linewidth=1.2, label=label, alpha=0.85)
        ax2.plot(ticks_hrs, dist_bin,  color=col, linewidth=1.2, label=label, alpha=0.85)
        ax3.plot(ticks_hrs, idle_bin,  color=col, linewidth=1.2, label=label, alpha=0.85)
        ax4.plot(ticks_hrs, cong_bin,  color=col, linewidth=1.2, label=label, alpha=0.85)

    def _padded_ylim(vals, pad_frac=0.08, force_zero_visible=False):
        """Return (lo, hi) with padding above/below the data range.
        If force_zero_visible, ensure 0 sits a little above the bottom so
        a flat-at-zero line is still visible above the x-axis."""
        if not vals:
            return 0, 1
        lo, hi = min(vals), max(vals)
        if hi == lo:
            hi = lo + 1
        span = hi - lo
        pad  = span * pad_frac
        if force_zero_visible:
            return -pad, hi + pad
        return max(0, lo - pad), hi + pad

    panels = [
        (ax1, "Picks",          "Picks",         _padded_ylim(all_picks)),
        (ax2, "Distance (metres)", "Metres walked", _padded_ylim(all_dist)),
        (ax3, "Idle time",       "Seconds", _padded_ylim(all_idle, force_zero_visible=True)),
        (ax4, "Congestion time", "Seconds", _padded_ylim(all_cong, force_zero_visible=True)),
    ]
    for ax, title, ylabel, (ylo, yhi) in panels:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Shift time (hours)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(9))
        ax.set_xlim(0, max(ticks_hrs) + 0.1)
        ax.set_ylim(ylo, yhi)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"  Agent traces plot saved → {out_file}")
    plt.close(fig)


def plot_speed_trace(filename="speed_trace.csv", n_agents=6,
                     out_file="speed_trace.png", roll_window=120,
                     single_agent=None):
    """
    Plot per-tick walking speed for each agent over the full 8-hour shift.

    Raw per-second data is plotted as a faint shaded band; a rolling mean
    (default 2-minute window) is overlaid as a solid line so the trend is
    readable without losing the moment-to-moment variation.

    Gaps in the line = agent not walking that tick (idle / picking).

    Reference lines:
        green dashed  — free speed (1.2 m/s)
        red dotted    — congestion threshold (0.24 m/s)
    """
    FREE_SPEED  = 1.2
    CONG_THRESH = FREE_SPEED * 0.5   # 0.6 m/s — 50% of free speed

    # Load CSV: columns = tick, agent_1, agent_2, ...
    ticks, speeds = [], {i: [] for i in range(1, n_agents + 1)}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["tick"])
            ticks.append(t / 3600.0)   # convert to hours
            for i in range(1, n_agents + 1):
                val = row.get(f"agent_{i}", "")
                speeds[i].append(float(val) if val != "" else np.nan)

    ticks_arr = np.array(ticks)

    fig, ax = plt.subplots(figsize=(14, 5))
    agent_label = f"Agent {single_agent}" if single_agent else "All Agents"
    fig.suptitle(
        f"Walking Speed — {agent_label} — per-second resolution",
        fontsize=12, fontweight="bold",
    )

    agents_to_plot = [single_agent] if single_agent else range(1, n_agents + 1)

    for idx in agents_to_plot:
        col   = AGENT_COLOURS[(idx - 1) % len(AGENT_COLOURS)]
        raw   = np.array(speeds[idx])

        ax.plot(ticks_arr, raw, color=col, linewidth=0.6, alpha=0.85,
                label=f"Agent {idx}")

    ax.axhline(FREE_SPEED,  color="green", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Free speed ({FREE_SPEED} m/s)")
    ax.axhline(CONG_THRESH, color="red",   linestyle=":",  linewidth=1.2,
               alpha=0.7, label=f"Congestion threshold — 50% free speed ({CONG_THRESH:.2f} m/s)")

    ax.set_xlabel("Shift time (hours)", fontsize=10)
    ax.set_ylabel("Walking speed (m/s)", fontsize=10)
    ax.set_xticks(range(9))
    ax.set_xlim(0, ticks_arr[-1] + 0.02)
    ax.set_ylim(0, FREE_SPEED * 1.15)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"  Per-tick speed trace plot saved → {out_file}")
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
    parser.add_argument("--speed-trace", default=None, metavar="SPEED_CSV",
                        help="Plot per-tick speed trace from speed_trace.csv "
                             "(e.g. --speed-trace speed_trace.csv)")
    parser.add_argument("--speed-out",   default="speed_trace.png",
                        help="Output PNG for speed trace (default: speed_trace.png)")
    parser.add_argument("--agents", type=int, default=6,
                        help="Number of agents in speed_trace.csv (default: 6)")
    parser.add_argument("--agent", type=int, default=None,
                        help="Plot a single agent only (e.g. --agent 1)")
    args = parser.parse_args()

    if args.speed_trace:
        plot_speed_trace(filename=args.speed_trace,
                         n_agents=args.agents,
                         out_file=args.speed_out,
                         single_agent=args.agent)
    else:
        plot_from_traces(filename=args.file, out_file=args.out)


if __name__ == "__main__":
    main()
