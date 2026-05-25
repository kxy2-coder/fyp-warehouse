# =============================================================================
# plot_workload.py — NHPP Workload Visualisation
# =============================================================================
# Two ways to use this file:
#
#   A) From main.py (--plot-workload flag):
#      main.py collects arrivals/completions during its own simulation runs
#      and calls plot_from_data() directly — no second simulation needed.
#
#   B) Standalone:
#      python plot_workload.py              # single run, 10-min bins
#      python plot_workload.py --runs 5     # average over 5 runs (smoother)
#      python plot_workload.py --bin 300    # 5-min bins (more detail)
#
# Output: workload_analysis.png (always saved; plt.show() skipped on WSL/headless)
# =============================================================================

import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")   # file-only backend — works on WSL/headless; open the PNG to view
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from agent   import (nhpp_arrival, get_multiplier, SHIFT_TICKS,
                     LAMBDA_BASE, DEMAND_BETA)


# =============================================================================
# Helpers (used by both paths)
# =============================================================================

def bin_data(data, bin_size):
    """Sum tick-level data into fixed-size bins."""
    n_bins = SHIFT_TICKS // bin_size
    return data[:n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)


def theoretical_arrivals_per_bin(bin_size):
    """
    Expected arrivals per bin from the NHPP formula:
        E[arrivals in bin b] = sum_{t in bin} LAMBDA_BASE * m(t)
    """
    n_bins = SHIFT_TICKS // bin_size
    rates  = np.zeros(n_bins)
    for b in range(n_bins):
        t0 = b * bin_size
        t1 = t0 + bin_size
        rates[b] = sum(LAMBDA_BASE * get_multiplier(t) for t in range(t0, t1))
    return rates


# =============================================================================
# Core plotting function — accepts pre-collected data from any source
# =============================================================================

def save_workload_csv(all_arrivals, all_completions,
                      out_file="workload_data.csv"):
    """
    Save averaged tick-level workload data to CSV.
    Columns: tick, arrivals, completions (avg across runs).
    Re-plot later via: plot_from_csv("workload_data.csv")
    """
    import csv
    avg_arrivals    = np.mean(all_arrivals,    axis=0)
    avg_completions = np.mean(all_completions, axis=0)
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "arrivals", "completions"])
        for t in range(len(avg_arrivals)):
            w.writerow([t, avg_arrivals[t], avg_completions[t]])
    print(f"  Workload data saved → {out_file}")


def plot_from_csv(csv_file="workload_data.csv", bin_size=600,
                  out_file="workload_analysis.png", n_agents=4):
    """
    Re-plot workload analysis from a saved CSV (no simulation needed).
    Useful when you want to tweak bin size, colours, labels without
    re-running the full simulation.
    """
    import csv
    arrivals    = []
    completions = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arrivals.append(float(row["arrivals"]))
            completions.append(float(row["completions"]))
    arr  = np.array(arrivals,    dtype=np.float64)
    comp = np.array(completions, dtype=np.float64)
    # Wrap as single-run lists so plot_from_data can consume them unchanged
    plot_from_data([arr], [comp], bin_size=bin_size,
                   out_file=out_file, n_agents=n_agents)


def plot_from_data(all_arrivals, all_completions, bin_size=600,
                   out_file="workload_analysis.png", n_agents=4):
    """
    Build and save the workload analysis plot from already-collected tick data.

    Parameters
    ----------
    all_arrivals    : list of int32 arrays (one per run), length SHIFT_TICKS each
    all_completions : list of int32 arrays (one per run), length SHIFT_TICKS each
    bin_size        : ticks per bar (default 600 = 10 min)
    out_file        : output PNG filename
    n_agents        : number of agents (used in title only)
    """
    runs = len(all_arrivals)

    # Average tick-level data across runs before binning
    avg_arrivals    = np.mean(all_arrivals,    axis=0)
    avg_completions = np.mean(all_completions, axis=0)

    # Bin into time windows
    arr_binned  = bin_data(avg_arrivals,    bin_size)
    comp_binned = bin_data(avg_completions, bin_size)
    theory      = theoretical_arrivals_per_bin(bin_size)

    n_bins  = len(arr_binned)
    bin_hrs = bin_size / 3600
    x_left  = np.arange(n_bins) * bin_hrs

    # Cumulative (tick resolution, x in hours)
    cum_arrived   = np.cumsum(avg_arrivals)
    cum_completed = np.cumsum(avg_completions)
    backlog       = cum_arrived - cum_completed
    tick_hours    = np.arange(SHIFT_TICKS) / 3600

    # Summary stats
    total_arrived   = int(round(avg_arrivals.sum()))
    total_completed = int(round(avg_completions.sum()))
    completion_pct  = 100.0 * total_completed / max(total_arrived, 1)

    print(f"\n  Workload summary (avg over {runs} run(s)):")
    print(f"    Jobs arrived    : {total_arrived}")
    print(f"    Jobs completed  : {total_completed}")
    print(f"    Queue remaining : {total_arrived - total_completed}")
    print(f"    Completion rate : {completion_pct:.1f}%")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    run_label = f"avg over {runs} runs" if runs > 1 else "1 run"
    fig.suptitle(
        f"NHPP Workload Analysis  —  {n_agents} agents, 8-hour shift  "
        f"({run_label})\n"
        f"λ_base = {LAMBDA_BASE},  β = {DEMAND_BETA},  "
        f"bin = {bin_size // 60} min",
        fontsize=12, fontweight="bold",
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, :])   # top row — full width
    ax2 = fig.add_subplot(gs[1, 0])   # bottom left
    ax3 = fig.add_subplot(gs[1, 1])   # bottom right

    def shade_periods(ax):
        ax.axvspan(0, 2, alpha=0.07, color="#D94040", zorder=0)
        ax.axvspan(2, 6, alpha=0.07, color="#4075D9", zorder=0)
        ax.axvspan(6, 8, alpha=0.07, color="#D94040", zorder=0)
        for xv in (2, 6):
            ax.axvline(xv, color="gray", linewidth=0.8, linestyle=":", zorder=1)

    # Panel 1: arrival rate vs completion rate
    bar_w = bin_hrs * 0.42
    ax1.bar(x_left,         arr_binned,  width=bar_w, align="edge",
            color="#4A90D9", alpha=0.80, label="Actual arrivals")
    ax1.bar(x_left + bar_w, comp_binned, width=bar_w, align="edge",
            color="#5CB85C", alpha=0.80, label="Completions by agents")
    ax1.plot(x_left + bin_hrs / 2, theory,
             color="#E8A838", linewidth=2, linestyle="--",
             marker="o", markersize=3, label="Theoretical λ(t)·Δt")
    shade_periods(ax1)
    ax1.set_xlabel("Shift time (hours)")
    ax1.set_ylabel(f"Jobs per {bin_size // 60}-min bin")
    ax1.set_title("Arrival Rate vs Completion Rate per Time Window")
    ax1.set_xticks(range(9))
    ax1.set_xlim(0, 8)
    ax1.legend(fontsize=9, ncol=3, loc="upper center")
    ax1.grid(axis="y", alpha=0.3)
    ylim_top = ax1.get_ylim()[1]
    for xc, label in [(1, "Morning\npeak"), (4, "Mid-shift\nlull"), (7, "End-of-shift\nrush")]:
        ax1.text(xc, ylim_top * 0.88, label, ha="center", va="top",
                 fontsize=8, color="gray", style="italic")

    # Panel 2: cumulative arrivals vs completions
    ax2.plot(tick_hours, cum_arrived,   color="#4A90D9", linewidth=1.5,
             label="Cumulative arrived")
    ax2.plot(tick_hours, cum_completed, color="#5CB85C", linewidth=1.5,
             label="Cumulative completed")
    ax2.fill_between(tick_hours, cum_completed, cum_arrived,
                     alpha=0.25, color="#E8A838", label="Backlog (queue)")
    shade_periods(ax2)
    ax2.set_xlabel("Shift time (hours)")
    ax2.set_ylabel("Cumulative jobs")
    ax2.set_title("Cumulative Arrivals vs Completions")
    ax2.set_xticks(range(9))
    ax2.set_xlim(0, 8)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: queue backlog
    ax3.plot(tick_hours, backlog, color="#E85858", linewidth=1.5, label="Queue backlog")
    ax3.fill_between(tick_hours, 0, backlog, alpha=0.25, color="#E85858")
    shade_periods(ax3)
    ax3.set_xlabel("Shift time (hours)")
    ax3.set_ylabel("Jobs waiting in queue")
    ax3.set_title("Queue Backlog  (arrived − completed)")
    ax3.set_xticks(range(9))
    ax3.set_xlim(0, 8)
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    summary = (
        f"Total arrived: {total_arrived}   "
        f"Total completed: {total_completed}   "
        f"Completion rate: {completion_pct:.1f}%   "
        f"Final backlog: {int(round(backlog[-1]))}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9,
             color="dimgray", style="italic")

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_file}")
    plt.close(fig)


# =============================================================================
# Standalone simulation runner (used only when running this file directly)
# =============================================================================

def _run_tracked_simulation(seed=None):
    """
    Run one full 8-hour simulation internally.
    Only called when plot_workload.py is run standalone (not via main.py).
    """
    from grid    import Grid
    from agent   import Agent
    from metrics import MetricsTracker
    from rl_env  import resolve_right_of_way, GRID_ROWS, GRID_COLS, NUM_AGENTS, REPLENISH_DELAY

    if seed is not None:
        random.seed(seed)

    grid    = Grid(rows=GRID_ROWS, cols=GRID_COLS, replenish_delay=REPLENISH_DELAY)
    agent   = Agent(NUM_AGENTS, grid, quota=0)
    metrics = MetricsTracker()

    arrivals    = np.zeros(SHIFT_TICKS, dtype=np.int32)
    completions = np.zeros(SHIFT_TICKS, dtype=np.int32)
    prev_completed = 0

    for tick in range(SHIFT_TICKS):
        if nhpp_arrival(tick):
            agent.add_job()
            arrivals[tick] = 1
        resolve_right_of_way(agent)
        agent.step()
        grid.tick_replenishment()
        metrics.update(agent, grid.depot)
        current_completed    = int(agent.orders_completed.sum())
        completions[tick]    = current_completed - prev_completed
        prev_completed       = current_completed

    return arrivals, completions


def main():
    parser = argparse.ArgumentParser(description="NHPP workload visualisation (standalone)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Simulation runs to average over (default 1)")
    parser.add_argument("--bin",  type=int, default=600,
                        help="Bin size in ticks/seconds (default 600 = 10 min)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default 42)")
    parser.add_argument("--save-csv", action="store_true",
                        help="Also save tick-level data to workload_data.csv "
                             "(for later re-plotting via plot_from_csv())")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Skip simulation and re-plot from saved CSV instead")
    args = parser.parse_args()

    # Re-plot path: load CSV and skip simulation entirely
    if args.from_csv:
        print(f"\n  Re-plotting from {args.from_csv} (no simulation)")
        plot_from_csv(args.from_csv, bin_size=args.bin)
        return

    print(f"\nRunning {args.runs} simulation(s) × {SHIFT_TICKS} ticks each...")

    all_arrivals    = []
    all_completions = []
    for i in range(args.runs):
        seed = args.seed + i * 7919
        print(f"  Run {i + 1}/{args.runs}  (seed={seed})")
        arr, comp = _run_tracked_simulation(seed=seed)
        all_arrivals.append(arr)
        all_completions.append(comp)

    if args.save_csv:
        save_workload_csv(all_arrivals, all_completions)

    plot_from_data(all_arrivals, all_completions, bin_size=args.bin)


if __name__ == "__main__":
    main()
