# =============================================================================
# slidesplot.py — Picks/hr vs λ for different agent counts (slide figure)
# =============================================================================
# Runs the simulator at a fixed layout, sweeping (N_agents × λ_base) and
# plotting one line per N. Marks the plateau point on each line in red.
#
# Configure the layout, agent counts, and λ values in the CONFIG block below.
#
# USAGE:
#   python slidesplot.py
#   python slidesplot.py --out my_plot.png --eval-runs 5
# =============================================================================

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import agent as _agent_module
import rl_env
from rl_env import run_single_simulation, compute_depot_cols

# =============================================================================
# CONFIG — edit these to choose the layout + sweep ranges
# =============================================================================

# Layout to evaluate (every (N, λ) point uses THIS layout).
LAYOUT = {
    "aisle_width":         2,
    "centre_aisle_width":  3,
    "depot_count":         1,                                # n_d
    "shelf_start_row":     2,
    "shelf_end_row":       23,
    "cross_aisle_row":     12,
    "cross_aisle_on":      True,                             # b_ca
}

# Agent team sizes to compare (one line per entry).
AGENT_COUNTS = [4, 6, 8]

# λ values to sweep along the x-axis.
LAMBDAS      = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

# Plateau detection: a point counts as plateaued if its picks/hr is within
# this fraction of the line's peak (3% = 0.03).
PLATEAU_TOLERANCE = 0.03

AGENT_COLOURS = {
    4: "#4A90D9",   # blue
    6: "#5CB85C",   # green
    8: "#E8A838",   # amber
}


# =============================================================================
# HELPERS
# =============================================================================

def evaluate(n_agents, lam, eval_runs):
    """Run the simulator at the configured layout under (n_agents, lam).
    Returns (picks_per_hour, completion_rate) or (None, None) if invalid."""
    rl_env.NUM_AGENTS         = n_agents
    rl_env.LAMBDA_BASE        = lam
    _agent_module.LAMBDA_BASE = lam

    depot_cols = compute_depot_cols(LAYOUT["depot_count"])
    result = run_single_simulation(
        aisle_width=LAYOUT["aisle_width"],
        centre_aisle_width=LAYOUT["centre_aisle_width"],
        depot_col=depot_cols[0],
        shelf_start_row=LAYOUT["shelf_start_row"],
        shelf_end_row=LAYOUT["shelf_end_row"],
        cross_aisle_row=LAYOUT["cross_aisle_row"],
        eval_runs=eval_runs,
        depot_cols=depot_cols,
        depot_count=LAYOUT["depot_count"],
        depot_row=0,
        cross_aisle_enabled=LAYOUT["cross_aisle_on"],
    )
    if result is None:
        return None, None
    picks    = result["picks_per_hour"]
    arrived  = result.get("jobs_arrived", 0)
    done     = result.get("jobs_completed", 0)
    comp     = (done / arrived) if arrived > 0 else None
    return picks, comp


def find_plateau_index(picks, tol=PLATEAU_TOLERANCE):
    """First index where picks is within tol × max(picks) of the curve peak."""
    if len(picks) < 2:
        return len(picks) - 1
    peak = max(picks)
    threshold = tol * peak
    for i, p in enumerate(picks):
        if peak - p <= threshold:
            return i
    return len(picks) - 1


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Picks/hr vs λ plot — sweep N × λ at a fixed layout.")
    parser.add_argument("--eval-runs", type=int, default=3,
                        help="Simulation runs per (N, λ) point (default 3)")
    parser.add_argument("--out", default="picks_vs_lambda.png",
                        help="Output PNG filename")
    args = parser.parse_args()

    print("=" * 60)
    print("  PICKS/HR vs λ  —  fixed layout")
    print("=" * 60)
    print("  Layout:")
    for k, v in LAYOUT.items():
        print(f"    {k:<22} = {v}")
    print(f"  Agent counts : {AGENT_COUNTS}")
    print(f"  λ values     : {LAMBDAS}")
    print(f"  Eval runs    : {args.eval_runs}")
    print("=" * 60)

    # Collect results: {n: {"picks": [...], "comp": [...]}}
    results = {n: {"picks": [], "comp": []} for n in AGENT_COUNTS}

    for n in AGENT_COUNTS:
        print(f"\n  Sweeping {n} agents...")
        for lam in LAMBDAS:
            picks, comp = evaluate(n, lam, args.eval_runs)
            if picks is None:
                print(f"    λ={lam}: invalid — skipped")
                results[n]["picks"].append(np.nan)
                results[n]["comp"].append(np.nan)
            else:
                print(f"    λ={lam}: picks/hr={picks:.1f}  comp={comp:.3f}")
                results[n]["picks"].append(picks)
                results[n]["comp"].append(comp if comp is not None else np.nan)

    def _save(fig, out_path):
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        pdf = out_path.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print(f"  Saved {out_path} and {pdf}")
        plt.close(fig)

    # ── Figure 1: Picks per hour ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    for n in AGENT_COUNTS:
        picks = results[n]["picks"]
        col   = AGENT_COLOURS.get(n, "#888888")
        ax1.plot(LAMBDAS, picks, color=col, linewidth=2.2, marker="o",
                 markersize=7, label=f"{n} agents")

        valid = [(l, p) for l, p in zip(LAMBDAS, picks) if not np.isnan(p)]
        if valid:
            vl, vp = zip(*valid)
            i = find_plateau_index(list(vp))
            ax1.scatter([vl[i]], [vp[i]], color="red", s=70, zorder=5,
                        edgecolor="white", linewidth=1.2)
            ax1.annotate(f"plateau\nλ={vl[i]:.2f}",
                         xy=(vl[i], vp[i]),
                         xytext=(10, -25), textcoords="offset points",
                         fontsize=10, color="red", fontweight="bold",
                         ha="left")

    ax1.set_xlabel(r"Arrival rate $\lambda_0$ (jobs per tick)", fontsize=12)
    ax1.set_ylabel("Picks per hour", fontsize=12)
    ax1.set_title("Picks per hour vs demand intensity",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc="lower right")
    ax1.tick_params(axis="both", labelsize=11)
    print()
    _save(fig1, args.out)

    # ── Figure 2: Completion rate ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    for n in AGENT_COUNTS:
        comp = results[n]["comp"]
        col  = AGENT_COLOURS.get(n, "#888888")
        ax2.plot(LAMBDAS, comp, color=col, linewidth=2.2, marker="o",
                 markersize=7, label=f"{n} agents")

    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_xlabel(r"Arrival rate $\lambda_0$ (jobs per tick)", fontsize=12)
    ax2.set_ylabel("Completion rate (jobs completed / arrived)", fontsize=12)
    ax2.set_title("Completion rate vs demand intensity",
                  fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc="lower left")
    ax2.tick_params(axis="both", labelsize=11)

    comp_out = args.out.rsplit(".", 1)[0] + "_completion.png"
    _save(fig2, comp_out)


if __name__ == "__main__":
    main()
