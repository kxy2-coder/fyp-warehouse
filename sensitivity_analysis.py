# =============================================================================
# sensitivity_analysis.py — One-at-a-time sensitivity analysis for layout KPIs
# =============================================================================
# Sweeps each of the 7 layout decision variables independently across its full
# valid range while keeping all other variables fixed at the baseline layout.
#
# Decision variables (matching rl_env.py action space):
#   w_a   aisle_width           ∈ [2, 3]      ±1
#   w_c   centre_aisle_width    ∈ [1, 5]      ±2
#   n_d   depot_count           ∈ [1, 4]      ±1  (positions equally spaced)
#   r_s   shelf_start_row       ∈ [2, 8]      ±1
#   r_e   shelf_end_row         ∈ [18, 23]    ±1
#   r_ca  cross_aisle_row       ∈ [2, 23]     ±1  (only if b_ca=on)
#   b_ca  cross_aisle_enabled   on / off
#
# OUTPUT:
#   - sensitivity_results.csv   — raw KPI values for every sweep point
#   - sensitivity_analysis.png  — 4x2 KPI response curves (relative to baseline)
#   - sensitivity_summary.png   — ranked bar chart of overall sensitivity
#
# USAGE:
#   python sensitivity_analysis.py
#   python sensitivity_analysis.py --eval-runs 5
#   python sensitivity_analysis.py --plot-only      # re-plot from CSV
# =============================================================================

import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rl_env import (
    run_single_simulation,
    compute_depot_cols,
    GRID_ROWS, GRID_COLS,
    DEFAULT_AISLE_WIDTH, DEFAULT_CENTRE_AISLE,
    DEFAULT_DEPOT_COUNT,
    DEFAULT_SHELF_START, DEFAULT_SHELF_END,
    DEFAULT_CROSS_AISLE_ROW, DEFAULT_CROSS_AISLE_ON,
)

# =============================================================================
# BASELINE — every sweep holds the other 6 parameters at these values
# =============================================================================

DEFAULT = {
    "aisle_width":         DEFAULT_AISLE_WIDTH,        # 2
    "centre_aisle_width":  DEFAULT_CENTRE_AISLE,       # 3
    "depot_count":         DEFAULT_DEPOT_COUNT,        # 1
    "shelf_start_row":     DEFAULT_SHELF_START,        # 2
    "shelf_end_row":       DEFAULT_SHELF_END,          # 23
    "cross_aisle_row":     DEFAULT_CROSS_AISLE_ROW,    # 12
    "cross_aisle_on":      DEFAULT_CROSS_AISLE_ON,     # 1
}

# =============================================================================
# PARAMETER SWEEP RANGES (full valid ranges from rl_env.py)
# Format: (param_name, display_name, list_of_values)
# =============================================================================

SWEEPS = [
    ("aisle_width",        "$w_a$ — Aisle width",          list(range(2, 4))),
    ("centre_aisle_width", "$w_c$ — Centre aisle width",   list(range(1, 6))),
    ("depot_count",        "$n_d$ — Depot count",          list(range(1, 5))),
    ("shelf_start_row",    "$r_s$ — Shelf start row",      list(range(2, 9))),
    ("shelf_end_row",      "$r_e$ — Shelf end row",        list(range(18, 24))),
    # cross_aisle_row: must lie strictly inside [shelf_start, shelf_end]
    # → for default shelf [2, 23] valid r_ca is [3, 22]
    ("cross_aisle_row",    "$r_{ca}$ — Cross-aisle row",   list(range(3, 23))),
    ("cross_aisle_on",     "$b_{ca}$ — Cross-aisle on/off", [0, 1]),
]


# =============================================================================
# SWEEP RUNNER
# =============================================================================

def _simulate(kwargs, eval_runs):
    """Build active depot_cols and call run_single_simulation."""
    depot_cols = compute_depot_cols(kwargs["depot_count"])
    return run_single_simulation(
        aisle_width=kwargs["aisle_width"],
        centre_aisle_width=kwargs["centre_aisle_width"],
        depot_col=depot_cols[0],
        shelf_start_row=kwargs["shelf_start_row"],
        shelf_end_row=kwargs["shelf_end_row"],
        cross_aisle_row=kwargs["cross_aisle_row"],
        eval_runs=eval_runs,
        depot_cols=depot_cols,
        depot_count=kwargs["depot_count"],
        depot_row=0,
        cross_aisle_enabled=bool(kwargs["cross_aisle_on"]),
    )


def run_sweep(param_name, values, eval_runs):
    """Vary one parameter across its range; hold the others at DEFAULT."""
    results = []
    for val in values:
        kwargs = dict(DEFAULT)
        kwargs[param_name] = val

        # Maintain shelf_start < shelf_end and r_ca inside the shelf zone.
        if kwargs["shelf_start_row"] >= kwargs["shelf_end_row"]:
            kwargs["shelf_end_row"] = kwargs["shelf_start_row"] + 1
        kwargs["cross_aisle_row"] = max(
            kwargs["shelf_start_row"] + 1,
            min(kwargs["shelf_end_row"] - 1, kwargs["cross_aisle_row"])
        )

        result = _simulate(kwargs, eval_runs)
        if result is None:
            print(f"    {param_name}={val}: invalid layout — skipped")
            continue

        results.append({
            "param_name":      param_name,
            "param_value":     val,
            "picks_per_hour":  result["picks_per_hour"],
            "dist_per_agent":  result["dist_per_agent"],
            "congestion_rate": result["congestion_rate"],
            "num_items":       result["num_items"],
        })
        print(f"    {param_name}={val}: picks={result['picks_per_hour']:.1f}  "
              f"dist={result['dist_per_agent']:.0f}  "
              f"cong={result['congestion_rate']:.4f}")
    return results


# =============================================================================
# BASELINE EVALUATION (used to normalise the response curves)
# =============================================================================

BASELINE_FILE = "sensitivity_baseline.json"


def save_baseline(result, filename=BASELINE_FILE):
    import json
    import agent as _agent_module
    with open(filename, "w") as f:
        json.dump({
            "picks_per_hour":  result["picks_per_hour"],
            "dist_per_agent":  result["dist_per_agent"],
            "congestion_rate": result["congestion_rate"],
            "lambda_base":     _agent_module.LAMBDA_BASE,
        }, f, indent=2)


def load_baseline(filename=BASELINE_FILE):
    import json, os
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        return json.load(f)


def run_baseline(eval_runs):
    print("\n  Running baseline layout for normalisation...")
    result = _simulate(dict(DEFAULT), eval_runs)
    if result is None:
        raise RuntimeError("Baseline layout itself is invalid!")
    print(f"    Baseline picks/hr     = {result['picks_per_hour']:.2f}")
    print(f"    Baseline dist/agent   = {result['dist_per_agent']:.1f}")
    print(f"    Baseline congestion   = {result['congestion_rate']:.4f}")
    save_baseline(result)
    return result


# =============================================================================
# PLOTTING
# =============================================================================

KPI_STYLES = {
    "picks_per_hour":  {"label": "Picks/hr",        "color": "#2ca02c", "marker": "o"},
    "dist_per_agent":  {"label": "Dist/agent",      "color": "#d62728", "marker": "s"},
    "congestion_rate": {"label": "Congestion rate", "color": "#1f77b4", "marker": "^"},
}


def plot_sensitivity(all_results, baseline, output_file="sensitivity_analysis.png"):
    """
    For each parameter, plot the 3 KPIs normalised to their baseline value
    (1.0 = same as baseline). A flat horizontal line means the KPI is
    insensitive to that parameter.
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Sensitivity Analysis (KPIs normalised to baseline = 1.0)",
        fontsize=18, fontweight="bold", y=0.975,
    )

    # Just enough room for suptitle + subplot titles (pad=22) + delta-line text.
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.65, wspace=0.15,
                           top=0.915, bottom=0.05, left=0.06, right=0.97)

    for idx, (param_name, display_name, _) in enumerate(SWEEPS):
        data = [r for r in all_results if r["param_name"] == param_name]
        if not data:
            continue

        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

        x = [r["param_value"] for r in data]

        # Per-panel normaliser: use the sweep point AT the baseline parameter
        # value (not the separate baseline run) so the curve passes through 1.0
        # at the baseline by construction. Isolates the effect of THIS parameter
        # from simulation noise in the global baseline.
        baseline_val_for_param = DEFAULT[param_name]
        ref_point = next((r for r in data
                          if r["param_value"] == baseline_val_for_param), None)
        if ref_point is None:
            ref_point = data[0]   # fallback: first sweep point

        for kpi, style in KPI_STYLES.items():
            base_val = ref_point[kpi] if ref_point[kpi] != 0 else 1.0
            y_norm   = [r[kpi] / base_val for r in data]

            if param_name == "cross_aisle_on":
                # Binary — render as bars side-by-side for clarity.
                width  = 0.25
                offset = {"picks_per_hour": -width,
                          "dist_per_agent":  0.0,
                          "congestion_rate": width}[kpi]
                ax.bar([xi + offset for xi in x], y_norm, width=width,
                       color=style["color"], label=style["label"],
                       edgecolor="black", linewidth=0.5)
            else:
                ax.plot(x, y_norm,
                        color=style["color"], marker=style["marker"],
                        markersize=5, linewidth=1.8, label=style["label"])

        # Baseline reference line at y=1.0
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.6)

        # Mark the baseline parameter value (skip for binary)
        if param_name != "cross_aisle_on":
            ax.axvline(DEFAULT[param_name], color="grey", linestyle=":",
                       linewidth=1.2, alpha=0.7)

        # Per-panel sensitivity range
        picks = [r["picks_per_hour"]  for r in data]
        dist  = [r["dist_per_agent"]  for r in data]
        cong  = [r["congestion_rate"] for r in data]
        delta_line = (
            f"Δpicks={(max(picks)-min(picks))/ref_point['picks_per_hour']*100:+.1f}%  "
            f"Δdist={(max(dist)-min(dist))/ref_point['dist_per_agent']*100:+.1f}%  "
            f"Δcong={(max(cong)-min(cong))/(ref_point['congestion_rate'] or 1)*100:+.1f}%"
        )
        # Bold display name as the actual title; non-bold delta line below it.
        ax.set_title(display_name, fontsize=13, fontweight="bold", pad=22)
        ax.text(0.5, 1.01, delta_line, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10)
        ax.set_xlabel(display_name.split(" — ")[0], fontsize=13)
        ax.set_ylabel("KPI / baseline", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
        # Legend only on the first panel — the rest are visually self-explanatory.
        if idx == 0:
            ax.legend(fontsize=11, loc="best")

        if param_name == "cross_aisle_on":
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["off", "on"], fontsize=12)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    # Also save vector version (PDF) — infinite resolution for Word/LaTeX.
    pdf_file = output_file.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()
    print(f"\n  Response curves saved to {output_file}")


def plot_summary(all_results, baseline, output_file="sensitivity_summary.png"):
    """
    Single bar chart ranking parameters by overall sensitivity.
    Sensitivity index = max relative range across the 3 KPIs (%).
    """
    rows = []
    for param_name, display_name, _ in SWEEPS:
        data = [r for r in all_results if r["param_name"] == param_name]
        if not data:
            continue
        picks = [r["picks_per_hour"]  for r in data]
        dist  = [r["dist_per_agent"]  for r in data]
        cong  = [r["congestion_rate"] for r in data]
        d_pk = (max(picks)-min(picks)) / baseline["picks_per_hour"]  * 100
        d_ds = (max(dist) -min(dist))  / baseline["dist_per_agent"]  * 100
        d_cg = ((max(cong)-min(cong))  / baseline["congestion_rate"] * 100
                if baseline["congestion_rate"] > 0 else 0.0)
        rows.append({
            "param":  display_name.split(" — ")[0],
            "picks":  d_pk,
            "dist":   d_ds,
            "cong":   d_cg,
            "total":  d_pk + d_ds + d_cg,    # ranking key
        })

    rows.sort(key=lambda r: r["total"], reverse=True)

    labels = [r["param"] for r in rows]
    picks  = [r["picks"] for r in rows]
    dist   = [r["dist"]  for r in rows]
    cong   = [r["cong"]  for r in rows]

    x = np.arange(len(labels))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w, picks, w, color=KPI_STYLES["picks_per_hour"]["color"],
           label="Picks/hr range (%)")
    ax.bar(x,     dist,  w, color=KPI_STYLES["dist_per_agent"]["color"],
           label="Dist/agent range (%)")
    ax.bar(x + w, cong,  w, color=KPI_STYLES["congestion_rate"]["color"],
           label="Congestion range (%)")

    ax.set_title("Parameter Sensitivity Ranking "
                 "(relative KPI range as % of baseline)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Relative range (%)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    # Also save vector version (PDF) — infinite resolution for Word/LaTeX.
    pdf_file = output_file.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()
    print(f"  Summary plot saved to {output_file}")

    print("\n  Sensitivity ranking (by sum of relative KPI ranges):")
    print(f"    {'Param':<8} {'Picks Δ%':>10} {'Dist Δ%':>10} {'Cong Δ%':>10} {'Total':>10}")
    for r in rows:
        print(f"    {r['param']:<8} {r['picks']:>10.1f} {r['dist']:>10.1f} "
              f"{r['cong']:>10.1f} {r['total']:>10.1f}")


# =============================================================================
# CSV I/O
# =============================================================================

def save_csv(all_results, output_file="sensitivity_results.csv"):
    if not all_results:
        return
    keys = ["param_name", "param_value", "picks_per_hour",
            "dist_per_agent", "congestion_rate", "num_items"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  Raw results saved to {output_file}")


def load_csv(filename="sensitivity_results.csv"):
    rows = []
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "param_name":      row["param_name"],
                "param_value":     int(row["param_value"]),
                "picks_per_hour":  float(row["picks_per_hour"]),
                "dist_per_agent":  float(row["dist_per_agent"]),
                "congestion_rate": float(row["congestion_rate"]),
                "num_items":       int(row["num_items"]),
            })
    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="One-at-a-time sensitivity analysis for warehouse layout KPIs")
    parser.add_argument("--eval-runs", type=int, default=3,
                        help="Simulation runs per layout evaluation (default 3)")
    parser.add_argument("--lambda-base", type=float, default=None,
                        help="Override NHPP base arrival rate per tick "
                             "(default: use agent.py / rl_env.py value)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation; re-plot from sensitivity_results.csv")
    args = parser.parse_args()

    # Apply lambda override BEFORE any simulation runs — nhpp_arrival reads
    # agent.LAMBDA_BASE directly, so we have to mutate the module attribute.
    if args.lambda_base is not None:
        import agent as _agent_module
        import rl_env as _rl_env_module
        _agent_module.LAMBDA_BASE  = args.lambda_base
        _rl_env_module.LAMBDA_BASE = args.lambda_base
        print(f"  Overriding LAMBDA_BASE → {args.lambda_base}")

    if args.plot_only:
        print("  Loading sensitivity_results.csv...")
        all_results = load_csv()
        print(f"  Loaded {len(all_results)} rows.")
        baseline = load_baseline()
        import agent as _agent_module
        current_lambda = _agent_module.LAMBDA_BASE
        if baseline is None:
            print(f"  No {BASELINE_FILE} found — running baseline...")
            baseline = run_baseline(args.eval_runs)
        else:
            saved_lambda = baseline.get("lambda_base")
            if saved_lambda is not None and abs(saved_lambda - current_lambda) > 1e-9:
                print(f"  WARNING: cached baseline was built at λ={saved_lambda}, "
                      f"but current λ={current_lambda}. Regenerating...")
                baseline = run_baseline(args.eval_runs)
            else:
                print(f"  Loaded baseline from {BASELINE_FILE}: "
                      f"picks={baseline['picks_per_hour']:.2f}, "
                      f"dist={baseline['dist_per_agent']:.1f}, "
                      f"cong={baseline['congestion_rate']:.4f}  "
                      f"(λ={saved_lambda})")
        plot_sensitivity(all_results, baseline)
        plot_summary(all_results, baseline)
        return

    print("=" * 60)
    print("  WAREHOUSE LAYOUT SENSITIVITY ANALYSIS (OAT)")
    print("=" * 60)
    print("  Baseline layout:")
    for k, v in DEFAULT.items():
        print(f"    {k:<22} = {v}")
    print(f"  Eval runs per sweep point: {args.eval_runs}")
    print("=" * 60)

    baseline = run_baseline(args.eval_runs)

    all_results = []
    for param_name, display_name, values in SWEEPS:
        print(f"\n  Sweeping {display_name}  ({len(values)} values)...")
        all_results.extend(run_sweep(param_name, values, args.eval_runs))

    save_csv(all_results)
    plot_sensitivity(all_results, baseline)
    plot_summary(all_results, baseline)
    print("\n  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
