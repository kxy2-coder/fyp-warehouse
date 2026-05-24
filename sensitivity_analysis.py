# =============================================================================
# sensitivity_analysis.py — One-at-a-time sensitivity analysis for layout KPIs
# =============================================================================
# Varies each of the 6 layout parameters across its full range while keeping
# all other parameters fixed at the default layout values.
#
# For each parameter value, runs the full warehouse simulation (eval_runs times)
# and records picks/hr, distance/agent, and congestion rate.
#
# OUTPUT:
#   - sensitivity_results.csv   — raw KPI values for every parameter sweep
#   - sensitivity_analysis.png  — 6-panel KPI response curve figure
#
# USAGE:
#   python sensitivity_analysis.py
#   python sensitivity_analysis.py --eval-runs 5
# =============================================================================

import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for WSL/headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rl_env import (
    run_single_simulation,
    GRID_ROWS, GRID_COLS,
)

# =============================================================================
# DEFAULT LAYOUT (baseline all other parameters are fixed at)
# =============================================================================

DEFAULT = {
    "aisle_width":        2,
    "centre_aisle_width": 3,
    "depot_col":          GRID_COLS // 2,   # 17
    "shelf_start_row":    1,
    "shelf_end_row":      GRID_ROWS - 2,    # 23
    "cross_aisle_row":    (1 + GRID_ROWS - 2) // 2,  # 12
}

# =============================================================================
# PARAMETER SWEEP RANGES
# Each entry: (parameter_name, display_name, list_of_values_to_test)
# =============================================================================

SWEEPS = [
    ("aisle_width",        "Aisle width",         list(range(1, 4))),           # 1-3
    ("centre_aisle_width", "Centre aisle width",   list(range(1, 6))),           # 1-5
    ("depot_col",          "Depot column",         list(range(2, GRID_COLS-2))), # 2-32
    ("shelf_start_row",    "Shelf start row",      list(range(1, 7))),           # 1-6
    ("shelf_end_row",      "Shelf end row",        list(range(GRID_ROWS-7,
                                                              GRID_ROWS-1))),    # 18-23
    ("cross_aisle_row",    "Cross-aisle row",      list(range(1, GRID_ROWS-1))), # 1-23
]


def run_sweep(param_name, values, eval_runs):
    """
    Vary one parameter across its range, keep all others at DEFAULT.
    Returns list of dicts with parameter value and KPI results.
    """
    results = []

    for val in values:
        # Build kwargs from default, override the swept parameter
        kwargs = dict(DEFAULT)
        kwargs[param_name] = val

        # Ensure cross_aisle_row stays within shelf bounds
        kwargs["cross_aisle_row"] = max(
            kwargs["shelf_start_row"],
            min(kwargs["shelf_end_row"], kwargs["cross_aisle_row"])
        )

        result = run_single_simulation(
            aisle_width=kwargs["aisle_width"],
            centre_aisle_width=kwargs["centre_aisle_width"],
            depot_col=kwargs["depot_col"],
            shelf_start_row=kwargs["shelf_start_row"],
            shelf_end_row=kwargs["shelf_end_row"],
            cross_aisle_row=kwargs["cross_aisle_row"],
            eval_runs=eval_runs,
        )

        if result is not None:
            results.append({
                "param_name":    param_name,
                "param_value":   val,
                "picks_per_hour":  result["picks_per_hour"],
                "dist_per_agent":  result["dist_per_agent"],
                "congestion_rate": result["congestion_rate"],
                "num_items":       result["num_items"],
            })
        else:
            print(f"    Skipping {param_name}={val} (invalid layout)")

    return results


def plot_sensitivity(all_results, output_file="sensitivity_analysis.png"):
    """
    Plot KPI response curves for each parameter in a 6-panel figure.
    Each panel shows picks/hr, distance, and congestion on dual axes.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Sensitivity Analysis — One-at-a-Time Parameter Sweep\n"
                 "(all other parameters fixed at default layout)",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # Default values for vertical reference lines
    default_values = {
        "aisle_width":        DEFAULT["aisle_width"],
        "centre_aisle_width": DEFAULT["centre_aisle_width"],
        "depot_col":          DEFAULT["depot_col"],
        "shelf_start_row":    DEFAULT["shelf_start_row"],
        "shelf_end_row":      DEFAULT["shelf_end_row"],
        "cross_aisle_row":    DEFAULT["cross_aisle_row"],
    }

    for idx, (param_name, display_name, _) in enumerate(SWEEPS):
        # Filter results for this parameter
        data = [r for r in all_results if r["param_name"] == param_name]
        if not data:
            continue

        x      = [r["param_value"]   for r in data]
        picks  = [r["picks_per_hour"] for r in data]
        dist   = [r["dist_per_agent"] for r in data]
        cong   = [r["congestion_rate"] for r in data]
        items  = [r["num_items"]      for r in data]

        row, col = divmod(idx, 2)
        ax1 = fig.add_subplot(gs[row, col])
        ax2 = ax1.twinx()

        # Picks/hr — left axis (green)
        l1, = ax1.plot(x, picks, color="#2ca02c", marker="o", markersize=4,
                       linewidth=2, label="Picks/hr")

        # Distance — left axis (orange)
        l2, = ax1.plot(x, dist, color="#d62728", marker="s", markersize=4,
                       linewidth=2, label="Dist/agent (cells)")

        # Congestion — right axis (pink)
        l3, = ax2.plot(x, cong, color="#e377c2", marker="^", markersize=4,
                       linewidth=2, linestyle="--", label="Congestion rate")

        # Default value vertical line
        ax1.axvline(x=default_values[param_name], color="grey",
                    linestyle=":", linewidth=1.5, label="Default value")

        # Axis labels
        ax1.set_xlabel(display_name, fontsize=10)
        ax1.set_ylabel("Picks/hr  |  Distance (cells)", fontsize=8)
        ax2.set_ylabel("Congestion rate", fontsize=8, color="#e377c2")
        ax2.tick_params(axis="y", labelcolor="#e377c2")

        ax1.set_title(f"{display_name}", fontsize=11, fontweight="bold")

        # Range annotation
        picks_range = max(picks) - min(picks)
        dist_range  = max(dist)  - min(dist)
        cong_range  = max(cong)  - min(cong)
        ax1.set_title(
            f"{display_name}\n"
            f"Δpicks={picks_range:.1f}  "
            f"Δdist={dist_range:.0f}  "
            f"Δcong={cong_range:.4f}",
            fontsize=9, fontweight="bold"
        )

        # Legend
        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=7, loc="best")

        ax1.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n  Sensitivity plot saved to {output_file}")
    plt.close()


def save_csv(all_results, output_file="sensitivity_results.csv"):
    if not all_results:
        return
    keys = all_results[0].keys()
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  Raw results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="One-at-a-time sensitivity analysis for warehouse layout KPIs")
    parser.add_argument("--eval-runs", type=int, default=3,
                        help="Simulation runs per layout evaluation (default 3)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation, load existing sensitivity_results.csv and plot")
    args = parser.parse_args()

    # If --plot-only, just load CSV and regenerate plot
    if args.plot_only:
        print("  Loading existing sensitivity_results.csv...")
        all_results = []
        with open("sensitivity_results.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_results.append({
                    "param_name":      row["param_name"],
                    "param_value":     int(row["param_value"]),
                    "picks_per_hour":  float(row["picks_per_hour"]),
                    "dist_per_agent":  float(row["dist_per_agent"]),
                    "congestion_rate": float(row["congestion_rate"]),
                    "num_items":       int(row["num_items"]),
                })
        print(f"  Loaded {len(all_results)} records.")
        plot_sensitivity(all_results)
        print("  Done.")
        return

    print("=" * 60)
    print("  WAREHOUSE LAYOUT SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"  Default layout:")
    print(f"    aisle_width        = {DEFAULT['aisle_width']}")
    print(f"    centre_aisle_width = {DEFAULT['centre_aisle_width']}")
    print(f"    depot_col          = {DEFAULT['depot_col']}")
    print(f"    shelf_start_row    = {DEFAULT['shelf_start_row']}")
    print(f"    shelf_end_row      = {DEFAULT['shelf_end_row']}")
    print(f"    cross_aisle_row    = {DEFAULT['cross_aisle_row']}")
    print(f"  Eval runs per value : {args.eval_runs}")
    print("=" * 60)

    all_results = []

    for param_name, display_name, values in SWEEPS:
        print(f"\n  Sweeping: {display_name} ({min(values)} → {max(values)}, "
              f"{len(values)} values)...")
        results = run_sweep(param_name, values, args.eval_runs)
        all_results.extend(results)
        print(f"    Done — {len(results)} valid configurations")

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    for param_name, display_name, _ in SWEEPS:
        data = [r for r in all_results if r["param_name"] == param_name]
        if not data:
            continue

        picks  = [r["picks_per_hour"]  for r in data]
        dist   = [r["dist_per_agent"]  for r in data]
        cong   = [r["congestion_rate"] for r in data]

        print(f"\n  {display_name}:")
        print(f"    Picks/hr      : {min(picks):.1f} — {max(picks):.1f}  "
              f"(range: {max(picks)-min(picks):.1f})")
        print(f"    Dist/agent    : {min(dist):.0f} — {max(dist):.0f}  "
              f"(range: {max(dist)-min(dist):.0f})")
        print(f"    Congestion    : {min(cong):.4f} — {max(cong):.4f}  "
              f"(range: {max(cong)-min(cong):.4f})")

    save_csv(all_results)
    plot_sensitivity(all_results)

    print("\n  Done. Run 'python sensitivity_analysis.py' to regenerate plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
