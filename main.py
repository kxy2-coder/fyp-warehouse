# =============================================================================
# main.py — Run the Simulation
# =============================================================================
# Runs an N-agent warehouse simulation.
# Each agent completes a randomly drawn number of pickup orders per run.
# Congestion is measured via JuPedSim physics: fraction of walking ticks
# where agents moved below 20% of free speed (slow_ticks / walk_ticks).
#
# Usage:
#   python main.py                              # default 25x35 grid, 4 agents
#   python main.py --agents 6                   # run with 6 agents
#   python main.py --rows 20 --cols 25          # larger grid
#   python main.py --speed 0.15                 # faster animation
#   python main.py --aisle-width 3              # wider aisles
#   python main.py --centre-aisle 5             # wider centre aisle (must be odd)
#   python main.py --depot-col 5                # shift depot left
# =============================================================================

import argparse
import math
import random
import pygame
import sys
import time

from grid           import Grid, EMPTY, SHELF, ITEM, DEPOT
import agent as _agent_module
from agent          import (nhpp_arrival, SHIFT_TICKS, LAMBDA_BASE,
                            DEMAND_BETA, NUM_RUNS)
from jupedsim_agent import (JupedSimAgent, FREE_SPEED,
                            CONGESTION_SPEED, CONGESTION_RADIUS,
                            STATE_WAITING, STATE_TO_ITEM, STATE_PICKING,
                            STATE_TO_DEPOT, STATE_RESTING, STATE_ALL_DONE)
from metrics        import MetricsTracker, TraceLogger

# No BLOCKED_WAIT_TICKS needed — JuPedSim handles collision avoidance physically
BLOCKED_WAIT_TICKS = 1   # kept as placeholder so any references don't crash


# =============================================================================
# CLI ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="N-agent warehouse simulator")
    parser.add_argument("--agents",        type=int,   default=6,    help="Number of agents (default 6)")
    parser.add_argument("--rows",          type=int,   default=25,   help="Grid rows (default 25)")
    parser.add_argument("--cols",          type=int,   default=35,   help="Grid cols (default 35, odd recommended)")
    parser.add_argument("--speed",         type=float, default=0.1, help="Seconds between steps (default 0.15)")
    parser.add_argument("--aisle-width",   type=int,   default=2,    help="Aisle width between shelf blocks (default 2)")
    parser.add_argument("--centre-aisle",  type=int,   default=3,    help="Centre aisle width, must be odd (default 3)")
    parser.add_argument("--depot-row",     type=int,   default=None, help="Depot row (default 0 = top row)")
    parser.add_argument("--depot-col",     type=int,   default=None, help="Single depot column (overridden by --depot-count)")
    parser.add_argument("--depot-count",   type=int,   default=None, choices=[1,2,3,4],
                        help="Number of equally-spaced depots 1-4 (positions auto-computed; overrides --depot-col)")
    parser.add_argument("--shelf-start",     type=int,   default=2,    help="First row shelves appear (default 3 for Sim_V3 JuPedSim agents)")
    parser.add_argument("--shelf-end",         type=int,   default=23,   help="Last row shelves appear (default 23 = rows-2, matching RL training)")
    parser.add_argument("--cross-aisle-row",   type=int,   default=None, help="Row of horizontal cross-aisle (default = middle of shelf zone)")
    parser.add_argument("--cross-aisle-width", type=int,   default=2,    help="Width in rows of the cross-aisle (default 2)")
    parser.add_argument("--no-cross-aisle",    action="store_true",      help="Disable the cross-aisle entirely")
    parser.add_argument("--cell-size",     type=int,   default=35,   help="Pixel size per cell (default 35)")
    parser.add_argument("--experiment",      action="store_true",      help="Silent multi-run averaging mode (no Pygame)")
    parser.add_argument("--runs",            type=int,   default=None, help="Override NUM_RUNS for --experiment mode")
    parser.add_argument("--replenish-delay", type=int,   default=100,  help="Ticks before a picked shelf restocks (default 100)")
    parser.add_argument("--lambda-base",    type=float, default=0.1, help="NHPP base arrival rate per tick (default 0.0833)")
    parser.add_argument("--beta",           type=float, default=None, help=f"NHPP amplitude parameter (default {DEMAND_BETA})")
    parser.add_argument("--plot-workload",  action="store_true",      help="Save workload analysis plot after --experiment run")
    parser.add_argument("--plot-traces",   action="store_true",      help="Save per-agent trace plot after --experiment run")
    parser.add_argument("--plot-heatmap",  action="store_true",      help="Save conflict heatmap after --experiment run")
    parser.add_argument("--plot-layout",   action="store_true",      help="Save clean black/white layout figure (paper-style)")
    parser.add_argument("--bin",            type=int,   default=600,  help="Bin size (ticks) for workload plot (default 600 = 10 min)")
    parser.add_argument("--save",           action="store_true",      help="Save experiment results to results/ folder for the dashboard")
    parser.add_argument("--name",           type=str,   default=None, help="Label for this saved run (default: timestamp + config)")
    parser.add_argument("--replay",         action="store_true",
                        help="Record the full shift first, then open an interactive "
                             "replay with pause / rewind / scrub controls. Adds ~10-30 s "
                             "of upfront recording time before the window opens. "
                             "Without this flag, the simulation runs live as before.")
    return parser.parse_args()



# =============================================================================
# EXPERIMENT MODE - silent multi-run averaging
# =============================================================================

def run_experiment(args):
    """
    Run NUM_RUNS silent simulations, compute and print the 3 KPIs,
    then open the Pygame visual replaying the most representative run.
    """
    from rl_env import compute_depot_cols as _compute_depot_cols

    # Resolve depot columns: --depot-count takes priority over --depot-col.
    # If neither is given, default to 1 depot at centre column.
    if args.depot_count is not None:
        _depot_cols = _compute_depot_cols(args.depot_count, args.cols)
    elif args.depot_col is not None:
        _depot_cols = [args.depot_col]
    else:
        _depot_cols = [args.cols // 2]   # single centred depot (default)

    num_runs = args.runs if args.runs is not None else NUM_RUNS

    print("=" * 60)
    print("  EXPERIMENT MODE")
    print("  Runs: {}   Agents: {}   Grid: {}x{}".format(num_runs, args.agents, args.rows, args.cols))
    print("  NHPP demand: λ_base={}, β={}, shift={} ticks (8 hr)".format(LAMBDA_BASE, DEMAND_BETA, SHIFT_TICKS))
    print("=" * 60)

    accumulated  = {}
    run_records  = []

    # Workload tracking — collected when --plot-workload or --save is set
    all_arrivals    = []
    all_completions = []
    collect_workload = args.plot_workload or args.save

    # Trace accumulator: dict keyed by (tick, agent_id) → list of snapshots
    trace_accum = {}

    # Per-tick speed accumulator — only collected when --plot-traces is set.
    # Shape: (SHIFT_TICKS, n_agents). NaN when agent is not walking that tick.
    # Averaged across runs then saved to speed_trace.csv for high-res plot.
    import numpy as _np
    if args.plot_traces:
        _speed_sum   = _np.zeros((SHIFT_TICKS, args.agents), dtype=_np.float64)
        _speed_count = _np.zeros((SHIFT_TICKS, args.agents), dtype=_np.int32)

    # Traffic density accumulator: (row, col) → total agent-ticks across runs
    # (all walking ticks — shows popular paths)
    spatial_accum = {}

    # Congestion-only accumulator: (row, col) → slow-tick count across runs
    # (only ticks where agent speed < 20% free speed — shows true bottlenecks)
    congestion_accum = {}

    for run_i in range(1, num_runs + 1):
        seed = run_i * 7919
        random.seed(seed)

        grid = Grid(
            rows=args.rows, cols=args.cols,
            aisle_width=args.aisle_width,
            centre_aisle_width=args.centre_aisle,
            depot_row=args.depot_row,
            depot_cols=_depot_cols,
            shelf_start_row=args.shelf_start,
            shelf_end_row=args.shelf_end,
            cross_aisle_row=args.cross_aisle_row,
            cross_aisle_width=args.cross_aisle_width,
            cross_aisle_enabled=not args.no_cross_aisle,
            replenish_delay=args.replenish_delay,
        )
        num_items = len(grid.get_all_item_positions())
        # JupedSimAgent replaces Agent — JuPedSim handles physical movement
        agent        = JupedSimAgent(args.agents, grid)
        metrics      = MetricsTracker()
        trace_logger = TraceLogger(args.agents, bin_size=600)

        if collect_workload:
            import numpy as np
            run_arrivals    = np.zeros(SHIFT_TICKS, dtype=np.int32)
            run_completions = np.zeros(SHIFT_TICKS, dtype=np.int32)
            prev_completed  = 0

        for tick in range(SHIFT_TICKS):
            if nhpp_arrival(tick):
                agent.add_job()
                if collect_workload:
                    run_arrivals[tick] = 1
            agent.step()          # JuPedSim moves agents + handles avoidance
            grid.tick_replenishment()

            # Heatmap accumulation — two separate counters per cell:
            #   spatial_accum    : all walking ticks  → traffic density (popular paths)
            #   congestion_accum : slow ticks only    → true congestion hotspots
            # Only agents actively walking (TO_ITEM / TO_DEPOT) are counted.
            if args.plot_heatmap or args.save:
                for i in range(agent.n):
                    if agent.state[i] in (STATE_TO_ITEM, STATE_TO_DEPOT):
                        r, c = agent.get_row_col(i)
                        if grid.is_walkable(r, c):   # never count shelf cells
                            spatial_accum[(r, c)] = spatial_accum.get((r, c), 0) + 1
                            # Congestion heatmap: same two-condition definition as
                            # jupedsim_agent.py — slow AND another agent nearby
                            if agent.get_speed(i) < CONGESTION_SPEED:
                                cx_i, cy_i = agent.get_xy(i)
                                for _j in range(agent.n):
                                    if _j != i:
                                        cx_j, cy_j = agent.get_xy(_j)
                                        if math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2) < CONGESTION_RADIUS:
                                            congestion_accum[(r, c)] = congestion_accum.get((r, c), 0) + 1
                                            break

            # Per-tick speed — record every second when --plot-traces is set
            if args.plot_traces:
                for i in range(agent.n):
                    if agent.state[i] in (STATE_TO_ITEM, STATE_TO_DEPOT):
                        _speed_sum[tick, i]   += agent.get_speed(i)
                        _speed_count[tick, i] += 1

            if (tick + 1) % 600 == 0:
                trace_logger.snapshot(tick + 1, agent)
            if collect_workload:
                cur = int(agent.orders_completed.sum())
                run_completions[tick] = cur - prev_completed
                prev_completed = cur

        if collect_workload:
            all_arrivals.append(run_arrivals)
            all_completions.append(run_completions)

        for rec in trace_logger.records:
            key = (rec["tick"], rec["agent_id"])
            trace_accum.setdefault(key, []).append(rec)

        raw = agent.collect_raw()
        for k, v in raw.items():
            accumulated[k] = accumulated.get(k, 0.0) + v

        completed_ = raw["jobs_completed"]
        run_picks  = completed_ / 8.0
        run_dist   = raw["total_distance"] / args.agents
        run_cong   = agent.congestion_rate() * 100   # as percentage
        run_records.append((seed, run_picks, run_dist, run_cong))

        print("  Run {:3d}/{} | arrived={:.0f} | completed={:.0f} | dist={:.0f} | conflicts={:.0f}".format(
            run_i, num_runs,
            raw["jobs_arrived"], raw["jobs_completed"],
            raw["total_distance"], raw["cell_conflicts"]))

    avg = {k: v / num_runs for k, v in accumulated.items()}

    all_picks = [r[1] for r in run_records]
    all_dist  = [r[2] for r in run_records]
    all_cong  = [r[3] for r in run_records]

    picks_per_hour      = avg["jobs_completed"] / 8.0
    dist_per_agent      = avg["total_distance"] / args.agents
    congestion_rate     = sum(all_cong) / len(all_cong)   # avg % across runs
    # Reference: item count of the *actual* default layout (shelf_start_row=3).
    # Previously hardcoded as 352, which was the old shelf_start_row=1 count.
    # Building the reference grid takes <1ms and is always accurate.
    _ref_grid       = Grid(rows=args.rows, cols=args.cols,
                           shelf_start_row=3,
                           depot_row=args.depot_row, depot_cols=_depot_cols,
                           cross_aisle_enabled=not args.no_cross_aisle)
    DEFAULT_STORAGE     = len(_ref_grid.get_all_item_positions())
    storage_utilisation = min(num_items / DEFAULT_STORAGE, 1.0) * 100

    range_picks = max(all_picks) - min(all_picks) or 1.0
    range_dist  = max(all_dist)  - min(all_dist)  or 1.0
    range_cong  = max(all_cong)  - min(all_cong)  or 1.0

    best_seed, best_dist = None, float("inf")
    for seed, rp, rd, rc in run_records:
        d = (((rp - picks_per_hour) / range_picks) ** 2 +
             ((rd - dist_per_agent) / range_dist)  ** 2 +
             ((rc - congestion_rate) / range_cong) ** 2)
        if d < best_dist:
            best_dist, best_seed = d, seed

    W = 60
    print()
    print("=" * W)
    print("  RESULTS  ({} runs averaged)".format(num_runs))
    print("=" * W)
    print()
    print("  KEY PERFORMANCE INDICATORS")
    print("  " + "-" * (W - 2))
    print("  {:<42s} {:>10.2f}".format("Picks per Hour  [PRIMARY]",               picks_per_hour))
    print("  {:<42s} {:>10.1f}".format("Avg Travel Distance per Agent (metres)",   dist_per_agent))
    print("  {:<42s} {:>9.2f}%".format("Congestion Rate  (% shift time affected)", congestion_rate))
    print("  {:<42s} {:>9.1f}%".format("Storage Utilisation (vs default layout)",  storage_utilisation))
    print("  {:<42s} {:>10d}".format("Item cells available",                        num_items))
    print()
    print("  RAW AVERAGES")
    print("  " + "-" * (W - 2))
    print("  {:<42s} {:>10.1f}".format("Avg jobs arrived (NHPP)",      avg["jobs_arrived"]))
    print("  {:<42s} {:>10.1f}".format("Avg jobs completed",           avg["jobs_completed"]))
    print("  {:<42s} {:>10.1f}".format("Avg total distance (metres)",  avg["total_distance"]))
    print("  {:<42s} {:>10.1f}".format("Slow-walk ticks (congestion proxy)", avg["cell_conflicts"]))
    print("=" * W)
    print()
    print("  Replaying most representative run (seed={}) — close window to exit.".format(best_seed))
    print("=" * W)
    print()

    if args.plot_workload:
        from plot_workload import plot_from_data, save_workload_csv
        # Always save tick-level CSV alongside the plot so it can be
        # re-rendered later (same convention as heatmap/traces).
        save_workload_csv(all_arrivals, all_completions,
                          out_file="workload_data.csv")
        plot_from_data(all_arrivals, all_completions,
                       bin_size=args.bin, n_agents=args.agents)

    # Save averaged agent traces to CSV
    trace_csv = "agent_traces.csv"
    avg_trace_records = []
    for (tick, agent_id), recs in sorted(trace_accum.items()):
        n = len(recs)
        avg_trace_records.append({
            "tick":       tick,
            "agent_id":   agent_id,
            "picks":      round(sum(r["picks"]      for r in recs) / n, 2),
            "distance":   round(sum(r["distance"]   for r in recs) / n, 1),
            "idle_ticks": round(sum(r["idle_ticks"] for r in recs) / n, 2),
            "slow_ticks": round(sum(r["slow_ticks"] for r in recs) / n, 2),
            "walk_ticks": round(sum(r["walk_ticks"] for r in recs) / n, 2),
        })
    import csv as _csv
    if avg_trace_records:
        with open(trace_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=avg_trace_records[0].keys())
            writer.writeheader()
            writer.writerows(avg_trace_records)
        print(f"  Agent traces saved → {trace_csv}")

    if args.plot_traces:
        from plot_agent_traces import plot_from_traces

        # Save per-tick speed (averaged across runs) to speed_trace.csv
        speed_csv = "speed_trace.csv"
        import csv as _csv_spd
        speed_avg = _np.where(_speed_count > 0,
                              _speed_sum / _np.maximum(_speed_count, 1),
                              _np.nan)
        with open(speed_csv, "w", newline="") as _f:
            _w = _csv_spd.writer(_f)
            _w.writerow(["tick"] + [f"agent_{i+1}" for i in range(args.agents)])
            for t in range(SHIFT_TICKS):
                row = [t] + [
                    round(float(speed_avg[t, i]), 4) if not _np.isnan(speed_avg[t, i]) else ""
                    for i in range(args.agents)
                ]
                _w.writerow(row)
        print(f"  Per-tick speed trace saved → {speed_csv}")

        from plot_agent_traces import plot_speed_trace

        # 3-panel figure: picks, distance, idle ticks (10-min bins)
        plot_from_traces(trace_csv)
        # Full-width standalone figure: per-tick walking speed with rolling mean
        plot_speed_trace(speed_csv, n_agents=args.agents)

    # Save averaged heatmap CSVs
    import csv as _csv2
    heatmap_csv      = "traffic_heatmap.csv"
    congestion_csv   = "congestion_heatmap.csv"

    def _write_heatmap_csv(filename, accum):
        with open(filename, "w", newline="") as f:
            writer = _csv2.writer(f)
            writer.writerow(["row", "col", "avg_count"])
            for (row, col), total in sorted(accum.items()):
                writer.writerow([row, col, round(total / num_runs, 4)])

    if spatial_accum:
        _write_heatmap_csv(heatmap_csv, spatial_accum)
        print(f"  Traffic density heatmap CSV saved  → {heatmap_csv}")
    if congestion_accum:
        _write_heatmap_csv(congestion_csv, congestion_accum)
        print(f"  Congestion hotspot heatmap CSV saved → {congestion_csv}")

    if args.plot_heatmap:
        from plot_heatmap import plot_from_csv
        _grid_kwargs = dict(
            aisle_width=args.aisle_width,
            centre_aisle_width=args.centre_aisle,
            depot_row=args.depot_row,
            depot_cols=_depot_cols,
            shelf_start_row=args.shelf_start,
            shelf_end_row=args.shelf_end,
            cross_aisle_row=args.cross_aisle_row,
            cross_aisle_width=args.cross_aisle_width,
            cross_aisle_enabled=not args.no_cross_aisle,
        )
        # Two separate figures with independent colour scales so each shows
        # full detail — a shared scale would wash out the congestion heatmap
        # since its values are much lower than traffic density.
        if spatial_accum:
            plot_from_csv(heatmap_csv,
                          grid_rows=args.rows, grid_cols=args.cols,
                          out_file="traffic_heatmap.png",
                          **_grid_kwargs)
        if congestion_accum:
            plot_from_csv(congestion_csv,
                          grid_rows=args.rows, grid_cols=args.cols,
                          out_file="congestion_heatmap.png",
                          **_grid_kwargs)

    if args.save:
        import json as _json, shutil as _shutil, os as _os
        from datetime import datetime as _dt
        run_name = args.name if args.name else \
            "{}_agents{}_{}x{}".format(_dt.now().strftime("%Y-%m-%d_%H-%M"),
                                       args.agents, args.rows, args.cols)
        run_dir = _os.path.join("results", run_name)
        _os.makedirs(run_dir, exist_ok=True)
        for fname in [trace_csv, heatmap_csv, congestion_csv]:
            if _os.path.exists(fname):
                _shutil.copy2(fname, _os.path.join(run_dir, _os.path.basename(fname)))
        if all_arrivals:
            import numpy as _np
            _np.savez(_os.path.join(run_dir, "workload_data.npz"),
                      arrivals=_np.array(all_arrivals),
                      completions=_np.array(all_completions))
        ref_grid = Grid(
            rows=args.rows, cols=args.cols,
            aisle_width=args.aisle_width,
            centre_aisle_width=args.centre_aisle,
            depot_row=args.depot_row, depot_cols=_depot_cols,
            shelf_start_row=args.shelf_start, shelf_end_row=args.shelf_end,
            cross_aisle_row=args.cross_aisle_row,
            cross_aisle_width=args.cross_aisle_width,
            cross_aisle_enabled=not args.no_cross_aisle,
        )
        _summary = {
            "name": run_name,
            "timestamp": _dt.now().isoformat(),
            "config": {
                "agents":          args.agents,
                "rows":            args.rows,
                "cols":            args.cols,
                "aisle_width":     args.aisle_width,
                "centre_aisle":    args.centre_aisle,
                "depot_row":       ref_grid.depot_row,
                "depot_col":       ref_grid.depot_col,
                "shelf_start":     ref_grid.shelf_start_row,
                "shelf_end":       ref_grid.shelf_end_row,
                "cross_aisle_row": ref_grid.cross_aisle_row,
                "num_runs":        num_runs,
            },
            "kpis": {
                "picks_per_hour":  round(picks_per_hour, 4),
                "dist_per_agent":  round(dist_per_agent, 4),
                "congestion_rate": round(congestion_rate, 4),
            },
            "raw_averages": {
                "jobs_arrived":      round(avg["jobs_arrived"], 2),
                "jobs_completed":    round(avg["jobs_completed"], 2),
                "total_distance":  round(avg["total_distance"], 2),
                "cell_conflicts":  round(avg["cell_conflicts"], 2),
            },
        }
        with open(_os.path.join(run_dir, "run_summary.json"), "w") as _f:
            _json.dump(_summary, _f, indent=2)
        extras = ", workload_data.npz" if all_arrivals else ""
        print(f"\n  Results saved → results/{run_name}/")
        print(f"  Files: run_summary.json, agent_traces.csv, conflict_heatmap.csv{extras}")

    run_visual(args, demo_mode=True, kpi_results={
        "num_runs":            num_runs,
        "picks_per_hour":      picks_per_hour,
        "dist_per_agent":      dist_per_agent,
        "congestion_rate":     congestion_rate,
        "storage_utilisation": storage_utilisation,
        "num_items":           num_items,
    }, replay_seed=best_seed)


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

CELL_SIZE   = 40
MARGIN      = 2
PANEL_WIDTH = 320

COL_BG           = (15,  20,  30)
COL_EMPTY        = (30,  38,  50)
COL_SHELF        = (50,  58,  72)
COL_ITEM_DOT     = (220, 160,  40)
COL_DEPOT        = (30,  70, 130)
COL_DEPOT_LABEL  = (100, 160, 255)
COL_CONFLICT     = (220,  50,  50)
COL_PANEL        = (20,  26,  36)
COL_WHITE        = (230, 235, 245)
COL_MUTED        = (100, 115, 135)
COL_DONE         = ( 60, 200,  80)
COL_FATIGUE_LOW  = ( 60, 200,  80)
COL_FATIGUE_MED  = (220, 180,  40)
COL_FATIGUE_HIGH = (220,  60,  50)
COL_RESTING      = ( 80, 140, 220)

AGENT_COLOURS = [
    ( 60, 200,  80),
    ( 60, 180, 220),
    (220, 100,  50),
    (180,  60, 200),
    (220, 220,  60),
    (200,  80, 120),
    ( 80, 200, 180),
    (200, 140,  60),
]

PATH_COLOURS = [
    ( 40,  80, 120),
    ( 40, 110,  90),
    (100,  60,  40),
    ( 80,  40, 100),
    ( 90,  90,  20),
    (100,  40,  60),
    ( 30,  90,  80),
    ( 90,  70,  20),
]


def agent_colour(agent_id):
    return AGENT_COLOURS[(agent_id - 1) % len(AGENT_COLOURS)]

def target_colour(agent_id):
    return AGENT_COLOURS[(agent_id - 1) % len(AGENT_COLOURS)]  # matches agent colour

def path_colour(agent_id):
    return PATH_COLOURS[(agent_id - 1) % len(PATH_COLOURS)]


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def cell_rect(row, col):
    x = col * (CELL_SIZE + MARGIN) + MARGIN
    y = row * (CELL_SIZE + MARGIN) + MARGIN
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def jps_to_pixel(jps_x, jps_y):
    """
    Convert a continuous JuPedSim position (metres) to pygame pixel coordinates.

    JuPedSim cell centres sit at (col + 0.5, row + 0.5) in metres.
    The pygame cell centre for (row, col) is at:
        px = col * (CELL_SIZE + MARGIN) + MARGIN + CELL_SIZE // 2
           = MARGIN//2 + jps_x * (CELL_SIZE + MARGIN)          [simplified]

    This means agents are drawn at their true sub-cell position,
    giving smooth continuous movement instead of cell-by-cell snapping.
    """
    px = int(MARGIN // 2 + jps_x * (CELL_SIZE + MARGIN))
    py = int(MARGIN // 2 + jps_y * (CELL_SIZE + MARGIN))
    return px, py




def draw_grid(surface, grid, agent, metrics, font_small):
    """Draw the warehouse grid, agents, paths, targets, conflict flash."""
    # JuPedSim routes internally — no pre-computed path to visualise.
    # In Sim_V2, A* produced a list of cells shown as dots on screen.
    # Here we just use empty sets (path dots are simply not drawn).
    upcoming = [set() for _ in range(agent.n)]

    for r in range(grid.rows):
        for c in range(grid.cols):
            rect = cell_rect(r, c)
            cell = grid.cells[r, c]
            pos  = (r, c)

            if cell == DEPOT:
                colour = COL_DEPOT
            elif cell in (SHELF, ITEM):
                colour = COL_SHELF
            else:
                colour = COL_EMPTY

            pygame.draw.rect(surface, colour, rect, border_radius=3)

            if cell == ITEM:
                dot_size = CELL_SIZE // 5
                dot_rect = pygame.Rect(
                    rect.centerx - dot_size // 2,
                    rect.centery - dot_size // 2,
                    dot_size, dot_size
                )
                pygame.draw.rect(surface, COL_ITEM_DOT, dot_rect, border_radius=2)

            if CELL_SIZE >= 18 and cell in (SHELF, ITEM):
                label = grid.shelf_labels.get((r, c), "")
                if label:
                    lbl_surf = font_small.render(label, True, COL_MUTED)
                    lbl_rect = lbl_surf.get_rect(center=rect.center)
                    if cell == ITEM:
                        lbl_rect.y = rect.y + 3
                    surface.blit(lbl_surf, lbl_rect)

            dot_r = CELL_SIZE // 8
            cx, cy = rect.center
            for i in range(agent.n):
                agent_id = i + 1
                pos_i    = (int(agent.pos_row[i]), int(agent.pos_col[i]))
                if not agent.is_done(i) and pos == agent.targets[i]:
                    pygame.draw.rect(surface, target_colour(agent_id), rect, width=2, border_radius=3)
                if pos in upcoming[i] and pos != pos_i and not agent.is_done(i):
                    pygame.draw.circle(surface, path_colour(agent_id), (cx, cy), dot_r)

            if metrics.flash_timer > 0 and pos == metrics.conflict_cell:
                pygame.draw.rect(surface, COL_CONFLICT, rect, width=3, border_radius=3)

    # Draw agents on top — use continuous JuPedSim position for smooth movement
    for i in range(agent.n):
        if not agent.is_done(i):
            jps_x, jps_y = agent.get_xy(i)
            cx, cy       = jps_to_pixel(jps_x, jps_y)
            col_fill     = agent_colour(i + 1)
            if agent.state[i] == STATE_RESTING:
                pygame.draw.circle(surface, COL_RESTING, (cx, cy),
                                   CELL_SIZE // 2 - 2, width=3)
            elif agent.state[i] == STATE_PICKING:
                pygame.draw.circle(surface, COL_RESTING,
                                   (cx, cy), CELL_SIZE // 2 - 2, width=3)
            pygame.draw.circle(surface, col_fill, (cx, cy), CELL_SIZE // 2 - 4)
            pygame.draw.circle(surface, COL_WHITE, (cx, cy), CELL_SIZE // 8)


def draw_panel(surface, grid, agent, metrics, font_big, font_med, font_small,
               kpi_results=None, sim_tick=0):
    """Right-side info panel with agent status and KPI results."""
    panel_x = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    pygame.draw.rect(surface, COL_PANEL,
                     pygame.Rect(panel_x, 0, PANEL_WIDTH, surface.get_height()))
    pygame.draw.line(surface, (40, 50, 65),
                     (panel_x, 0), (panel_x, surface.get_height()), 1)

    x           = panel_x + 16
    y           = 20
    bar_width   = PANEL_WIDTH - 36
    divider_end = panel_x + PANEL_WIDTH - 16

    surface.blit(font_big.render("WAREHOUSE SIM", True, COL_DEPOT_LABEL), (x, y))
    y += 28
    surface.blit(font_small.render(f"{agent.n} agents | NHPP λ={LAMBDA_BASE:.4f} β={DEMAND_BETA}", True, COL_MUTED), (x, y))
    y += 18

    # ── Shift clock ───────────────────────────────────────────────────────────
    total_secs  = sim_tick
    s_hrs       = total_secs // 3600
    s_mins      = (total_secs % 3600) // 60
    s_secs      = total_secs % 60
    time_str    = f"{s_hrs}hr {s_mins:02d}min {s_secs:02d}sec"
    progress    = min(sim_tick / SHIFT_TICKS, 1.0)

    # Colour: green → yellow → red
    if progress < 0.5:
        t = progress / 0.5
        clock_col = (int(60 + t * 160), int(200 - t * 60), 60)
    else:
        t = (progress - 0.5) / 0.5
        clock_col = (220, int(140 - t * 100), 60)

    surface.blit(font_med.render(time_str, True, clock_col), (x, y)); y += 18

    # Thin progress bar across the full panel width
    bx, by, bh = x, y, 5
    pygame.draw.rect(surface, (40, 50, 65),  pygame.Rect(bx, by, bar_width, bh), border_radius=2)
    filled = int(bar_width * progress)
    if filled > 0:
        pygame.draw.rect(surface, clock_col, pygame.Rect(bx, by, filled, bh), border_radius=2)
    y += bh + 10

    # State integer → display string
    state_label = {
        STATE_WAITING:  "Starting...",
        STATE_TO_ITEM:  "Heading to item",
        STATE_PICKING:  "Picking up item",
        STATE_TO_DEPOT: "Returning to depot",
        STATE_RESTING:  "Resting at depot",
        STATE_ALL_DONE: "Shift complete!",
    }

    def divider():
        nonlocal y
        pygame.draw.line(surface, (40, 50, 65), (x, y), (divider_end, y))
        y += 8

    for i in range(agent.n):
        agent_id  = i + 1
        divider()
        col_agent = agent_colour(agent_id)

        surface.blit(font_small.render(
            f"AGENT {agent_id}",
            True, col_agent), (x, y));  y += 16

        status = state_label.get(int(agent.state[i]), "unknown")
        col    = COL_DONE if agent.is_done(i) else COL_WHITE
        surface.blit(font_med.render(status, True, col), (x, y));  y += 20

        surface.blit(font_small.render(
            f"Done: {agent.orders_completed[i]}  "
            f"Arrived: {agent.total_orders}  Dist: {agent.distance[i]:.1f}m",
            True, COL_MUTED), (x, y));  y += 16

        if agent.state[i] == STATE_PICKING:
            surface.blit(font_small.render(
                f"Pickup ticks left: {agent.pickup_ticks[i]}",
                True, COL_ITEM_DOT), (x, y));  y += 14

        y += 4

    # Congestion
    divider()
    surface.blit(font_small.render("CONGESTION", True, COL_MUTED), (x, y));       y += 16
    conflict_col = COL_CONFLICT if metrics.cell_conflicts > 0 else COL_WHITE
    surface.blit(font_big.render(str(metrics.cell_conflicts), True, conflict_col), (x, y)); y += 30
    surface.blit(font_small.render("cell conflict steps", True, COL_MUTED), (x, y)); y += 20

    # Simulation complete summary
    if agent.all_done():
        divider()
        surface.blit(font_med.render("Simulation complete!", True, COL_DONE), (x, y)); y += 20
        for i in range(agent.n):
            surface.blit(font_small.render(
                f"A{i+1}: {agent.distance[i]:.1f}m | congestion: {agent._slow_ticks[i]} slow ticks",
                True, agent_colour(i + 1)), (x, y)); y += 14
        surface.blit(font_small.render(
            f"Conflicts: {metrics.cell_conflicts}",
            True, COL_CONFLICT), (x, y)); y += 14
        surface.blit(font_small.render("(close window to exit)", True, COL_MUTED), (x, y))

    # KPI Results panel (shown when launched from experiment mode)
    if kpi_results:
        divider()
        surface.blit(font_big.render("KPI RESULTS", True, COL_DEPOT_LABEL), (x, y)); y += 24
        surface.blit(font_med.render(f"(avg over {kpi_results['num_runs']} runs)", True, COL_MUTED), (x, y)); y += 20
        kpi_col = (180, 220, 255)
        surface.blit(font_med.render(
            f"Picks/hr        : {kpi_results['picks_per_hour']:.2f}",
            True, kpi_col), (x, y)); y += 20
        surface.blit(font_med.render(
            f"Dist/agent      : {kpi_results['dist_per_agent']:.1f} cells",
            True, kpi_col), (x, y)); y += 20
        surface.blit(font_med.render(
            f"Congestion rate : {kpi_results['congestion_rate']:.2f}%",
            True, kpi_col), (x, y)); y += 20
        if "storage_utilisation" in kpi_results:
            surface.blit(font_med.render(
                f"Storage util    : {kpi_results['storage_utilisation']:.1f}%"
                f"  ({kpi_results['num_items']} items)",
                True, kpi_col), (x, y)); y += 20

    legend_y = surface.get_height() - 100
    if legend_y > y + 10:
        pygame.draw.line(surface, (40, 50, 65), (x, legend_y), (divider_end, legend_y))
        legend_y += 8
        surface.blit(font_small.render("LEGEND", True, COL_MUTED), (x, legend_y)); legend_y += 16
        for colour, text in [
            (COL_EMPTY,    "Floor"),
            (COL_SHELF,    "Shelf"),
            (COL_ITEM_DOT, "Item on shelf"),
            (COL_DEPOT,    "Depot"),
            (COL_CONFLICT, "Conflict cell"),
        ]:
            sq = pygame.Rect(x, legend_y + 2, 10, 10)
            pygame.draw.rect(surface, colour, sq, border_radius=2)
            surface.blit(font_small.render(text, True, COL_MUTED), (x + 16, legend_y))
            legend_y += 14


# resolve_conflicts() removed — JuPedSim handles collision avoidance physically.
# Agents now slow down and steer around each other using the Collision Free
# Speed Model instead of the reactive cell-blocking approach from Sim_V2.


# =============================================================================
# TIMELINE BAR
# =============================================================================

def _draw_timeline(surface, current_tick, total_ticks, step_delay, playing,
                   grid_px_w, win_h, font_small):
    """
    Horizontal scrubber bar drawn at the very bottom of the grid area.
    Click anywhere on the bar to jump to that point in the shift.
    """
    PAD   = MARGIN + 4
    bar_y = win_h - 16
    bar_x0, bar_x1 = PAD, grid_px_w - PAD
    bar_w = bar_x1 - bar_x0
    bar_h = 6

    # Track background
    pygame.draw.rect(surface, (25, 33, 46),
                     (bar_x0, bar_y, bar_w, bar_h), border_radius=3)

    # Filled (elapsed) portion
    filled = int(bar_w * current_tick / max(total_ticks - 1, 1))
    if filled > 0:
        pygame.draw.rect(surface, (55, 110, 220),
                         (bar_x0, bar_y, filled, bar_h), border_radius=3)

    # Playhead knob
    pygame.draw.circle(surface, (180, 210, 255),
                       (bar_x0 + filled, bar_y + bar_h // 2), 5)

    # Status + key-binding hint
    hrs  = current_tick // 3600
    mins = (current_tick % 3600) // 60
    secs = current_tick % 60
    # Express step_delay as a playback multiplier (real-time = 1×)
    mult = round(1.0 / max(step_delay, 0.001))
    hint = font_small.render(
        f"{'▶' if playing else '⏸'}  {hrs}h {mins:02d}m {secs:02d}s  "
        f"(tick {current_tick}/{total_ticks})   {mult}×  │  "
        f"SPACE pause  ·  ←/→ step  ·  SHIFT+←/→ jump 10 min  ·  "
        f"↑/↓ speed  ·  HOME/END  ·  click bar to scrub",
        True, (70, 90, 120))
    surface.blit(hint, (PAD, bar_y - 15))


# =============================================================================
# VISUAL MODE  (two-phase: silent record → interactive replay)
# =============================================================================

def run_visual(args, demo_mode=False, kpi_results=None, replay_seed=None):
    """
    Dispatch to either the live simulation or the record-then-replay version
    depending on args.replay.

    Without --replay (default):
        Live simulation — opens the window immediately and renders each tick
        as JuPedSim computes it.  Fastest startup, no scrubbing.

    With --replay:
        Phase 1 records the full shift silently (~10-30s of upfront cost),
        then Phase 2 opens an interactive replay window with pause / rewind /
        scrub / speed controls.  Use this when you want to inspect specific
        moments without watching the whole shift.
    """
    if getattr(args, "replay", False):
        return run_visual_replay(args, demo_mode, kpi_results, replay_seed)
    else:
        return run_visual_live(args, demo_mode, kpi_results, replay_seed)


def run_visual_replay(args, demo_mode=False, kpi_results=None, replay_seed=None):
    """
    Phase 1 — run the full simulation silently and record every tick's state.
    Phase 2 — open Pygame and let the user replay, rewind, and scrub freely.

    Keyboard / mouse controls (shown in timeline bar):
      SPACE              pause / resume
      ← / →              step back / forward 1 tick
      SHIFT + ← / →      jump back / forward 600 ticks (10 minutes)
      HOME / END         jump to first / last tick
      ↑ / ↓              double / halve playback speed
      click timeline     scrub to any point in the shift
    """
    from rl_env         import compute_depot_cols as _compute_depot_cols
    from jupedsim_agent import CELL_SIZE as JPS_CELL   # 1.0 m per grid cell
    import numpy as _np

    global CELL_SIZE
    CELL_SIZE = args.cell_size

    if replay_seed is not None:
        random.seed(replay_seed)

    if args.depot_count is not None:
        _depot_cols = _compute_depot_cols(args.depot_count, args.cols)
    elif args.depot_col is not None:
        _depot_cols = [args.depot_col]
    else:
        _depot_cols = [args.cols // 2]

    grid = Grid(
        rows=args.rows, cols=args.cols,
        aisle_width=args.aisle_width,
        centre_aisle_width=args.centre_aisle,
        depot_row=args.depot_row,
        depot_cols=_depot_cols,
        shelf_start_row=args.shelf_start,
        shelf_end_row=args.shelf_end,
        cross_aisle_row=args.cross_aisle_row,
        cross_aisle_width=args.cross_aisle_width,
        cross_aisle_enabled=not args.no_cross_aisle,
        replenish_delay=args.replenish_delay,
    )
    agent = JupedSimAgent(args.agents, grid)

    # ── Phase 1: record all frames (no display, runs as fast as possible) ────
    print("=" * 55)
    if demo_mode and replay_seed is not None:
        mode_label = "REPRESENTATIVE RUN (seed={}, closest to {}-run avg)".format(
            replay_seed, kpi_results.get("num_runs", "?") if kpi_results else "?")
    elif demo_mode:
        mode_label = "DEMO RUN (after experiment)"
    else:
        mode_label = "VISUAL MODE"
    print("  Warehouse Simulator  [{}]".format(mode_label))
    print("=" * 55)
    print(f"  Grid       : {grid.rows}×{grid.cols}  |  Agents: {agent.n}")
    print(f"  Recording {SHIFT_TICKS} ticks silently…")

    # Each tick stores a list of per-agent tuples:
    #   (x, y, state, target_row, target_col, orders_done, dist, pickup_ticks_left)
    all_frames = []
    for tick in range(SHIFT_TICKS):
        if nhpp_arrival(tick):
            agent.add_job()
        agent.step()
        grid.tick_replenishment()

        frame = []
        for i in range(agent.n):
            x, y = agent.get_xy(i)
            tgt  = agent.targets[i]
            tr, tc = tgt if tgt is not None else (-1, -1)
            frame.append((x, y,
                          int(agent.state[i]),
                          int(tr), int(tc),
                          int(agent.orders_completed[i]),
                          float(agent.distance[i]),
                          int(agent.pickup_ticks[i])))
        all_frames.append(frame)

        if (tick + 1) % 5760 == 0:
            pct = (tick + 1) * 100 // SHIFT_TICKS
            print(f"  {pct:3d}%  ({tick + 1}/{SHIFT_TICKS} ticks)")

    _total_orders = agent.total_orders
    _final_slow   = list(agent._slow_ticks)
    print("  Recording complete — opening interactive replay.")
    print("=" * 55)

    # ── Phase 2: interactive Pygame replay ───────────────────────────────────
    pygame.init()

    # Playback state — FrameProxy reads these via closure
    current_tick   = 0
    playing        = True
    # step_delay: seconds of wall-clock time between displayed ticks.
    # Default args.speed (0.1 s) → about 10× real-time.
    step_delay     = max(args.speed, 0.005)
    last_step_time = time.time()
    n_ag           = agent.n

    # FrameProxy — mimics the JupedSimAgent interface so draw_* functions
    # work unchanged.  All reads reflect the currently displayed tick because
    # current_tick is captured by closure (Python reads it at call time).
    class FrameProxy:
        n            = n_ag
        total_orders = _total_orders

        def _d(self, i):
            return all_frames[current_tick][i]

        def get_xy(self, i):
            d = self._d(i); return d[0], d[1]

        def is_done(self, i):
            return self._d(i)[2] == STATE_ALL_DONE

        def all_done(self):
            return all(self._d(i)[2] == STATE_ALL_DONE
                       for i in range(self.n))

        def get_speed(self, i):
            return 0.0

        @property
        def state(self):
            return [self._d(i)[2] for i in range(self.n)]

        @property
        def targets(self):
            return [(self._d(i)[3], self._d(i)[4])
                    if self._d(i)[3] >= 0 else None
                    for i in range(self.n)]

        @property
        def orders_completed(self):
            return _np.array([self._d(i)[5] for i in range(self.n)],
                             dtype=_np.int32)

        @property
        def distance(self):
            return [self._d(i)[6] for i in range(self.n)]

        @property
        def pickup_ticks(self):
            return [self._d(i)[7] for i in range(self.n)]

        @property
        def pos_row(self):
            return [max(0, min(grid.rows - 1,
                               int(self._d(i)[1] / JPS_CELL)))
                    for i in range(self.n)]

        @property
        def pos_col(self):
            return [max(0, min(grid.cols - 1,
                               int(self._d(i)[0] / JPS_CELL)))
                    for i in range(self.n)]

        @property
        def _slow_ticks(self):
            return _final_slow

    proxy   = FrameProxy()
    metrics = MetricsTracker()

    grid_px_w = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    grid_px_h = grid.rows * (CELL_SIZE + MARGIN) + MARGIN
    TL_STRIP  = 36          # pixels reserved at the bottom for the timeline bar
    win_w     = grid_px_w + PANEL_WIDTH
    win_h     = max(grid_px_h + TL_STRIP, 400 + agent.n * 90)

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        ("DEMO | " if demo_mode else "REPLAY | ") +
        f"Warehouse Sim — {grid.rows}×{grid.cols} — {agent.n} agents"
    )

    font_big   = pygame.font.SysFont("monospace", 18, bold=True)
    font_med   = pygame.font.SysFont("monospace", 14)
    font_small = pygame.font.SysFont("monospace", 11)
    font_grid  = pygame.font.SysFont("monospace", max(7, CELL_SIZE // 4))
    clock      = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                jump = 600 if (mods & pygame.KMOD_SHIFT) else 1

                if event.key == pygame.K_SPACE:
                    playing = not playing

                elif event.key == pygame.K_LEFT:
                    current_tick = max(0, current_tick - jump)
                    playing = False

                elif event.key == pygame.K_RIGHT:
                    current_tick = min(SHIFT_TICKS - 1, current_tick + jump)
                    playing = False

                elif event.key == pygame.K_HOME:
                    current_tick = 0

                elif event.key == pygame.K_END:
                    current_tick = SHIFT_TICKS - 1

                elif event.key in (pygame.K_UP,
                                   pygame.K_EQUALS, pygame.K_PLUS,
                                   pygame.K_KP_PLUS):
                    step_delay = max(step_delay / 2.0, 0.001)

                elif event.key in (pygame.K_DOWN,
                                   pygame.K_MINUS,
                                   pygame.K_KP_MINUS):
                    step_delay = min(step_delay * 2.0, 2.0)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my  = event.pos
                tl_x0   = MARGIN + 4
                tl_x1   = grid_px_w - MARGIN - 4
                tl_y    = win_h - 16
                # Click within ±12 px of the timeline bar
                if tl_x0 <= mx <= tl_x1 and abs(my - tl_y) <= 12:
                    frac = (mx - tl_x0) / max(tl_x1 - tl_x0, 1)
                    current_tick = int(frac * (SHIFT_TICKS - 1))
                    current_tick = max(0, min(SHIFT_TICKS - 1, current_tick))

        # Advance playback
        now = time.time()
        if playing and (now - last_step_time) >= step_delay:
            current_tick += 1
            if current_tick >= SHIFT_TICKS:
                current_tick = SHIFT_TICKS - 1
                playing = False
            last_step_time = now

        # Draw
        screen.fill(COL_BG)
        draw_grid(screen, grid, proxy, metrics, font_grid)
        draw_panel(screen, grid, proxy, metrics,
                   font_big, font_med, font_small,
                   kpi_results=kpi_results, sim_tick=current_tick)
        _draw_timeline(screen, current_tick, SHIFT_TICKS,
                       step_delay, playing, grid_px_w, win_h, font_small)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


def run_visual_live(args, demo_mode=False, kpi_results=None, replay_seed=None):
    """
    Original live-simulation visualiser — opens the window immediately and
    renders each tick as JuPedSim computes it.  No recording, no scrubbing.
    Use this for the fastest startup when you just want to watch the run.
    """
    from rl_env import compute_depot_cols as _compute_depot_cols
    global CELL_SIZE
    CELL_SIZE = args.cell_size

    if replay_seed is not None:
        random.seed(replay_seed)

    if args.depot_count is not None:
        _depot_cols = _compute_depot_cols(args.depot_count, args.cols)
    elif args.depot_col is not None:
        _depot_cols = [args.depot_col]
    else:
        _depot_cols = [args.cols // 2]

    pygame.init()

    grid = Grid(
        rows=args.rows, cols=args.cols,
        aisle_width=args.aisle_width,
        centre_aisle_width=args.centre_aisle,
        depot_row=args.depot_row,
        depot_cols=_depot_cols,
        shelf_start_row=args.shelf_start,
        shelf_end_row=args.shelf_end,
        cross_aisle_row=args.cross_aisle_row,
        cross_aisle_width=args.cross_aisle_width,
        cross_aisle_enabled=not args.no_cross_aisle,
        replenish_delay=args.replenish_delay,
    )
    agent = JupedSimAgent(args.agents, grid)

    print("=" * 55)
    if demo_mode and replay_seed is not None:
        mode_label = "REPRESENTATIVE RUN (seed={}, closest to {}-run avg)".format(
            replay_seed, kpi_results.get("num_runs", "?") if kpi_results else "?")
    elif demo_mode:
        mode_label = "DEMO RUN (after experiment)"
    else:
        mode_label = "LIVE MODE"
    print(f"  Warehouse Simulator  [{mode_label}]")
    print("=" * 55)
    print(f"  Grid       : {grid.rows}×{grid.cols}  |  Agents: {agent.n}")
    print(f"  Step delay : {args.speed}s   (pass --replay for scrub controls)")
    print("=" * 55)

    grid_px_w = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    grid_px_h = grid.rows * (CELL_SIZE + MARGIN) + MARGIN
    win_w     = grid_px_w + PANEL_WIDTH
    win_h     = max(grid_px_h, 400 + agent.n * 90)

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        ("DEMO | " if demo_mode else "LIVE | ") +
        f"Warehouse Sim — {grid.rows}×{grid.cols} — {agent.n} agents"
    )

    font_big   = pygame.font.SysFont("monospace", 18, bold=True)
    font_med   = pygame.font.SysFont("monospace", 14)
    font_small = pygame.font.SysFont("monospace", 11)
    font_grid  = pygame.font.SysFont("monospace", max(7, CELL_SIZE // 4))
    clock      = pygame.time.Clock()

    metrics        = MetricsTracker()
    last_step_time = time.time()
    sim_tick       = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now      = time.time()
        sim_done = (sim_tick >= SHIFT_TICKS)

        if not sim_done and (now - last_step_time) >= args.speed:
            if nhpp_arrival(sim_tick):
                agent.add_job()
            agent.step()
            grid.tick_replenishment()
            last_step_time = now
            sim_tick      += 1

            if sim_tick >= SHIFT_TICKS:
                metrics.print_summary(agent)

        metrics.tick_flash()

        screen.fill(COL_BG)
        draw_grid(screen, grid, agent, metrics, font_grid)
        draw_panel(screen, grid, agent, metrics,
                   font_big, font_med, font_small,
                   kpi_results=kpi_results, sim_tick=sim_tick)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


def main():
    args = parse_args()

    # Apply CLI overrides to NHPP parameters before any simulation runs.
    # nhpp_arrival() reads LAMBDA_BASE/DEMAND_BETA from the agent module at
    # call time, so patching the module variables here is sufficient.
    if args.lambda_base is not None:
        _agent_module.LAMBDA_BASE = args.lambda_base
    if args.beta is not None:
        _agent_module.DEMAND_BETA = args.beta

    # Re-import the (now possibly patched) values so print statements are correct
    global LAMBDA_BASE, DEMAND_BETA
    LAMBDA_BASE = _agent_module.LAMBDA_BASE
    DEMAND_BETA = _agent_module.DEMAND_BETA

    if args.plot_layout:
        from rl_env import compute_depot_cols as _compute_depot_cols
        from plot_heatmap import plot_layout
        _depot_cols = (_compute_depot_cols(args.depot_count, args.cols)
                       if args.depot_count is not None
                       else ([args.depot_col] if args.depot_col is not None
                             else [args.cols // 2]))
        plot_layout(
            grid_rows=args.rows, grid_cols=args.cols,
            out_file="layout.png",
            aisle_width=args.aisle_width,
            centre_aisle_width=args.centre_aisle,
            depot_row=args.depot_row,
            depot_cols=_depot_cols,
            shelf_start_row=args.shelf_start,
            shelf_end_row=args.shelf_end,
            cross_aisle_row=args.cross_aisle_row,
            cross_aisle_width=args.cross_aisle_width,
            cross_aisle_enabled=not args.no_cross_aisle,
        )
        return   # layout-only mode: no simulation needed

    if args.experiment:
        run_experiment(args)
    else:
        run_visual(args)


if __name__ == "__main__":
    main()
