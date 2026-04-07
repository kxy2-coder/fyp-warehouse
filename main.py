# =============================================================================
# main.py — Run the Simulation
# =============================================================================
# Runs an N-agent warehouse simulation.
# Each agent completes a randomly drawn number of pickup orders per run.
# Human factors (fatigue, experience) are modeled per Malpas & Relvas (2025).
# Congestion is measured as "cell conflicts": simulation steps where two or
# more agents occupy the same cell simultaneously.
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
import random
import pygame
import sys
import time

from grid    import Grid, EMPTY, SHELF, ITEM, DEPOT
from agent   import (Agent, nhpp_arrival, SHIFT_TICKS, LAMBDA_BASE,
                     DEMAND_BETA, NUM_RUNS, BLOCKED_WAIT_TICKS,
                     STATE_WAITING, STATE_TO_ITEM, STATE_PICKING,
                     STATE_TO_DEPOT, STATE_RESTING, STATE_ALL_DONE)
from metrics import MetricsTracker, TraceLogger


# =============================================================================
# CLI ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="N-agent warehouse simulator")
    parser.add_argument("--agents",        type=int,   default=4,    help="Number of agents (default 4)")
    parser.add_argument("--rows",          type=int,   default=25,   help="Grid rows (default 25)")
    parser.add_argument("--cols",          type=int,   default=35,   help="Grid cols (default 35, odd recommended)")
    parser.add_argument("--speed",         type=float, default=0.15, help="Seconds between steps (default 0.15)")
    parser.add_argument("--aisle-width",   type=int,   default=2,    help="Aisle width between shelf blocks (default 2)")
    parser.add_argument("--centre-aisle",  type=int,   default=3,    help="Centre aisle width, must be odd (default 3)")
    parser.add_argument("--depot-row",     type=int,   default=None, help="Depot row (default 0 = top row)")
    parser.add_argument("--depot-col",     type=int,   default=None, help="Depot column (default = centre)")
    parser.add_argument("--shelf-start",     type=int,   default=None, help="First row shelves appear (default 1)")
    parser.add_argument("--shelf-end",       type=int,   default=None, help="Last row shelves appear (default rows-2)")
    parser.add_argument("--cross-aisle-row", type=int,   default=None, help="Row of horizontal cross-aisle (default = middle of shelf zone)")
    parser.add_argument("--cell-size",     type=int,   default=35,   help="Pixel size per cell (default 35)")
    parser.add_argument("--experiment",      action="store_true",      help="Silent multi-run averaging mode (no Pygame)")
    parser.add_argument("--runs",            type=int,   default=None, help="Override NUM_RUNS for --experiment mode")
    parser.add_argument("--replenish-delay", type=int,   default=100,  help="Ticks before a picked shelf restocks (default 100)")
    parser.add_argument("--lambda-base",    type=float, default=None, help=f"NHPP base arrival rate per tick (default {LAMBDA_BASE})")
    parser.add_argument("--beta",           type=float, default=None, help=f"NHPP amplitude parameter (default {DEMAND_BETA})")
    parser.add_argument("--plot-workload",  action="store_true",      help="Save workload analysis plot after --experiment run")
    parser.add_argument("--plot-traces",   action="store_true",      help="Save per-agent trace plot after --experiment run")
    parser.add_argument("--plot-heatmap",  action="store_true",      help="Save conflict heatmap after --experiment run")
    parser.add_argument("--bin",            type=int,   default=600,  help="Bin size (ticks) for workload plot (default 600 = 10 min)")
    return parser.parse_args()



# =============================================================================
# EXPERIMENT MODE - silent multi-run averaging
# =============================================================================

def run_experiment(args):
    """
    Run NUM_RUNS silent simulations, compute and print the 3 KPIs,
    then open the Pygame visual replaying the most representative run.
    """
    num_runs = args.runs if args.runs is not None else NUM_RUNS

    print("=" * 60)
    print("  EXPERIMENT MODE")
    print("  Runs: {}   Agents: {}   Grid: {}x{}".format(num_runs, args.agents, args.rows, args.cols))
    print("  NHPP demand: λ_base={}, β={}, shift={} ticks (8 hr)".format(LAMBDA_BASE, DEMAND_BETA, SHIFT_TICKS))
    print("=" * 60)

    accumulated  = {}
    run_records  = []

    # Workload tracking (only populated when --plot-workload is set)
    all_arrivals    = []
    all_completions = []

    # Trace accumulator: dict keyed by (tick, agent_id) → list of snapshots
    trace_accum = {}

    # Spatial congestion accumulator: (row, col) → total conflict count across runs
    spatial_accum = {}

    for run_i in range(1, num_runs + 1):
        seed = run_i * 7919
        random.seed(seed)

        grid = Grid(
            rows=args.rows, cols=args.cols,
            aisle_width=args.aisle_width,
            centre_aisle_width=args.centre_aisle,
            depot_row=args.depot_row,
            depot_col=args.depot_col,
            shelf_start_row=args.shelf_start,
            shelf_end_row=args.shelf_end,
            cross_aisle_row=args.cross_aisle_row,
            replenish_delay=args.replenish_delay,
        )
        # ONE Agent object manages all args.agents workers simultaneously
        agent        = Agent(args.agents, grid, quota=0)
        metrics      = MetricsTracker()
        trace_logger = TraceLogger(args.agents, bin_size=600)

        if args.plot_workload:
            import numpy as np
            run_arrivals    = np.zeros(SHIFT_TICKS, dtype=np.int32)
            run_completions = np.zeros(SHIFT_TICKS, dtype=np.int32)
            prev_completed  = 0

        for tick in range(SHIFT_TICKS):
            if nhpp_arrival(tick):
                agent.add_job()
                if args.plot_workload:
                    run_arrivals[tick] = 1
            agent.step()
            resolve_conflicts(agent)
            grid.tick_replenishment()
            metrics.update(agent, grid.depot, tick)
            if (tick + 1) % 600 == 0:
                trace_logger.snapshot(tick + 1, agent)
            if args.plot_workload:
                cur = int(agent.orders_completed.sum())
                run_completions[tick] = cur - prev_completed
                prev_completed = cur

        if args.plot_workload:
            all_arrivals.append(run_arrivals)
            all_completions.append(run_completions)

        for rec in trace_logger.records:
            key = (rec["tick"], rec["agent_id"])
            trace_accum.setdefault(key, []).append(rec)

        for _, row, col in metrics.spatial_log:
            spatial_accum[(row, col)] = spatial_accum.get((row, col), 0) + 1

        raw = metrics.collect_raw(agent)
        for k, v in raw.items():
            accumulated[k] = accumulated.get(k, 0.0) + v

        completed_ = raw["jobs_completed"]
        run_picks  = completed_ / 8.0
        run_dist   = raw["total_distance"] / args.agents
        run_cong   = (raw["cell_conflicts"] * BLOCKED_WAIT_TICKS * 100) / SHIFT_TICKS
        run_records.append((seed, run_picks, run_dist, run_cong))

        print("  Run {:3d}/{} | arrived={:.0f} | completed={:.0f} | dist={:.0f} | conflicts={:.0f}".format(
            run_i, num_runs,
            raw["jobs_arrived"], raw["jobs_completed"],
            raw["total_distance"], raw["cell_conflicts"]))

    avg = {k: v / num_runs for k, v in accumulated.items()}

    picks_per_hour  = avg["jobs_completed"] / 8.0
    dist_per_agent  = avg["total_distance"] / args.agents
    congestion_rate = (avg["cell_conflicts"] * BLOCKED_WAIT_TICKS * 100 ) / SHIFT_TICKS

    all_picks = [r[1] for r in run_records]
    all_dist  = [r[2] for r in run_records]
    all_cong  = [r[3] for r in run_records]

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
    print("  {:<42s} {:>10.1f}".format("Avg Travel Distance per Agent (cells)",    dist_per_agent))
    print("  {:<42s} {:>9.2f}%".format("Congestion Rate  (% shift time affected)", congestion_rate))
    print()
    print("  RAW AVERAGES")
    print("  " + "-" * (W - 2))
    print("  {:<42s} {:>10.1f}".format("Avg jobs arrived (NHPP)",      avg["jobs_arrived"]))
    print("  {:<42s} {:>10.1f}".format("Avg jobs completed",           avg["jobs_completed"]))
    print("  {:<42s} {:>10.1f}".format("Avg total distance (cells)",   avg["total_distance"]))
    print("  {:<42s} {:>10.4f}".format("Avg total work time (hrs)",    avg["total_work_time"]))
    print("  {:<42s} {:>10.3f}".format("Avg final fatigue",            avg["avg_final_fatigue"]))
    print("  {:<42s} {:>10.1f}".format("Avg cell conflicts",           avg["cell_conflicts"]))
    print("=" * W)
    print()
    print("  Replaying most representative run (seed={}) — close window to exit.".format(best_seed))
    print("=" * W)
    print()

    if args.plot_workload:
        from plot_workload import plot_from_data
        plot_from_data(all_arrivals, all_completions,
                       bin_size=args.bin, n_agents=args.agents)

    # Save averaged agent traces to CSV
    trace_csv = "agent_traces.csv"
    avg_trace_records = []
    for (tick, agent_id), recs in sorted(trace_accum.items()):
        n = len(recs)
        avg_trace_records.append({
            "tick":           tick,
            "agent_id":       agent_id,
            "picks":          round(sum(r["picks"]          for r in recs) / n, 2),
            "distance":       round(sum(r["distance"]       for r in recs) / n, 2),
            "idle_ticks":     round(sum(r["idle_ticks"]     for r in recs) / n, 2),
            "blocked_events": round(sum(r["blocked_events"] for r in recs) / n, 2),
            "fatigue":        round(sum(r["fatigue"]        for r in recs) / n, 4),
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
        plot_from_traces(trace_csv)

    # Save averaged spatial conflict heatmap CSV
    heatmap_csv = "conflict_heatmap.csv"
    if spatial_accum:
        import csv as _csv2
        with open(heatmap_csv, "w", newline="") as f:
            writer = _csv2.writer(f)
            writer.writerow(["row", "col", "avg_count"])
            for (row, col), total in sorted(spatial_accum.items()):
                writer.writerow([row, col, round(total / num_runs, 4)])
        print(f"  Conflict heatmap CSV saved → {heatmap_csv}")

    if args.plot_heatmap:
        from plot_heatmap import plot_from_csv
        plot_from_csv(heatmap_csv, grid_rows=args.rows, grid_cols=args.cols)

    run_visual(args, demo_mode=True, kpi_results={
        "num_runs":        num_runs,
        "picks_per_hour":  picks_per_hour,
        "dist_per_agent":  dist_per_agent,
        "congestion_rate": congestion_rate,
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

TARGET_COLOURS = [
    (220, 100,  50),
    (180,  60, 200),
    ( 60, 200,  80),
    ( 60, 180, 220),
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
    return TARGET_COLOURS[(agent_id - 1) % len(TARGET_COLOURS)]

def path_colour(agent_id):
    return PATH_COLOURS[(agent_id - 1) % len(PATH_COLOURS)]


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def cell_rect(row, col):
    x = col * (CELL_SIZE + MARGIN) + MARGIN
    y = row * (CELL_SIZE + MARGIN) + MARGIN
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def fatigue_colour(fatigue):
    """Interpolate colour from green -> yellow -> red based on fatigue 0-1."""
    if fatigue < 0.5:
        t = fatigue / 0.5
        r = int(COL_FATIGUE_LOW[0] + t * (COL_FATIGUE_MED[0] - COL_FATIGUE_LOW[0]))
        g = int(COL_FATIGUE_LOW[1] + t * (COL_FATIGUE_MED[1] - COL_FATIGUE_LOW[1]))
        b = int(COL_FATIGUE_LOW[2] + t * (COL_FATIGUE_MED[2] - COL_FATIGUE_LOW[2]))
    else:
        t = (fatigue - 0.5) / 0.5
        r = int(COL_FATIGUE_MED[0] + t * (COL_FATIGUE_HIGH[0] - COL_FATIGUE_MED[0]))
        g = int(COL_FATIGUE_MED[1] + t * (COL_FATIGUE_HIGH[1] - COL_FATIGUE_MED[1]))
        b = int(COL_FATIGUE_MED[2] + t * (COL_FATIGUE_HIGH[2] - COL_FATIGUE_MED[2]))
    return (r, g, b)


def draw_fatigue_bar(surface, x, y, width, fatigue, font_small):
    """Draw a labelled fatigue progress bar."""
    bar_h    = 10
    filled_w = int(width * fatigue)
    col      = fatigue_colour(fatigue)

    pygame.draw.rect(surface, (40, 48, 60), pygame.Rect(x, y, width, bar_h), border_radius=3)
    if filled_w > 0:
        pygame.draw.rect(surface, col, pygame.Rect(x, y, filled_w, bar_h), border_radius=3)
    pygame.draw.rect(surface, (70, 82, 100), pygame.Rect(x, y, width, bar_h), width=1, border_radius=3)
    label = font_small.render(f"Fatigue: {fatigue:.0%}", True, col)
    surface.blit(label, (x, y + bar_h + 2))
    return y + bar_h + 16


def draw_grid(surface, grid, agent, metrics, font_small):
    """Draw the warehouse grid, agents, paths, targets, conflict flash."""
    # Upcoming path cells for each agent (used to draw path dots).
    upcoming = [
        set(agent.paths[i][agent.path_indices[i]:])
        for i in range(agent.n)
    ]

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

    # Draw agents on top
    for i in range(agent.n):
        if not agent.is_done(i):
            r        = cell_rect(int(agent.pos_row[i]), int(agent.pos_col[i]))
            col_fill = agent_colour(i + 1)
            if agent.state[i] == STATE_RESTING:
                pygame.draw.circle(surface, COL_RESTING, r.center,
                                   CELL_SIZE // 2 - 2, width=3)
            elif agent.state[i] == STATE_PICKING:
                pygame.draw.circle(surface, fatigue_colour(float(agent.fatigue[i])),
                                   r.center, CELL_SIZE // 2 - 2, width=3)
            pygame.draw.circle(surface, col_fill, r.center, CELL_SIZE // 2 - 4)
            pygame.draw.circle(surface, COL_WHITE, r.center, CELL_SIZE // 8)


def draw_panel(surface, grid, agent, metrics, font_big, font_med, font_small, kpi_results=None):
    """Right-side info panel with agent status, fatigue bars, KPI results."""
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
    y += 22

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
        exp_label = "Experienced" if agent.experience_B[i] >= 100 else "Novice"

        surface.blit(font_small.render(
            f"AGENT {agent_id}  [{exp_label}]",
            True, col_agent), (x, y));  y += 16

        status = state_label.get(int(agent.state[i]), "unknown")
        col    = COL_DONE if agent.is_done(i) else COL_WHITE
        surface.blit(font_med.render(status, True, col), (x, y));  y += 20

        surface.blit(font_small.render(
            f"Done: {agent.orders_completed[i]}  "
            f"Arrived: {agent.total_orders}  Dist: {agent.distance[i]}",
            True, COL_MUTED), (x, y));  y += 16

        if agent.state[i] == STATE_PICKING:
            surface.blit(font_small.render(
                f"Pickup ticks left: {agent.pickup_ticks[i]}",
                True, COL_ITEM_DOT), (x, y));  y += 14

        y = draw_fatigue_bar(surface, x, y, bar_width, float(agent.fatigue[i]), font_small)
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
                f"A{i+1}: {agent.distance[i]} cells | fat={agent.fatigue[i]:.2f}",
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


# =============================================================================
# CONFLICT RESOLUTION
# =============================================================================

def resolve_conflicts(agent):
    """
    Detect agents that physically share the same non-depot cell after moving,
    and block the lower-priority one for BLOCKED_WAIT_TICKS ticks.

    This models human workers who react when they actually meet in an aisle —
    one steps aside and waits for the other to pass.  No look-ahead or path
    knowledge is used, matching how a real human would behave.

    Priority (highest first):
      1. State: to_depot (carrying item, returning) beats to_item (going to pick)
      2. Tiebreaker: lower agent index beats higher

    Called AFTER agent.step() so positions reflect where agents actually moved.
    """
    active = [i for i in range(agent.n) if not agent.is_done(i)]

    def priority(i):
        state_score = 1 if agent.state[i] == STATE_TO_DEPOT else 0
        return (state_score, random.random())

    # Build a map of cell → list of agents currently on it
    pos_map = {}
    for i in active:
        pos = (int(agent.pos_row[i]), int(agent.pos_col[i]))
        pos_map.setdefault(pos, []).append(i)

    for pos, occupants in pos_map.items():
        if len(occupants) < 2:
            continue
        if pos == tuple(agent.grid.depot):
            continue   # depot is a shared waiting area — no conflict

        # Highest priority agent passes through; all others wait
        winner = max(occupants, key=priority)
        for loser in occupants:
            if loser == winner:
                continue
            agent.blocked_ticks[loser]  = BLOCKED_WAIT_TICKS
            agent.blocked_events[loser] += 1


# =============================================================================
# VISUAL MODE + ENTRY POINT
# =============================================================================

def run_visual(args, demo_mode=False, kpi_results=None, replay_seed=None):
    """Run one simulation with the Pygame window."""
    global CELL_SIZE
    CELL_SIZE = args.cell_size

    if replay_seed is not None:
        random.seed(replay_seed)

    pygame.init()

    grid = Grid(
        rows=args.rows, cols=args.cols,
        aisle_width=args.aisle_width,
        centre_aisle_width=args.centre_aisle,
        depot_row=args.depot_row,
        depot_col=args.depot_col,
        shelf_start_row=args.shelf_start,
        shelf_end_row=args.shelf_end,
        cross_aisle_row=args.cross_aisle_row,
        replenish_delay=args.replenish_delay,
    )

    # ONE Agent object manages all args.agents workers simultaneously.
    agent = Agent(args.agents, grid, quota=0)

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
    print(f"  Grid         : {grid.rows} rows x {grid.cols} cols")
    print(f"  Aisle width  : {grid.aisle_width}  |  Centre aisle: {grid.centre_aisle_width}")
    print(f"  Depot        : row {grid.depot[0]}, col {grid.depot[1]}")
    print(f"  Shelf zone   : rows {grid.shelf_start_row} to {grid.shelf_end_row}")
    print(f"  NHPP demand  : λ_base={LAMBDA_BASE}, β={DEMAND_BETA}, shift={SHIFT_TICKS} ticks")
    print(f"  Agents       : {agent.n}")
    for i in range(agent.n):
        exp_label = "Experienced" if agent.experience_B[i] >= 100 else "Novice"
        print(f"    Agent {i+1} : {exp_label} (B={agent.experience_B[i]:.0f}h)")
    print(f"  Step delay   : {args.speed}s")
    print("=" * 55)

    grid_px_w = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    grid_px_h = grid.rows * (CELL_SIZE + MARGIN) + MARGIN
    win_w     = grid_px_w + PANEL_WIDTH
    win_h     = max(grid_px_h, 400 + agent.n * 90)

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        ("DEMO RUN | " if demo_mode else "") +
        f"Warehouse Sim — {grid.rows}x{grid.cols} — {agent.n} agents — NHPP λ={LAMBDA_BASE}"
    )

    font_big   = pygame.font.SysFont("monospace", 18, bold=True)
    font_med   = pygame.font.SysFont("monospace", 14)
    font_small = pygame.font.SysFont("monospace", 11)
    font_grid  = pygame.font.SysFont("monospace", max(7, CELL_SIZE // 4))

    clock          = pygame.time.Clock()
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
            resolve_conflicts(agent)
            grid.tick_replenishment()
            last_step_time = now
            sim_tick      += 1

            metrics.update(agent, grid.depot)

            if sim_tick >= SHIFT_TICKS:
                metrics.print_summary(agent)

        metrics.tick_flash()

        screen.fill(COL_BG)
        draw_grid(screen, grid, agent, metrics, font_grid)
        draw_panel(screen, grid, agent, metrics, font_big, font_med, font_small, kpi_results=kpi_results)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


def main():
    args = parse_args()
    if args.experiment:
        run_experiment(args)
    else:
        run_visual(args)


if __name__ == "__main__":
    main()