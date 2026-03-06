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
import pygame
import sys
import time
import itertools

from grid    import Grid, EMPTY, SHELF, ITEM, DEPOT
from agent   import Agent, draw_job_quota, JOBS_MEAN, JOBS_STD, NUM_RUNS, BLOCKED_WAIT_TICKS
from metrics import MetricsTracker


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
    parser.add_argument("--shelf-start",   type=int,   default=None, help="First row shelves appear (default 1)")
    parser.add_argument("--shelf-end",     type=int,   default=None, help="Last row shelves appear (default rows-2)")
    parser.add_argument("--cell-size",     type=int,   default=35,   help="Pixel size per cell (default 35)")
    parser.add_argument("--experiment",   action="store_true",      help="Silent multi-run averaging mode (no Pygame)")
    parser.add_argument("--runs",         type=int,   default=None, help="Override NUM_RUNS for --experiment mode")
    return parser.parse_args()



# =============================================================================
# EXPERIMENT MODE - silent multi-run averaging
# =============================================================================
# Runs NUM_RUNS simulations with no Pygame window.
# Each run draws a fresh job quota, builds a fresh grid, steps until all
# agents finish, collects raw numbers. After all runs, prints averaged results.
# KPIs are NOT calculated here - raw numbers only.
# Apply your KPI formulas to the averages once they are finalised.

def run_experiment(args):
    """
    Run NUM_RUNS silent simulations, compute and print the 3 KPIs,
    then automatically open the Pygame visual demo with the same settings.
    """
    num_runs = args.runs if args.runs is not None else NUM_RUNS

    print("=" * 60)
    print("  EXPERIMENT MODE")
    print("  Runs: {}   Agents: {}   Grid: {}x{}".format(num_runs, args.agents, args.rows, args.cols))
    print("  Job quota distribution: N(mean={}, std={})  [placeholder]".format(JOBS_MEAN, JOBS_STD))
    print("=" * 60)

    accumulated = {}

    for run_i in range(1, num_runs + 1):
        grid = Grid(
            rows=args.rows, cols=args.cols,
            aisle_width=args.aisle_width,
            centre_aisle_width=args.centre_aisle,
            depot_row=args.depot_row,
            depot_col=args.depot_col,
            shelf_start_row=args.shelf_start,
            shelf_end_row=args.shelf_end,
        )
        quota   = draw_job_quota()
        agents  = [Agent(grid, agent_id=i + 1, total_orders=quota) for i in range(args.agents)]
        metrics = MetricsTracker()

        while not all(a.is_done() for a in agents):
            resolve_right_of_way(agents)
            for agent in agents:
                agent.step()
            metrics.update(agents, grid.depot)

        raw = metrics.collect_raw(agents)
        for k, v in raw.items():
            accumulated[k] = accumulated.get(k, 0.0) + v

        print("  Run {:3d}/{} | quota={:3d} | orders={:.0f} | dist={:.0f} | conflicts={:.0f}".format(
            run_i, num_runs,
            raw["job_quota"], raw["total_orders"],
            raw["total_distance"], raw["cell_conflicts"]))

    # ── Compute averages ──────────────────────────────────────────────────────
    avg = {k: v / num_runs for k, v in accumulated.items()}

    # ── Compute the 3 KPIs ───────────────────────────────────────────────────
    picks_per_hour  = avg["total_orders"] / avg["total_work_time"] if avg["total_work_time"] > 0 else 0.0
    dist_per_agent  = avg["total_distance"] / args.agents
    congestion_rate = avg["cell_conflicts"] / avg["total_orders"] if avg["total_orders"] > 0 else 0.0

    # ── Print results ─────────────────────────────────────────────────────────
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
    print("  {:<42s} {:>10.4f}".format("Congestion Rate  (conflicts / pick)",      congestion_rate))
    print()
    print("  RAW AVERAGES")
    print("  " + "-" * (W - 2))
    print("  {:<42s} {:>10.1f}".format("Avg job quota drawn",          avg["job_quota"]))
    print("  {:<42s} {:>10.1f}".format("Avg total orders completed",   avg["total_orders"]))
    print("  {:<42s} {:>10.1f}".format("Avg total distance (cells)",   avg["total_distance"]))
    print("  {:<42s} {:>10.4f}".format("Avg total work time (hrs)",    avg["total_work_time"]))
    print("  {:<42s} {:>10.1f}".format("Avg total blocking events",    avg["total_blocked"]))
    print("  {:<42s} {:>10.3f}".format("Avg final fatigue",            avg["avg_final_fatigue"]))
    print("  {:<42s} {:>10.1f}".format("Avg cell conflicts",           avg["cell_conflicts"]))
    print("=" * W)
    print()
    print("  Opening visual demonstration run — close window to exit.")
    print("=" * W)
    print()

    # ── Launch visual demo with KPIs to show on panel ─────────────────────────
    run_visual(args, demo_mode=True, kpi_results={
        "num_runs":        num_runs,
        "picks_per_hour":  picks_per_hour,
        "dist_per_agent":  dist_per_agent,
        "congestion_rate": congestion_rate,
    })

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

CELL_SIZE   = 40
MARGIN      = 2
PANEL_WIDTH = 320

# Colours
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

# A palette of agent colours — cycles if more agents than colours
AGENT_COLOURS = [
    ( 60, 200,  80),   # green
    ( 60, 180, 220),   # blue
    (220, 100,  50),   # orange
    (180,  60, 200),   # purple
    (220, 220,  60),   # yellow
    (200,  80, 120),   # pink
    ( 80, 200, 180),   # teal
    (200, 140,  60),   # amber
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


def draw_grid(surface, grid, agents, metrics, font_small):
    """Draw the warehouse grid, agents, paths, targets, conflict flash."""
    upcoming = [set(a.path[a.path_index:]) for a in agents]

    for r in range(grid.rows):
        for c in range(grid.cols):
            rect = cell_rect(r, c)
            cell = grid.cells[r][c]
            pos  = (r, c)

            # Base colour
            if cell == DEPOT:
                colour = COL_DEPOT
            elif cell in (SHELF, ITEM):
                colour = COL_SHELF
            else:
                colour = COL_EMPTY

            pygame.draw.rect(surface, colour, rect, border_radius=3)

            # Item dot
            if cell == ITEM:
                dot_size = CELL_SIZE // 5
                dot_rect = pygame.Rect(
                    rect.centerx - dot_size // 2,
                    rect.centery - dot_size // 2,
                    dot_size, dot_size
                )
                pygame.draw.rect(surface, COL_ITEM_DOT, dot_rect, border_radius=2)

            # Shelf reference label
            if CELL_SIZE >= 18 and cell in (SHELF, ITEM):
                label = grid.shelf_labels.get((r, c), "")
                if label:
                    lbl_surf = font_small.render(label, True, COL_MUTED)
                    lbl_rect = lbl_surf.get_rect(center=rect.center)
                    if cell == ITEM:
                        lbl_rect.y = rect.y + 3
                    surface.blit(lbl_surf, lbl_rect)

            # Target rings and path dots for each agent
            dot_r = CELL_SIZE // 8
            cx, cy = rect.center
            for i, agent in enumerate(agents):
                if not agent.is_done() and pos == agent.target:
                    pygame.draw.rect(surface, target_colour(agent.agent_id), rect, width=2, border_radius=3)
                if pos in upcoming[i] and pos != agent.pos and not agent.is_done():
                    pygame.draw.circle(surface, path_colour(agent.agent_id), (cx, cy), dot_r)

            # Conflict flash
            if metrics.flash_timer > 0 and pos == metrics.conflict_cell:
                pygame.draw.rect(surface, COL_CONFLICT, rect, width=3, border_radius=3)

    # Draw agents on top
    for agent in agents:
        if not agent.is_done():
            r = cell_rect(*agent.pos)
            col_fill = agent_colour(agent.agent_id)
            if agent.state == "resting":
                pygame.draw.circle(surface, COL_RESTING, r.center,
                                   CELL_SIZE // 2 - 2, width=3)
            elif agent.state == "picking_up":
                pygame.draw.circle(surface, fatigue_colour(agent.fatigue),
                                   r.center, CELL_SIZE // 2 - 2, width=3)
            pygame.draw.circle(surface, col_fill, r.center, CELL_SIZE // 2 - 4)
            # Small white dot in centre
            pygame.draw.circle(surface, COL_WHITE, r.center, CELL_SIZE // 8)


def draw_panel(surface, grid, agents, metrics, font_big, font_med, font_small, kpi_results=None):
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
    surface.blit(font_small.render(f"{len(agents)} agents | jobs: N({JOBS_MEAN}, {JOBS_STD})", True, COL_MUTED), (x, y))
    y += 22

    state_label = {
        "waiting":    "Starting...",
        "to_item":    "Heading to item",
        "picking_up": "Picking up item",
        "to_depot":   "Returning to depot",
        "resting":    "Resting at depot",
        "all_done":   "Shift complete!",
    }

    def divider():
        nonlocal y
        pygame.draw.line(surface, (40, 50, 65), (x, y), (divider_end, y))
        y += 8

    for agent in agents:
        divider()
        col_agent = agent_colour(agent.agent_id)
        exp_label = "Experienced" if agent.experience_B >= 100 else "Novice"

        surface.blit(font_small.render(
            f"AGENT {agent.agent_id}  [{exp_label}]",
            True, col_agent), (x, y));  y += 16

        status = state_label.get(agent.state, agent.state)
        col    = COL_DONE if agent.is_done() else COL_WHITE
        surface.blit(font_med.render(status, True, col), (x, y));  y += 20

        surface.blit(font_small.render(
            f"Orders: {agent.orders_completed}/{agent.total_orders}"
            f"   Dist: {agent.distance}",
            True, COL_MUTED), (x, y));  y += 16
        surface.blit(font_small.render(
            f"Blocked: {agent.blocked_count}x  ({agent.blocked_count * BLOCKED_WAIT_TICKS}s)",
            True, COL_MUTED), (x, y));  y += 16

        if agent.state == "picking_up":
            surface.blit(font_small.render(
                f"Pickup ticks left: {agent.pickup_ticks_remaining}",
                True, COL_ITEM_DOT), (x, y));  y += 14

        y = draw_fatigue_bar(surface, x, y, bar_width, agent.fatigue, font_small)
        y += 4

    # Congestion
    divider()
    surface.blit(font_small.render("CONGESTION", True, COL_MUTED), (x, y));       y += 16
    conflict_col = COL_CONFLICT if metrics.cell_conflicts > 0 else COL_WHITE
    surface.blit(font_big.render(str(metrics.cell_conflicts), True, conflict_col), (x, y)); y += 30
    surface.blit(font_small.render("cell conflict steps", True, COL_MUTED), (x, y)); y += 20

    # Simulation complete summary
    all_done = all(a.is_done() for a in agents)
    if all_done:
        divider()
        surface.blit(font_med.render("Simulation complete!", True, COL_DONE), (x, y)); y += 20
        for agent in agents:
            surface.blit(font_small.render(
                f"A{agent.agent_id}: {agent.distance} cells | fat={agent.fatigue:.2f}",
                True, agent_colour(agent.agent_id)), (x, y)); y += 14
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
            f"Congestion rate : {kpi_results['congestion_rate']:.4f}",
            True, kpi_col), (x, y)); y += 20

    # Legend
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
# RIGHT-OF-WAY RESOLUTION (generalised for N agents)
# =============================================================================

def resolve_right_of_way(agents):
    """
    For every pair of agents: if they are heading to the same cell or about
    to swap positions head-on, block the lower-priority agent.

    Priority (highest first):
      1. State: to_depot (loaded, returning) beats to_item (going to pick up)
      2. Tiebreaker: lower agent_id beats higher
    """
    active = [a for a in agents if not a.is_done()]

    def priority(agent):
        state_score = 1 if agent.state == "to_depot" else 0
        id_score    = -agent.agent_id   # lower id = higher priority
        return (state_score, id_score)

    for a1, a2 in itertools.combinations(active, 2):
        next1 = a1.peek_next_pos()
        next2 = a2.peek_next_pos()
        if next1 is None or next2 is None:
            continue

        same_dest = (next1 == next2)
        head_on   = (next1 == a2.pos and next2 == a1.pos)
        if not (same_dest or head_on):
            continue

        if priority(a1) >= priority(a2):
            loser = a2
            winner_id = a1.agent_id
        else:
            loser = a1
            winner_id = a2.agent_id

        if loser.blocked_ticks_remaining == 0:
            loser.blocked_count += 1
        loser.blocked_ticks_remaining = BLOCKED_WAIT_TICKS


# =============================================================================
# VISUAL MODE + ENTRY POINT
# =============================================================================

def run_visual(args, demo_mode=False, kpi_results=None):
    """Run one simulation with the Pygame window.
    demo_mode=True and kpi_results provided when called from run_experiment.
    """
    global CELL_SIZE
    CELL_SIZE = args.cell_size

    pygame.init()

    grid   = Grid(
        rows=args.rows, cols=args.cols,
        aisle_width=args.aisle_width,
        centre_aisle_width=args.centre_aisle,
        depot_row=args.depot_row,
        depot_col=args.depot_col,
        shelf_start_row=args.shelf_start,
        shelf_end_row=args.shelf_end,
    )

    # Draw ONE job quota shared by all agents this run
    quota = draw_job_quota()

    # Build N agents — colours cycle from the palette
    agents = [
        Agent(grid, agent_id=i + 1, color=agent_colour(i + 1), total_orders=quota)
        for i in range(args.agents)
    ]

    print("=" * 55)
    mode_label = "DEMO RUN (after experiment)" if demo_mode else "VISUAL MODE"
    print("  Warehouse Simulator  [{}]".format(mode_label))
    print("=" * 55)
    print(f"  Grid         : {grid.rows} rows x {grid.cols} cols")
    print(f"  Aisle width  : {grid.aisle_width}  |  Centre aisle: {grid.centre_aisle_width}")
    print(f"  Depot        : row {grid.depot[0]}, col {grid.depot[1]}")
    print(f"  Shelf zone   : rows {grid.shelf_start_row} to {grid.shelf_end_row}")
    print(f"  Job quota    : drawn from N({JOBS_MEAN}, {JOBS_STD})  [placeholder]")
    print(f"  Agents       : {len(agents)}")
    for agent in agents:
        exp_label = "Experienced" if agent.experience_B >= 100 else "Novice"
        print(f"    Agent {agent.agent_id} : {exp_label} (B={agent.experience_B:.0f}h)")
    print(f"  Step delay   : {args.speed}s")
    print("=" * 55)

    grid_px_w = grid.cols * (CELL_SIZE + MARGIN) + MARGIN
    grid_px_h = grid.rows * (CELL_SIZE + MARGIN) + MARGIN
    win_w     = grid_px_w + PANEL_WIDTH
    win_h     = max(grid_px_h, 400 + len(agents) * 90)

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        ("DEMO RUN | " if demo_mode else "") +
        f"Warehouse Sim — {grid.rows}x{grid.cols} — {len(agents)} agents — quota={quota}"
    )

    font_big   = pygame.font.SysFont("monospace", 18, bold=True)
    font_med   = pygame.font.SysFont("monospace", 14)
    font_small = pygame.font.SysFont("monospace", 11)
    font_grid  = pygame.font.SysFont("monospace", max(7, CELL_SIZE // 4))

    clock          = pygame.time.Clock()
    metrics        = MetricsTracker()
    last_step_time = time.time()

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now      = time.time()
        all_done = all(a.is_done() for a in agents)

        if not all_done and (now - last_step_time) >= args.speed:
            resolve_right_of_way(agents)
            for agent in agents:
                agent.step()
            last_step_time = now

            metrics.update(agents, grid.depot)

            if all(a.is_done() for a in agents):
                metrics.print_summary(agents)

        metrics.tick_flash()

        screen.fill(COL_BG)
        draw_grid(screen, grid, agents, metrics, font_grid)
        draw_panel(screen, grid, agents, metrics, font_big, font_med, font_small, kpi_results=kpi_results)
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

