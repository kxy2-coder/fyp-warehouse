# =============================================================================
# record_replay.py — Export simulation replay for Godot 3D visualisation
# =============================================================================
# Runs the warehouse simulation headlessly (no Pygame) for a given number of
# ticks and saves every agent position + state to a JSON file that the Godot
# visualiser can load and replay.
#
# Usage:
#   python record_replay.py                                    # defaults
#   python record_replay.py --agents 6 --depot-count 2        # 6 agents, 2 depots
#   python record_replay.py --ticks 2000                      # shorter recording
#   python record_replay.py --out ../Godot_Vis/replay.json    # save to Godot folder
#   python record_replay.py --agents 6 --depot-count 2 --ticks 3600 --out ../Godot_Vis/replay.json
#
# Output JSON structure:
#   {
#     "meta":   { grid_rows, grid_cols, n_agents, ticks_recorded, free_speed_ms, seed }
#     "layout": { grid_rows, grid_cols, n_agents, shelves: [[r,c],...], depots: [[r,c],...] }
#     "frames": [ { "tick": 0, "agents": [ {"id":0,"x":1.5,"y":0.5,"state":"waiting","speed":0.0}, ... ] }, ... ]
#   }
#
# File size guide (compact JSON):
#   1 000 ticks × 6 agents ≈  0.3 MB
#   3 600 ticks × 6 agents ≈  1.0 MB
#  28 800 ticks × 6 agents ≈  8.0 MB   ← default (full 8-hour shift)
# =============================================================================

import argparse
import json
import os
import random

from grid           import Grid, SHELF, ITEM, DEPOT
from jupedsim_agent import (JupedSimAgent, FREE_SPEED,
                             STATE_WAITING, STATE_TO_ITEM, STATE_PICKING,
                             STATE_TO_DEPOT, STATE_RESTING, STATE_ALL_DONE)
import agent as agent_mod
from agent          import nhpp_arrival, SHIFT_TICKS
from rl_env         import compute_depot_cols


# Map integer state → readable string for Godot
STATE_NAMES = {
    STATE_WAITING:  "waiting",
    STATE_TO_ITEM:  "to_item",
    STATE_PICKING:  "picking",
    STATE_TO_DEPOT: "to_depot",
    STATE_RESTING:  "resting",
    STATE_ALL_DONE: "done",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export warehouse simulation replay for Godot 3D visualiser"
    )
    parser.add_argument("--agents",         type=int,   default=6,
                        help="Number of agents (default 6)")
    parser.add_argument("--rows",           type=int,   default=25,
                        help="Grid rows (default 25)")
    parser.add_argument("--cols",           type=int,   default=35,
                        help="Grid cols (default 35)")
    parser.add_argument("--depot-count",    type=int,   default=1,  choices=[1, 2, 3, 4],
                        help="Number of equally-spaced depots 1–4 (default 1)")
    parser.add_argument("--depot-col",      type=int,   default=None,
                        help="Single depot column override (ignored if --depot-count is set)")
    parser.add_argument("--depot-row",      type=int,   default=None,
                        help="Depot row (default 0 = top row)")
    parser.add_argument("--aisle-width",    type=int,   default=2,
                        help="Aisle width between shelf blocks (default 2)")
    parser.add_argument("--centre-aisle",   type=int,   default=3,
                        help="Centre aisle width (default 3)")
    parser.add_argument("--shelf-start",    type=int,   default=2,
                        help="First row shelves appear (default 2)")
    parser.add_argument("--shelf-end",      type=int,   default=23,
                        help="Last row shelves appear (default 23)")
    parser.add_argument("--no-cross-aisle", action="store_true",
                        help="Disable the cross-aisle")
    parser.add_argument("--ticks",          type=int,   default=28800,
                        help="Number of ticks to record (default 28800 = full 8-hour shift)")
    parser.add_argument("--lambda-base",    type=float, default=None,
                        help="NHPP base arrival rate per tick (default = agent.py LAMBDA_BASE)")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Random seed for reproducible replays (default 42)")
    parser.add_argument("--out",            type=str,   default="replay.json",
                        help="Output JSON path (default: replay.json)")
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)

    if args.lambda_base is not None:
        agent_mod.LAMBDA_BASE = args.lambda_base

    # ── Resolve depot columns ─────────────────────────────────────────────────
    if args.depot_count is not None:
        depot_cols = compute_depot_cols(args.depot_count, args.cols)
    elif args.depot_col is not None:
        depot_cols = [args.depot_col]
    else:
        depot_cols = [args.cols // 2]

    # ── Build grid ────────────────────────────────────────────────────────────
    grid = Grid(
        rows=args.rows, cols=args.cols,
        aisle_width=args.aisle_width,
        centre_aisle_width=args.centre_aisle,
        depot_row=args.depot_row,
        depot_cols=depot_cols,
        shelf_start_row=args.shelf_start,
        shelf_end_row=args.shelf_end,
        cross_aisle_enabled=not args.no_cross_aisle,
    )

    agent = JupedSimAgent(args.agents, grid)

    # ── Layout snapshot (shelves and depots, fixed for the whole replay) ──────
    shelves = []
    depots  = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            cell = grid.cells[r, c]
            if cell in (SHELF, ITEM):
                shelves.append([r, c])
            elif cell == DEPOT:
                depots.append([r, c])

    layout = {
        "grid_rows": args.rows,
        "grid_cols": args.cols,
        "n_agents":  args.agents,
        "shelves":   shelves,
        "depots":    depots,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    ticks_to_record = min(args.ticks, SHIFT_TICKS)
    print("=" * 55)
    print("  record_replay.py")
    print("=" * 55)
    print(f"  Grid         : {args.rows} rows × {args.cols} cols")
    print(f"  Agents       : {args.agents}")
    print(f"  Depots       : {len(depot_cols)} at columns {depot_cols}")
    print(f"  Ticks        : {ticks_to_record}  (1 tick = 1 second)")
    print(f"  Sim duration : {ticks_to_record // 3600}h {(ticks_to_record % 3600) // 60}m")
    print(f"  Seed         : {args.seed}")
    print(f"  Output       : {args.out}")
    print(f"  Shelves      : {len(shelves)}  |  Depots: {len(depots)}")
    print("=" * 55)

    # ── Record frames ─────────────────────────────────────────────────────────
    frames = []

    for tick in range(ticks_to_record):
        if nhpp_arrival(tick):
            agent.add_job()
        agent.step()
        grid.tick_replenishment()

        agent_data = []
        for i in range(agent.n):
            x, y = agent.get_xy(i)
            agent_data.append({
                "id":    i,
                "x":     round(x, 3),
                "y":     round(y, 3),
                "state": STATE_NAMES.get(int(agent.state[i]), "unknown"),
                "speed": round(float(agent.get_speed(i)), 3),
            })

        frames.append({
            "tick":   tick,
            "agents": agent_data,
        })

        # Progress update every 500 ticks
        if (tick + 1) % 500 == 0:
            pct = (tick + 1) / ticks_to_record * 100
            completed = int(agent.orders_completed.sum())
            print(f"  Tick {tick + 1:5d}/{ticks_to_record}  ({pct:5.1f}%)  "
                  f"jobs completed: {completed}")

    # ── Assemble and write JSON ───────────────────────────────────────────────
    replay = {
        "meta": {
            "grid_rows":      args.rows,
            "grid_cols":      args.cols,
            "n_agents":       args.agents,
            "ticks_recorded": ticks_to_record,
            "free_speed_ms":  FREE_SPEED,
            "seed":           args.seed,
        },
        "layout": layout,
        "frames": frames,
    }

    # Create output directory if it doesn't exist yet
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Writing {args.out} ...")
    with open(args.out, "w") as f:
        # separators=(",",":") removes all whitespace → compact file
        json.dump(replay, f, separators=(",", ":"))

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"  Done.  {size_mb:.1f} MB  |  {len(frames)} frames  |  {args.agents} agents")
    print()
    print("  Next step: copy replay.json to your Godot project folder")
    print("  and load it in the replay script.")
    print("=" * 55)


if __name__ == "__main__":
    main()
