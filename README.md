# Warehouse Simulator

A two-agent warehouse simulation built in Python for a Final Year Project.

Two simulated warehouse workers (agents) complete pickup orders inside a configurable grid warehouse. The simulation models realistic human factors — **fatigue**, **recovery**, and **experience** — based on the model from Malpas & Relvas (2025), and visualises everything in real time using Pygame.

---

## What it does

- Generates a warehouse layout: shelves, aisles, a depot, and item locations
- Runs two agents simultaneously:
  - **Agent 1** — experienced worker (1000 hours prior experience)
  - **Agent 2** — novice worker (20 hours prior experience)
- Each agent picks up 20 items, navigating via A* pathfinding
- Fatigue builds up during work and recovers at the depot
- A right-of-way system resolves collisions (loaded agents have priority)
- Tracks **cell conflicts** (steps where both agents share the same cell)
- Displays a live side panel with fatigue bars, order counts, and a summary

---

## Project structure

```
Sim/
├── main.py         # Entry point: runs the simulation (visual or headless)
├── agent.py        # Agent behaviour, fatigue, experience, and learning curve
├── grid.py         # Warehouse layout builder
├── pathfinder.py   # A* pathfinding algorithm
├── metrics.py      # Conflict detection and summary statistics
└── requirements.txt
```

---

## Requirements

- Python 3.10 or later
- [Pygame](https://www.pygame.org/)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Default run (25x35 grid)
python main.py

# Larger grid
python main.py --rows 30 --cols 45

# Faster animation
python main.py --speed 0.05

# Wider aisles
python main.py --aisle-width 3

# Shift the depot column
python main.py --depot-col 10

# All options
python main.py --help
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `--rows` | 25 | Grid rows |
| `--cols` | 35 | Grid columns (odd recommended) |
| `--speed` | 0.15 | Seconds between simulation steps |
| `--aisle-width` | 2 | Aisle width between shelf blocks |
| `--centre-aisle` | 3 | Centre aisle width (must be odd) |
| `--depot-row` | 0 | Row the depot sits on |
| `--depot-col` | centre | Column the depot sits on |
| `--shelf-start` | 1 | First row where shelves appear |
| `--shelf-end` | rows-2 | Last row where shelves appear |
| `--cell-size` | 35 | Pixel size of each grid cell |

---

## Headless mode (for experiments)

The `run_headless()` function in `main.py` runs the simulation without any graphics and returns a dictionary of metrics. Use this for batch experiments or layout optimisation:

```python
from main import run_headless

results = run_headless(rows=25, cols=35, aisle_width=2, seed=42)
print(results["throughput"])       # orders per combined work-hour
print(results["cell_conflicts"])   # congestion measure
```

Returned keys: `total_distance`, `agent1_distance`, `agent2_distance`, `cell_conflicts`, `agent1_work_time`, `agent2_work_time`, `agent1_fatigue`, `agent2_fatigue`, `agent1_blocked`, `agent2_blocked`, `ticks`, `throughput`, `completed`.

---

## Human factors model

Based on **Malpas & Relvas (2025)**:

| Factor | Formula | Notes |
|---|---|---|
| Fatigue buildup | `I(w) = 1 - e^(-d·w)` | d = 0.20 |
| Fatigue recovery | `R(x) = e^(-r·x) - 1` | r = 0.25 |
| Walking speed | Stochastic pause probability | Higher fatigue → more pauses |
| Pickup duration | `Da = (1 + α·F) · E(w,B) · Do` | Scaled by fatigue and experience |
| Learning curve | `E(w,B) = (w + B)^(-b)` | S-curve; B = prior experience hours |
