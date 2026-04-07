# Warehouse Layout Optimisation via Agent-Based Simulation and Reinforcement Learning

Layout design for logistics facilities to maximise operational throughput using simulation-optimisation, human factors modelling, and PPO-based reinforcement learning.

---

## Overview


This project simulates a warehouse operated by multiple human agents and uses a Reinforcement Learning (RL) model to find the layout configuration that maximises throughput.

Job arrivals follow a Nonhomogeneous Poisson Process (NHPP) with time-varying demand (morning peak, mid-shift lull, end-of-shift rush). Conflicts between agents are resolved reactively (post-movement collision detection), matching how human workers actually behave. The RL agent (PPO) explores different warehouse layouts by adjusting six layout parameters and is rewarded based on three KPIs: picks per hour, travel distance, and congestion rate.

---

## Features

- **Agent-based simulation** — multiple warehouse workers navigating shelves, picking items, and returning to depot
- **NHPP demand model** — job arrivals follow a nonhomogeneous Poisson process with piecewise demand multipliers across the 8-hour shift
- **Human factors model** — fatigue buildup, recovery, and experience/learning curve per worker
- **Reactive conflict resolution** — post-movement collision detection; one agent waits 3 ticks while the other passes
- **First-come-first-served dispatch** — idle agents are assigned new jobs in order of longest waiting time
- **Configurable warehouse layout** — aisle width, centre aisle, depot position, shelf zone, and cross-aisle position
- **Reinforcement learning** — PPO via Stable Baselines3 searches for the best layout configuration
- **Per-agent performance traces** — picks, distance, idle ticks, and blocked events logged every 10 minutes per agent
- **Spatial congestion heatmap** — conflict locations recorded per tick and overlaid on the warehouse grid
- **Representative visual** — after a multi-run experiment, replays the run closest to the averaged KPIs

---

## Project Structure

```
<<<<<<< HEAD
Sim/
├── main.py           — Simulation entry point: visual mode and experiment mode
├── agent.py          — Warehouse worker: fatigue, experience, FSM states
├── grid.py           — Warehouse layout 
├── pathfinder.py     — A* pathfinding for agent navigation
├── metrics.py        — KPI tracking: picks/hour, distance, congestion
├── rl_env.py         — Gymnasium environment for PPO layout optimisation
└── plot_training.py  — Visualise training results from training_log.csv
=======
Sim_V2/
├── main.py               — Simulation entry point: visual mode and experiment mode
├── agent.py              — Warehouse worker: fatigue, experience, NHPP arrivals, FSM states
├── grid.py               — Warehouse layout builder (numpy-backed grid)
├── pathfinder.py         — A* pathfinding for agent navigation
├── metrics.py            — KPI tracking: conflicts, spatial log, per-agent trace logger
├── rl_env.py             — Gymnasium environment for PPO layout optimisation
├── plot_workload.py      — Workload analysis plot (arrivals vs completions over shift)
├── plot_agent_traces.py  — Per-agent performance trace plots from agent_traces.csv
├── plot_heatmap.py       — Spatial conflict heatmap from conflict_heatmap.csv
└── plot_training.py      — Visualise training results from training_log.csv
>>>>>>> e5ec011 (add spatial congestion logging and per-agent performance trace)
```

---

## Dependencies

Install all required packages:

```bash
pip install pygame numpy gymnasium stable-baselines3 matplotlib pandas
```

| Package | Purpose |
|---------|---------|
| `pygame` | Visual simulation display |
| `numpy` | Grid storage and array operations |
| `gymnasium` | RL environment interface |
| `stable-baselines3` | PPO training algorithm |
| `matplotlib` | All visualisation plots |
| `pandas` | Reading training_log.csv |

---

## How to Run

### Visual Simulation

Run a single simulation with the Pygame display:

```bash
python main.py
```

### Experiment Mode (multi-run averaging)

Run 50 silent simulations, compute averaged KPIs, then replay the most representative run visually:

```bash
python main.py --experiment
```

With optional visualisation outputs:

```bash
python main.py --experiment --plot-workload    # arrivals vs completions over shift
python main.py --experiment --plot-traces      # per-agent performance traces
python main.py --experiment --plot-heatmap     # spatial conflict heatmap
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--agents` | 4 | Number of warehouse workers |
| `--rows` | 25 | Grid height |
| `--cols` | 35 | Grid width |
| `--aisle-width` | 2 | Width of aisles between shelf blocks |
| `--centre-aisle` | 3 | Width of the main vertical centre aisle |
| `--depot-row` | 0 | Row the depot sits on |
| `--depot-col` | centre | Column the depot sits on |
| `--shelf-start` | 1 | First row where shelves appear |
| `--shelf-end` | rows-2 | Last row where shelves appear |
| `--cross-aisle-row` | middle | Row of the horizontal cross-aisle |
| `--speed` | 0.15 | Seconds between simulation steps |
| `--cell-size` | 35 | Pixel size per cell |
| `--runs` | 50 | Number of runs in experiment mode |
| `--replenish-delay` | 100 | Ticks before a picked shelf restocks |
| `--lambda-base` | 0.0778 | NHPP base arrival rate per tick |
| `--beta` | 0.3 | NHPP demand amplitude (peak/trough strength) |
| `--experiment` | — | Run in silent multi-run mode |
| `--plot-workload` | — | Save workload analysis plot |
| `--plot-traces` | — | Save per-agent trace plot |
| `--plot-heatmap` | — | Save spatial conflict heatmap |
| `--bin` | 600 | Bin size in ticks for workload plot (default = 10 min) |

Example — run a specific layout found by the RL model:

```bash
python main.py --experiment --plot-heatmap --aisle-width 2 --centre-aisle 1 --depot-col 17 --shelf-start 1 --shelf-end 23 --cross-aisle-row 12
```

### RL Training

Train the PPO agent to optimise the warehouse layout:

```bash
python rl_env.py
```

Options:

```bash
python rl_env.py --timesteps 50000       # longer training
python rl_env.py --eval-runs 1           # faster (noisier) evaluation
python rl_env.py --calibration-runs 200  # more calibration samples
```

Outputs:
- `warehouse_layout_ppo_v3.zip` — saved trained model
- `training_log.csv` — episode-by-episode layout parameters and KPIs

### Plot Training Results

```bash
python plot_training.py
```

Outputs `training_results.png` with 6 subplots: reward, picks/hour, distance, congestion, layout parameters over time, and normalised KPI contributions.

### Plot Spatial Heatmap (standalone)

```bash
python plot_heatmap.py                                        # single layout
python plot_heatmap.py --compare layout_a.csv layout_b.csv --labels "Narrow" "Wide"
```

---

## NHPP Demand Model

Job arrivals follow a Nonhomogeneous Poisson Process with a piecewise time-varying multiplier across the 8-hour shift:

| Phase | Ticks | Multiplier |
|-------|-------|------------|
| Morning peak | 0 – 7200 | 1 + 0.4β |
| Mid-shift lull | 7200 – 21600 | 1 − 0.3β |
| End-of-shift rush | 21600 – 28800 | 1 + 0.4β |

With defaults (λ_base = 0.0778, β = 0.3), expected arrivals ≈ 2250 jobs per 8-hour shift.

---

## Layout Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `aisle_width` | 2 | 1–3 | Column gap between shelf blocks |
| `centre_aisle_width` | 3 | 1–3 | Width of the main vertical centre aisle |
| `depot_col` | 17 (centre) | 2–32 | Horizontal depot position |
| `shelf_start_row` | 1 | 1–6 | Top of the shelf zone |
| `shelf_end_row` | 23 | 18–23 | Bottom of the shelf zone |
| `cross_aisle_row` | 12 (middle) | shelf_start–shelf_end | Row of the horizontal cross-aisle |

---

## Key Performance Indicators (KPIs)

| KPI | Direction | Formula |
|-----|-----------|---------|
| **Picks per Hour** | Higher is better | `jobs_completed / 8.0` |
| **Avg Distance per Agent** | Lower is better | `total_distance / num_agents` |
| **Congestion Rate** | Lower is better | `(cell_conflicts × blocked_wait_ticks × 100) / shift_ticks` (% of shift time affected) |

All three KPIs are normalised to 0–1 and combined with equal weighting (1/3 each) to form the RL reward signal.

---

## Conflict Resolution

Conflicts are detected **reactively** after each movement step. When two agents occupy the same non-depot cell:

- The agent carrying an item (returning to depot) takes priority; ties are broken randomly
- The losing agent waits 3 ticks before resuming
- The cell coordinates and tick are logged to produce the spatial conflict heatmap

---

## Per-Agent Performance Traces

Every 10 minutes (600 ticks) a snapshot is recorded per agent:

| Field | Description |
|-------|-------------|
| `picks` | Cumulative orders completed |
| `distance` | Cumulative cells travelled |
| `idle_ticks` | Cumulative ticks spent waiting for a job |
| `blocked_events` | Cumulative collision losses |
| `fatigue` | Current fatigue level (0–1) |

Snapshots are averaged across all runs and saved to `agent_traces.csv`. Run `--plot-traces` to produce `agent_traces.png` showing per-bin (non-cumulative) rates for each agent.

---

## Spatial Conflict Heatmap

Every collision records `(tick, row, col)`. After all runs these are averaged per cell and saved to `conflict_heatmap.csv`. The heatmap (`conflict_heatmap.png`) overlays conflict intensity on the warehouse grid using a sqrt colour scale (yellow → dark red) clipped at the 98th percentile to prevent a single hotspot from washing out the rest of the map.

For layout comparison, `plot_heatmap.py --compare` uses the true raw maximum across both layouts so absolute magnitudes remain directly comparable.

---

## RL Environment

| Component | Detail |
|-----------|--------|
| **State** | 6 normalised layout parameters |
| **Actions** | 12 discrete nudges (±1 per parameter, ±2 for depot column) |
| **Reward** | Normalised weighted sum of 3 KPIs |
| **Episode length** | 5 parameter nudges, then evaluate |
| **Algorithm** | PPO (Stable Baselines3) |
| **Runs per evaluation** | 5 simulations averaged |

---

## Human Factors Model

Based on Malpas & Relvas (2025):

| Factor | Formula |
|--------|---------|
| Fatigue buildup | `I(w) = 1 - e^(-0.20 * work_time)` |
| Fatigue recovery | `R(x) = e^(-0.25 * rest_time) - 1` |
| Walking speed effect | Pause probability = `0.40 * fatigue / 2` |
| Pickup time effect | `(1 + 0.40 * fatigue) * experience_factor * base_ticks` |
| Experience curve | `E(w, B) = (work_time + B)^(-b)` |

Two worker types: **Novice** (B=20h, 65% of agents) and **Expert** (B=1000h, 35% of agents).

---

## Storage Map

The grid exposes a binary storage map for analysis and reporting:

```python
from grid import Grid
grid = Grid()
smap = grid.storage_map()  # numpy array: 1=shelf, 0=floor
utilisation = smap.sum() / smap.size
print(f"Storage utilisation: {utilisation:.1%}")
```
