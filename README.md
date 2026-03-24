# Warehouse Layout Optimisation via Agent-Based Simulation and Reinforcement Learning

Layout design for logistics facilities to maximise operational throughput using simulation-optimisation, human factors modelling, and PPO-based reinforcement learning.

---

## Overview

This project simulates a warehouse operated by multiple human agents and uses a Reinforcement Learning (RL) model to find the layout configuration that maximises performance. The simulation models realistic human behaviour — fatigue buildup, recovery, and experience-based learning curves — based on Malpas & Relvas (2025).

The RL agent (PPO) explores different warehouse layouts by adjusting six layout parameters, including a horizontal cross-aisle, and is rewarded based on three KPIs: picks per hour, travel distance, and congestion rate.

---

## Features

- **Agent-based simulation** — multiple warehouse workers navigating shelves, picking items, and returning to depot
- **Human factors model** — fatigue buildup, recovery, and experience/learning curve per worker
- **Configurable warehouse layout** — aisle width, centre aisle, depot position, shelf zone, and cross-aisle position
- **Horizontal cross-aisle** — a corridor cutting across the shelf zone, with position optimised by the RL model
- **Reinforcement learning** — PPO via Stable Baselines3 searches for the best layout configuration
- **Representative visual** — after a 50-run experiment, replays the run closest to the averaged KPIs
- **Training visualisation** — plots KPIs and layout parameters over RL training episodes

---

## Project Structure

```
Sim/
├── main.py           — Simulation entry point: visual mode and experiment mode
├── agent.py          — Warehouse worker: fatigue, experience, FSM states
├── grid.py           — Warehouse layout builder (numpy-backed grid)
├── pathfinder.py     — A* pathfinding for agent navigation
├── metrics.py        — KPI tracking: picks/hour, distance, congestion
├── rl_env.py         — Gymnasium environment for PPO layout optimisation
└── plot_training.py  — Visualise training results from training_log.csv
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
| `matplotlib` | Training result plots |
| `pandas` | Reading training_log.csv |

---

## How to Run

### Visual Simulation

Run a single simulation with the Pygame display:

```bash
python main.py
```

### Experiment Mode (50-run averaging)

Run 50 silent simulations, compute averaged KPIs, then replay the most representative run visually:

```bash
python main.py --experiment
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
| `--experiment` | — | Run in silent multi-run mode |

Example — run a specific layout found by the RL model:

```bash
python main.py --aisle-width 2 --centre-aisle 1 --depot-col 17 --shelf-start 1 --shelf-end 23 --cross-aisle-row 12
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
| **Picks per Hour** | Higher is better | `total_orders / total_work_time` |
| **Avg Distance per Agent** | Lower is better | `total_distance / num_agents` |
| **Congestion Rate** | Lower is better | `cell_conflicts / total_orders` |

All three KPIs are normalised to 0–1 and combined with equal weighting (1/3 each) to form the RL reward signal.

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