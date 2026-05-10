# AIDRA — Adaptive Intelligent Disaster Response Agent

> A hybrid AI system for intelligent disaster response, combining classical search, local search, constraint satisfaction, machine learning, and fuzzy logic in a real-time graphical simulation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [AI Techniques Used](#ai-techniques-used)
  - [Search Algorithms](#search-algorithms)
  - [Local Search](#local-search)
  - [Constraint Satisfaction (CSP)](#constraint-satisfaction-csp)
  - [Machine Learning](#machine-learning)
  - [Fuzzy Logic](#fuzzy-logic)
- [Scenario Setup](#scenario-setup)
- [KPI Dashboard](#kpi-dashboard)
- [Dynamic Events](#dynamic-events)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Overview

AIDRA simulates an intelligent disaster response system on a 12×10 urban grid. Two ambulances must locate, prioritise, and rescue five victims scattered across a disaster zone filled with fire, risk zones, obstacles, and blocked roads — all while operating under uncertainty.

The agent integrates **seven distinct AI paradigms** in a unified pipeline, making real-time trade-off decisions about speed vs. safety, then visualising every decision in a live tkinter + matplotlib GUI.

---

## Key Features

- **Multi-algorithm pathfinding** — BFS, DFS, Greedy Best-First, and A* run and are benchmarked on every mission
- **Dual local search** — Hill Climbing and Simulated Annealing both execute and compete to determine the optimal rescue order
- **CSP resource allocation** — Backtracking with MRV and Forward-Checking assigns all victims to ambulances without constraint violation
- **Ensemble ML survival prediction** — k-NN, Naïve Bayes, and MLP vote to estimate victim survival probability
- **Mamdani fuzzy inference** — Fuzzy logic assesses road blockage risk and per-victim mortality risk under uncertainty
- **Dynamic replanning** — Road blockages, fire spread, new victims, and aftershocks trigger real-time route recalculation
- **Live GUI** — Grid visualisation, KPI dashboard, decision log, algorithm benchmark table, confusion matrices, and comparative charts

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                      AIDRAGUI                       │
│  (tkinter + matplotlib — grid, panels, charts)      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    AIDRAAgent                       │
│                                                     │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │GridSearch │  │ LocalSearch  │  │ ResourceCSP │  │
│  │BFS/DFS/   │  │HillClimbing +│  │MRV+FC+BT   │  │
│  │Greedy/A*  │  │Sim.Annealing │  │             │  │
│  └───────────┘  └──────────────┘  └─────────────┘  │
│                                                     │
│  ┌───────────────────┐  ┌──────────────────────┐   │
│  │  MLRiskEstimator  │  │      FuzzyLogic       │   │
│  │  k-NN / NB / MLP  │  │  Road Blockage Risk   │   │
│  │  Ensemble vote    │  │  Victim Risk Score    │   │
│  └───────────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## AI Techniques Used

### Search Algorithms

All four algorithms navigate the grid from the base to each victim, then onward to the nearest medical centre. They are benchmarked head-to-head on every mission run.

| Algorithm | Strategy | Optimal? | Notes |
|---|---|---|---|
| BFS | Breadth-first | Yes (unweighted) | Explores level by level |
| DFS | Depth-first | No | Low memory, may find long paths |
| Greedy Best-First | Heuristic (Manhattan) | No | Fast but ignores cost |
| A* | f = g + h | Yes (admissible h) | Default; balances cost and heuristic |

**Risk weighting** is applied via the chosen strategy:
- **Fast** — risk_weight = 0.1 (shortest path, accepts hazard cells)
- **Balanced** — risk_weight = 1.0 (equal cost/risk trade-off)
- **Safe** — risk_weight = 4.0 (strongly avoids fire and risk zones)

### Local Search

Both methods optimise the victim rescue ordering within severity tiers. Results are compared after each mission.

- **Hill Climbing** — swaps pairs of victims, keeps improvements only; reports improvement count
- **Simulated Annealing** — accepts worse moves with probability `e^(-Δ/T)`, cooling at 0.95 per iteration; reports accepted moves
- The better-scoring method wins and its ordering is used (after severity-tier sorting)

### Constraint Satisfaction (CSP)

Assigns all victims to one of two ambulances under a hard capacity constraint (max 2 victims per ambulance).

- **Variables** — victim IDs
- **Domain** — ambulance indices {0, 1}
- **Techniques** — Backtracking + MRV (Most Constrained Variable) + Forward-Checking
- If an odd victim cannot fit, the least-loaded ambulance absorbs it (guaranteed full coverage)

### Machine Learning

An ensemble of three classifiers is trained on 500 synthetic patient records at startup.

| Model | Implementation |
|---|---|
| k-NN (k=7) | `sklearn.neighbors.KNeighborsClassifier` |
| Naïve Bayes | `sklearn.naive_bayes.GaussianNB` |
| MLP (16→8) | `sklearn.neural_network.MLPClassifier` |

**Features:** severity (encoded), estimated wait time, distance to medical centre, number of risk cells on path.  
**Label:** survival (binary).  
**Ensemble:** majority vote, blended with a severity-based base rate.  
**Metrics reported:** Accuracy, Precision, Recall, F1, Confusion Matrix.

### Fuzzy Logic

Mamdani-style inference (triangular and trapezoidal membership functions) models two risk factors under uncertainty:

- **Road Blockage Risk** — inputs: aftershock intensity (0–10), fire proximity (0–10); output: blockage probability (0–1)
- **Victim Risk Score** — inputs: severity label, wait time, distance; output: additional mortality risk (0–1)

Nine fuzzy rules cover all combinations of low/medium/high aftershock × fire proximity, using weighted centroid defuzzification.

---

## Scenario Setup

The simulation takes place on a **12 × 10 grid**:

| Cell Type | Colour | Meaning |
|---|---|---|
| Empty | Dark navy | Passable road |
| Fire | Dark red | Hazardous; increases path cost |
| Risk Zone | Dark gold | Moderate hazard |
| Blocked | Dark purple | Road closed (dynamic event) |
| Medical Centre | Dark teal | Rescue destination (2 locations) |
| Base | Dark blue | Ambulance start position |
| Obstacle | Near-black | Impassable |

**Initial victims:**

| ID | Position | Severity | Survival |
|---|---|---|---|
| V1 | (1, 8) | CRITICAL | 85% |
| V2 | (6, 1) | CRITICAL | 80% |
| V3 | (4, 7) | MODERATE | 90% |
| V4 | (1, 3) | MODERATE | 92% |
| V5 | (8, 5) | MINOR | 95% |

---

## KPI Dashboard

Six live metrics are displayed after each mission:

| KPI | Description |
|---|---|
| **SAVED** | Number of victims successfully rescued |
| **AVG TIME** | Average path cost per rescue |
| **RISK EXP** | Total hazardous cells traversed |
| **KITS LEFT** | Medical supply kits remaining (starts at 10) |
| **PATH OPT** | Ratio of actual path cost to optimal A* cost |
| **RES UTIL** | Fraction of total ambulance capacity used |

---

## Dynamic Events

Click **⚡ DYNAMIC EVENT** to inject a random real-time event. The agent replans immediately.

| Event | Effect |
|---|---|
| Road Block | A random empty cell becomes blocked; paths are recalculated |
| Fire Spread | Fire expands to an adjacent cell; fire proximity updated; fuzzy risk recalculated |
| New Victim | A new victim appears at a random empty cell with a random severity |
| Aftershock | Aftershock intensity increases; road blockage risk recalculated |

---

## Installation (Anaconda Recommended)

1. Clone the repository:
git clone https://github.com/your-username/aidra.git
cd aidra

2. Create a new conda environment:
conda create -n aidra-env python=3.10

3. Activate the environment:
conda activate aidra-env

4. Install dependencies:
pip install numpy matplotlib scikit-learn

# OR (optional)
conda install numpy matplotlib scikit-learn

---

## Running in Spyder (Optional)

You can also open AIDRA.py in Spyder (Anaconda) and run it directly.
Make sure the correct conda environment is selected.

---

## Alternative (Without Anaconda)

pip install numpy matplotlib scikit-learn
python AIDRA.py

---

## Usage

```bash
python AIDRA.py
```

### GUI Controls

| Control | Action |
|---|---|
| **▶ RUN MISSION** | Execute a full rescue mission with the selected strategy and algorithm |
| **⚡ DYNAMIC EVENT** | Inject a random environmental event and trigger replanning |
| **📊 COMPARE** | Open the comparative evaluation window (requires at least one mission run) |
| **↺ RESET** | Restore the grid and all state to initial conditions |
| **Strategy** | Fast / Balanced / Safe — controls the risk weighting for pathfinding |
| **Algorithm** | A* / Greedy / BFS / DFS — selects the active routing algorithm |

### Comparison Window

After running a mission, click **📊 COMPARE** to view six charts side-by-side:

- Nodes expanded per algorithm
- Path cost per algorithm
- Execution time per algorithm
- Risk exposure per algorithm
- ML model metrics (Accuracy, Precision, Recall, F1)
- Hill Climbing vs Simulated Annealing (score + time)

---

## Project Structure

```
AIDRA.py
│
├── FuzzyLogic              # Mamdani fuzzy inference (road blockage + victim risk)
├── GridSearch              # BFS, DFS, Greedy Best-First, A* on the 12×10 grid
├── LocalSearch             # Hill Climbing and Simulated Annealing for rescue ordering
├── ResourceCSP             # Backtracking + MRV + Forward-Checking victim allocation
├── MLRiskEstimator         # k-NN, Naïve Bayes, MLP ensemble survival predictor
├── AIDRAAgent              # Core mission controller — integrates all AI modules
└── AIDRAGUI                # tkinter + matplotlib GUI with all panels and charts
```

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.8+ |
| numpy | any recent |
| matplotlib | any recent |
| scikit-learn | any recent |
| tkinter | bundled with Python |

Install with:
```bash
pip install numpy matplotlib scikit-learn
```
---

## License

This project was developed as an academic demonstration of hybrid AI techniques applied to disaster response simulation.

## Author
Rimsha Fareed
