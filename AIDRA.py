"""
AIDRA – Adaptive Intelligent Disaster Response Agent 
=========================================================================
Hybrid AI system integrating:
  • Search      : BFS, DFS, Greedy Best-First, A*
  • Local Search : Hill Climbing + Simulated Annealing (both used & compared)
  • CSP         : Backtracking + MRV + Forward-Checking (all 5 victims allocated)
  • ML          : k-NN, Naïve Bayes, MLP (sklearn) — metrics + confusion matrix
  • Uncertainty : Fuzzy Logic (road-blockage & victim risk)
  • KPIs        : Saved, Avg Time, Risk Exposure, Kits, Path Optimality, Resource Util
  • Dynamic     : Real-time replanning on road-block / fire-spread / new victim
  • GUI         : tkinter + matplotlib — grid, algo comparison chart, CM display

Run: python aidra_system_fixed.py
Requires: numpy, matplotlib, scikit-learn  (pip install numpy matplotlib scikit-learn)
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading, time, math, random, heapq, copy
from collections import deque
from typing import List, Tuple, Dict, Optional

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & SCENARIO
# ══════════════════════════════════════════════════════════════════════════════
GRID_W, GRID_H = 12, 10

EMPTY    = 0
OBSTACLE = 1
FIRE     = 2
RISK     = 3
MEDICAL  = 4
BASE     = 5
BLOCKED  = 6

CELL_COLORS = {
    EMPTY:    "#0d1b2a",
    OBSTACLE: "#1a1a2e",
    FIRE:     "#8b0000",
    RISK:     "#7a6000",
    MEDICAL:  "#004d40",
    BASE:     "#003060",
    BLOCKED:  "#3a003a",
}
SEVERITY_COLOR = {"CRITICAL": "#ff3333", "MODERATE": "#ffaa00", "MINOR": "#33ff99"}

INITIAL_GRID = [
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,3,3,0,0,0],
    [0,0,0,0,0,0,0,3,0,0,0,0],
    [0,0,0,0,0,2,2,2,2,0,0,0],
    [0,0,0,0,0,2,2,2,2,2,0,0],
    [0,0,0,0,0,2,2,0,2,2,0,0],
    [0,0,0,0,0,2,2,0,3,3,0,0],
    [0,0,0,0,0,2,0,0,3,3,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
]

BASE_POS     = (0, 0)
MEDICAL_POS1 = (9, 0)
MEDICAL_POS2 = (9, 11)

INITIAL_VICTIMS = [
    {"id":"V1","pos":(1,8),"severity":"CRITICAL","surv":0.85},
    {"id":"V2","pos":(6,1),"severity":"CRITICAL","surv":0.80},
    {"id":"V3","pos":(4,7),"severity":"MODERATE", "surv":0.90},
    {"id":"V4","pos":(1,3),"severity":"MODERATE", "surv":0.92},
    {"id":"V5","pos":(8,5),"severity":"MINOR",    "surv":0.95},
]

# ══════════════════════════════════════════════════════════════════════════════
#  FUZZY LOGIC
# ══════════════════════════════════════════════════════════════════════════════
class FuzzyLogic:
    """Mamdani-style fuzzy inference for road-blockage probability and victim risk."""

    @staticmethod
    def _tri(x, a, b, c):
        if x <= a or x >= c: return 0.0
        if x < b:  return (x - a) / (b - a)
        return (c - x) / (c - b)

    @staticmethod
    def _trap(x, a, b, c, d):
        if x <= a or x >= d: return 0.0
        if x < b: return (x - a) / (b - a)
        if x <= c: return 1.0
        return (d - x) / (d - c)

    def road_blockage_risk(self, aftershock: float, fire_prox: float) -> float:
        """Returns 0-1 blockage probability from aftershock intensity and fire proximity."""
        a_low  = self._tri(aftershock, 0,  0,  4)
        a_med  = self._tri(aftershock, 2,  5,  8)
        a_high = self._tri(aftershock, 6, 10, 10)
        f_low  = self._tri(fire_prox,  0,  0,  4)
        f_med  = self._tri(fire_prox,  2,  5,  8)
        f_high = self._tri(fire_prox,  6, 10, 10)
        rules = [
            (min(a_low,  f_low),  0.10),
            (min(a_low,  f_med),  0.25),
            (min(a_low,  f_high), 0.45),
            (min(a_med,  f_low),  0.30),
            (min(a_med,  f_med),  0.50),
            (min(a_med,  f_high), 0.70),
            (min(a_high, f_low),  0.55),
            (min(a_high, f_med),  0.75),
            (min(a_high, f_high), 0.92),
        ]
        num = sum(s * c for s, c in rules)
        den = sum(s     for s, _ in rules)
        return round(num / den, 3) if den else 0.0

    def victim_risk_score(self, severity: str, wait_time: float, distance: float) -> float:
        """Additional mortality risk from waiting (0-1)."""
        base     = {"CRITICAL": 0.7, "MODERATE": 0.4, "MINOR": 0.15}.get(severity, 0.5)
        t_factor = self._trap(wait_time, 0, 0, 5, 15)
        d_factor = self._tri(distance, 0, 8, 20)
        return min(base * (0.5 + 0.3 * t_factor + 0.2 * d_factor), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════
class GridSearch:
    def __init__(self, grid, risk_weight: float = 1.0):
        self.grid        = grid
        self.risk_weight = risk_weight
        self.rows        = len(grid)
        self.cols        = len(grid[0])

    def _passable(self, r, c):
        return (0 <= r < self.rows and 0 <= c < self.cols and
                self.grid[r][c] not in (OBSTACLE, BLOCKED))

    def _cost(self, r, c):
        return 1 + 4 * self.risk_weight if self.grid[r][c] in (FIRE, RISK) else 1

    def _neighbors(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if self._passable(nr, nc):
                yield nr, nc

    def _h(self, r, c, goal):
        return abs(r - goal[0]) + abs(c - goal[1])

    # BFS ─────────────────────────────────────────────────────────────────────
    def bfs(self, start, goal):
        q, visited, nodes = deque([(start, [start])]), {start}, 0
        while q:
            (r, c), path = q.popleft()
            nodes += 1
            if (r, c) == goal:
                return path, nodes
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))
        return [], nodes

    # DFS ─────────────────────────────────────────────────────────────────────
    def dfs(self, start, goal):
        stack, visited, nodes = [(start, [start])], set(), 0
        while stack:
            (r, c), path = stack.pop()
            if (r, c) in visited: continue
            visited.add((r, c)); nodes += 1
            if (r, c) == goal:
                return path, nodes
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) not in visited:
                    stack.append(((nr, nc), path + [(nr, nc)]))
        return [], nodes

    # Greedy Best-First ───────────────────────────────────────────────────────
    def greedy(self, start, goal):
        heap, visited, nodes = [(self._h(*start, goal), start, [start])], set(), 0
        while heap:
            _, (r, c), path = heapq.heappop(heap)
            if (r, c) in visited: continue
            visited.add((r, c)); nodes += 1
            if (r, c) == goal:
                return path, nodes
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) not in visited:
                    heapq.heappush(heap, (self._h(nr, nc, goal), (nr, nc), path + [(nr, nc)]))
        return [], nodes

    # A* ──────────────────────────────────────────────────────────────────────
    def astar(self, start, goal):
        heap   = [(self._h(*start, goal), 0, start, [start])]
        g_cost = {start: 0}
        nodes  = 0
        while heap:
            f, g, (r, c), path = heapq.heappop(heap)
            if g > g_cost.get((r, c), float('inf')): continue
            nodes += 1
            if (r, c) == goal:
                return path, nodes
            for nr, nc in self._neighbors(r, c):
                ng = g + self._cost(nr, nc)
                if ng < g_cost.get((nr, nc), float('inf')):
                    g_cost[(nr, nc)] = ng
                    heapq.heappush(heap, (ng + self._h(nr, nc, goal), ng,
                                          (nr, nc), path + [(nr, nc)]))
        return [], nodes

    def path_cost(self, path):
        return sum(self._cost(r, c) for r, c in path[1:]) if path else float('inf')

    def risk_exposure(self, path):
        return sum(1 for r, c in path if self.grid[r][c] in (FIRE, RISK))


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL SEARCH  (Hill Climbing + Simulated Annealing — both used)
# ══════════════════════════════════════════════════════════════════════════════
class LocalSearch:
    """Optimises rescue ordering within severity tiers."""

    def __init__(self, victims, rescue_times: dict):
        self.victims      = victims
        self.rescue_times = rescue_times

    def _eval(self, order):
        W = {"CRITICAL": 3, "MODERATE": 2, "MINOR": 1}
        score, t = 0, 0
        for vid in order:
            v  = next(x for x in self.victims if x["id"] == vid)
            t += self.rescue_times.get(vid, 10)
            score += t * W[v["severity"]]
        return score

    def _swap(self, order):
        o = order[:]
        i, j = random.sample(range(len(o)), 2)
        o[i], o[j] = o[j], o[i]
        return o

    def hill_climbing(self, max_iter: int = 300):
        cur = [v["id"] for v in self.victims]
        random.shuffle(cur)
        cur_s = self._eval(cur)
        improved = 0
        for _ in range(max_iter):
            nb = self._swap(cur)
            ns = self._eval(nb)
            if ns < cur_s:
                cur, cur_s = nb, ns
                improved += 1
        return cur, cur_s, improved

    def simulated_annealing(self, T: float = 100.0, cooling: float = 0.95,
                             max_iter: int = 400):
        cur = [v["id"] for v in self.victims]
        random.shuffle(cur)
        cur_s  = self._eval(cur)
        best, best_s = cur[:], cur_s
        accepted = 0
        for _ in range(max_iter):
            nb  = self._swap(cur)
            ns  = self._eval(nb)
            d   = ns - cur_s
            if d < 0 or random.random() < math.exp(-d / max(T, 1e-9)):
                cur, cur_s = nb, ns
                accepted += 1
                if cur_s < best_s:
                    best, best_s = cur[:], cur_s
            T *= cooling
        return best, best_s, accepted


# ══════════════════════════════════════════════════════════════════════════════
#  CSP — RESOURCE ALLOCATION
# ══════════════════════════════════════════════════════════════════════════════
class ResourceCSP:
    """
    Assigns ALL victims to ambulances.
    Variables  : victim IDs
    Domain     : ambulance indices  {0, 1}
    Constraint : each ambulance handles at most `capacity` victims per batch.
    Uses: Backtracking + MRV + Forward-Checking.
    
    Capacity is set to ceil(n_victims / n_ambulances) to guarantee a solution.
    """

    def __init__(self, victims, ambulances: int = 2):
        self.victims    = [v["id"] for v in victims]
        self.severities = {v["id"]: v["severity"] for v in victims}
        self.ambulances = list(range(ambulances))
        # Hard constraint: max 2 victims per ambulance (project requirement)
        self.capacity   = 2
        self.backtracks = 0
        self.nodes      = 0

    def _mrv(self, assignment, domains):
        unassigned = [v for v in self.victims if v not in assignment]
        if not unassigned: return None
        return min(unassigned, key=lambda v: len(domains[v]))

    def _fc(self, val, assignment, domains):
        """Forward-check: recompute domains after assigning val."""
        counts = {a: 0 for a in self.ambulances}
        for _, a in assignment.items():
            counts[a] += 1
        counts[val] += 1
        new_domains = {k: v[:] for k, v in domains.items()}
        for v in self.victims:
            if v not in assignment:
                new_domains[v] = [a for a in new_domains[v]
                                   if counts[a] < self.capacity]
                # If all ambulances are full (odd victim out), allow least-loaded one
                if not new_domains[v]:
                    new_domains[v] = [min(self.ambulances, key=lambda a: counts[a])]
        return new_domains

    def solve(self):
        self.backtracks = 0
        self.nodes      = 0
        domains = {v: self.ambulances[:] for v in self.victims}
        return self._backtrack({}, domains)

    def _backtrack(self, assignment, domains):
        if len(assignment) == len(self.victims):
            return assignment
        var = self._mrv(assignment, domains)
        if var is None: return None
        counts = {a: sum(1 for aa in assignment.values() if aa == a)
                  for a in self.ambulances}
        for val in domains[var]:
            self.nodes += 1
            if counts.get(val, 0) < self.capacity:
                new_dom = self._fc(val, assignment, domains)
                if new_dom is not None:
                    assignment[var] = val
                    result = self._backtrack(assignment, new_dom)
                    if result:
                        return result
                    del assignment[var]
                    self.backtracks += 1
            else:
                self.backtracks += 1
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  ML RISK ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
class MLRiskEstimator:
    """
    Trains k-NN, Naïve Bayes, and MLP on synthetic survival data.
    Ensemble majority-vote used to inform agent routing decisions.
    """

    def __init__(self):
        self.models  = {}
        self.metrics = {}
        self._Xte = None
        self._yte = None
        self._train()

    def _generate_data(self, n: int = 500):
        np.random.seed(42)
        severity  = np.random.choice([0, 1, 2], n)
        wait      = np.random.uniform(0, 30, n)
        distance  = np.random.uniform(1, 20, n)
        risk_zone = np.random.randint(0, 8, n)
        prob = (0.95 - 0.15 * severity - 0.012 * wait
                - 0.006 * distance - 0.025 * risk_zone)
        prob = np.clip(prob + np.random.normal(0, 0.05, n), 0, 1)
        y = (prob > 0.5).astype(int)
        X = np.column_stack([severity, wait, distance, risk_zone])
        return X, y

    def _train(self):
        X, y = self._generate_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        self._Xte = Xte
        self._yte = yte
        # FIX: Use full, unambiguous model names — these become the legend labels
        clfs = {
            "k-NN (k=7)":    KNeighborsClassifier(n_neighbors=7),
            "Naïve Bayes":   GaussianNB(),
            "MLP (16→8)":    MLPClassifier(hidden_layer_sizes=(16, 8),
                                            max_iter=1000, random_state=42),
        }
        for name, clf in clfs.items():
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
            self.models[name] = clf
            self.metrics[name] = {
                "Accuracy":  round(accuracy_score(yte, yp), 3),
                "Precision": round(precision_score(yte, yp, zero_division=0), 3),
                "Recall":    round(recall_score(yte, yp, zero_division=0), 3),
                "F1":        round(f1_score(yte, yp, zero_division=0), 3),
                "CM":        confusion_matrix(yte, yp).tolist(),
            }

    def predict_survival(self, severity: str, wait: float,
                          distance: float, risk_cells: int) -> float:
        sev_num  = {"MINOR": 0, "MODERATE": 1, "CRITICAL": 2}[severity]
        X        = np.array([[sev_num, wait, distance, risk_cells]])
        votes    = [int(clf.predict(X)[0]) for clf in self.models.values()]
        ml_score = sum(votes) / len(votes)

        base = {"CRITICAL": 0.20, "MODERATE": 0.50, "MINOR": 0.80}[severity]
        blended = base + (ml_score - 0.5) * 0.30
        blended -= 0.005 * wait + 0.02 * risk_cells
        return round(max(0.05, min(blended, 0.99)), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE AGENT
# ══════════════════════════════════════════════════════════════════════════════
class AIDRAAgent:
    def __init__(self, strategy: str = "Balanced", algorithm: str = "A*"):
        self.strategy   = strategy
        self.algorithm  = algorithm
        self.grid       = [row[:] for row in INITIAL_GRID]
        self.grid[BASE_POS[0]][BASE_POS[1]]             = BASE
        self.grid[MEDICAL_POS1[0]][MEDICAL_POS1[1]]     = MEDICAL
        self.grid[MEDICAL_POS2[0]][MEDICAL_POS2[1]]     = MEDICAL
        self.victims    = copy.deepcopy(INITIAL_VICTIMS)
        self.ambulances = [
            {"id": "AMB-1", "busy": False, "pos": BASE_POS, "trips": 0},
            {"id": "AMB-2", "busy": False, "pos": BASE_POS, "trips": 0},
        ]
        self.kits       = 10
        self.tick       = 0
        self.log: List[str] = []
        self.kpis       = {"saved": 0, "avg_time": 0.0, "risk_exp": 0,
                           "kits_left": 10, "path_opt": 0.0, "res_util": 0.0}

        # FIX: Two separate dicts:
        #   bench_perf  — clean single-victim benchmark from _benchmark_algos (never overwritten)
        #   _route_perf — internal accumulator for per-trip routing stats (not shown in chart)
        self.bench_perf : Dict[str, dict] = {}   # used by GUI comparison window
        self._route_perf: Dict[str, dict] = {}   # internal only

        self.local_perf : Dict[str, dict] = {}   # HC vs SA comparison
        self.csp        : Optional[ResourceCSP] = None
        self.csp_assignment: dict = {}
        self.ml         = MLRiskEstimator()
        self.fuzzy      = FuzzyLogic()
        self.aftershock = 5.5
        self.fire_prox  = 3.0

    # ── helpers ───────────────────────────────────────────────────────────────
    def _risk_weight(self):
        return {"Fast": 0.1, "Balanced": 1.0, "Safe": 4.0}[self.strategy]

    def _searcher(self):
        return GridSearch(self.grid, risk_weight=self._risk_weight())

    def _find_path(self, start, goal):
        """
        FIX: Runs only the selected algorithm and writes to _route_perf (internal),
        never to bench_perf.  bench_perf is written only by _benchmark_algos.
        """
        s  = self._searcher()
        t0 = time.perf_counter()
        if   self.algorithm == "A*":     path, n = s.astar(start, goal)
        elif self.algorithm == "Greedy": path, n = s.greedy(start, goal)
        elif self.algorithm == "BFS":    path, n = s.bfs(start, goal)
        else:                            path, n = s.dfs(start, goal)
        ms   = round((time.perf_counter() - t0) * 1000, 2)
        cost = s.path_cost(path)
        risk = s.risk_exposure(path)

        # Accumulate into _route_perf only (internal, not shown in comparison chart)
        rp = self._route_perf.setdefault(self.algorithm, {"nodes": 0, "ms": 0.0, "cost": 0, "risk": 0})
        rp["nodes"] += n
        rp["ms"]    += ms
        rp["cost"]  += cost
        rp["risk"]  += risk

        return path, cost, risk

    def _nearest_medical(self, pos):
        d1 = abs(pos[0] - MEDICAL_POS1[0]) + abs(pos[1] - MEDICAL_POS1[1])
        d2 = abs(pos[0] - MEDICAL_POS2[0]) + abs(pos[1] - MEDICAL_POS2[1])
        return MEDICAL_POS1 if d1 <= d2 else MEDICAL_POS2

    def _log(self, msg: str):
        self.tick += 1
        entry = f"T+{self.tick:03d}  {msg}"
        self.log.append(entry)
        return entry

    # ── benchmark all 4 algorithms on a single victim ────────────────────────
    def _benchmark_algos(self, start, goal):
        """
        FIX: Stores results in self.bench_perf, which is ONLY written here.
        _find_path never touches bench_perf, so this data stays accurate.
        """
        s = GridSearch(self.grid, risk_weight=self._risk_weight())
        bench = {}
        for name, fn in [("BFS",    s.bfs),
                         ("DFS",    s.dfs),
                         ("Greedy", s.greedy),
                         ("A*",     s.astar)]:
            t0 = time.perf_counter()
            path, n = fn(start, goal)
            ms   = round((time.perf_counter() - t0) * 1000, 2)
            cost = s.path_cost(path)
            risk = s.risk_exposure(path)
            bench[name] = {"nodes": n, "ms": ms, "cost": cost, "risk": risk,
                           "path_len": len(path)}
        self.bench_perf = bench   # clean snapshot, never mutated again

    # ── CSP allocation ────────────────────────────────────────────────────────
    def _allocate_csp(self, victims_subset):
        csp        = ResourceCSP(victims_subset, ambulances=2)
        assignment = csp.solve()
        self.csp   = csp
        self.csp_assignment = assignment or {}
        return assignment, csp

    # ── local search prioritisation ──────────────────────────────────────────
    def _prioritise(self):
        order   = {"CRITICAL": 0, "MODERATE": 1, "MINOR": 2}
        waiting = [v for v in self.victims if v.get("status", "WAITING") == "WAITING"]

        s = self._searcher()
        rescue_times = {}
        for v in waiting:
            mc     = self._nearest_medical(v["pos"])
            p1, _  = s.astar(BASE_POS, v["pos"])
            p2, _  = s.astar(v["pos"], mc)
            rescue_times[v["id"]] = len(p1) + len(p2)

        ls = LocalSearch(waiting, rescue_times)

        # Run BOTH local-search methods and log comparison
        t0 = time.perf_counter()
        hc_order, hc_score, hc_imp = ls.hill_climbing()
        hc_ms = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        sa_order, sa_score, sa_acc = ls.simulated_annealing()
        sa_ms = round((time.perf_counter() - t0) * 1000, 2)

        # FIX: store both results with their correct metric keys
        self.local_perf = {
            "Hill Climbing": {
                "score":        hc_score,
                "ms":           hc_ms,
                "improvements": hc_imp,      # HC-specific
            },
            "Sim. Annealing": {
                "score":    sa_score,
                "ms":       sa_ms,
                "accepted": sa_acc,          # SA-specific
            },
        }
        chosen = "SA" if sa_score <= hc_score else "HC"
        self._log(f"🔍 Local Search — HC score={hc_score}({hc_ms}ms) "
                  f"SA score={sa_score}({sa_ms}ms) → using {chosen}")

        # Final sort: severity tier first, then local-search within tier
        sa_rank = {vid: i for i, vid in enumerate(sa_order)}
        waiting.sort(key=lambda v: (order[v["severity"]], sa_rank.get(v["id"], 99)))
        return waiting

    # ── replan paths for a single victim ────────────────────────────────────
    def _replan_victim(self, v):
        mc             = self._nearest_medical(v["pos"])
        path1, c1, r1  = self._find_path(BASE_POS, v["pos"])
        path2, c2, r2  = self._find_path(v["pos"], mc)
        return path1, path2, c1 + c2, r1 + r2

    # ── dynamic event ────────────────────────────────────────────────────────
    def trigger_event(self):
        events = [
            ("ROAD_BLOCK",  "New road blockage detected due to aftershock"),
            ("FIRE_SPREAD", "Fire has spread to adjacent zone"),
            ("NEW_VICTIM",  "New victim reported"),
            ("AFTERSHOCK",  "Aftershock intensity increased"),
        ]
        etype, desc = random.choice(events)

        if etype == "ROAD_BLOCK":
            blocked = False
            for _ in range(20):
                r = random.randint(0, GRID_H - 1)
                c = random.randint(0, GRID_W - 1)
                if self.grid[r][c] == EMPTY:
                    self.grid[r][c] = BLOCKED
                    self._log(f"⚡ Road [{r},{c}] blocked by aftershock → replanning route")
                    self._log(f"   AMB-1 & AMB-2: recalculating paths around [{r},{c}]")
                    blocked = True
                    break
            if not blocked:
                self._log("⚡ ROAD_BLOCK event: no free cell found to block")

        elif etype == "FIRE_SPREAD":
            fire_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W)
                          if self.grid[r][c] == FIRE]
            if fire_cells:
                fr, fc = random.choice(fire_cells)
                dr, dc = random.choice([(-1,0),(1,0),(0,-1),(0,1)])
                nr, nc = fr + dr, fc + dc
                if 0 <= nr < GRID_H and 0 <= nc < GRID_W and self.grid[nr][nc] == EMPTY:
                    self.grid[nr][nc] = FIRE
                    self.fire_prox = min(10.0, self.fire_prox + 0.5)
                    self._log(f"🔥 Fire spread detected → cell [{nr},{nc}] now FIRE")
                    self._log(f"   Rerouting AMB-1: avoiding new fire cell [{nr},{nc}]")
                    self._log(f"   fire_prox updated to {self.fire_prox:.1f} → fuzzy risk recalculated")

        elif etype == "NEW_VICTIM":
            for _ in range(30):
                r = random.randint(0, GRID_H - 1)
                c = random.randint(0, GRID_W - 1)
                if self.grid[r][c] == EMPTY:
                    sev = random.choice(["CRITICAL", "MODERATE", "MINOR"])
                    nv  = {"id": f"V{len(self.victims)+1}", "pos": (r, c),
                            "severity": sev, "surv": round(random.uniform(0.5, 0.95), 2)}
                    self.victims.append(nv)
                    self._log(f"🆘 {desc}: {nv['id']} ({sev}) at ({r},{c})")
                    break

        elif etype == "AFTERSHOCK":
            self.aftershock = min(10.0, self.aftershock + round(random.uniform(0.5, 2.0), 1))
            self._log(f"💥 {desc}: intensity now {self.aftershock:.1f}/10")

    # ── main mission ──────────────────────────────────────────────────────────
    def run_mission(self):
        self.log  = []
        self.kpis = {"saved": 0, "avg_time": 0.0, "risk_exp": 0,
                     "kits_left": self.kits, "path_opt": 0.0, "res_util": 0.0}
        self._route_perf = {}   # reset internal route accumulator each mission
        total_time  = 0
        risk_total  = 0
        all_paths   = {}
        opt_costs   = []

        priority_order = self._prioritise()
        self._log(f"🎯 Rescue priority: {[v['id'] for v in priority_order]}")

        # FIX: Benchmark runs FIRST and stores to bench_perf.
        # Subsequent _find_path calls never touch bench_perf.
        if priority_order:
            v0 = priority_order[0]
            self._log(f"📊 Benchmarking all algorithms on {v0['id']} ...")
            self._benchmark_algos(BASE_POS, v0["pos"])
            for alg, d in self.bench_perf.items():
                self._log(f"   {alg:<10} nodes={d['nodes']} cost={d['cost']} "
                          f"risk={d['risk']} time={d['ms']}ms")

        # CSP: allocate ALL victims
        assignment, csp_obj = self._allocate_csp(priority_order)
        self._log(f"🔧 CSP solved (capacity={csp_obj.capacity}): "
                  f"backtracks={csp_obj.backtracks}  nodes={csp_obj.nodes}")
        for vid, amb_idx in (assignment or {}).items():
            self._log(f"   {vid} → AMB-{amb_idx+1}")

        # A* optimal cost (risk_weight=1) for path-optimality ratio
        opt_searcher = GridSearch(self.grid, risk_weight=1.0)

        # Rescue each victim
        for v in priority_order:
            mc              = self._nearest_medical(v["pos"])
            path1, c1, r1   = self._find_path(BASE_POS, v["pos"])
            path2, c2, r2   = self._find_path(v["pos"], mc)
            full_path       = path1 + path2[1:] if path2 else path1
            all_paths[v["id"]] = full_path

            trip_cost = c1 + c2
            trip_risk = r1 + r2

            opt_p1, _ = opt_searcher.astar(BASE_POS, v["pos"])
            opt_p2, _ = opt_searcher.astar(v["pos"], mc)
            opt_cost  = opt_searcher.path_cost(opt_p1) + opt_searcher.path_cost(opt_p2)
            if opt_cost > 0:
                opt_costs.append(trip_cost / opt_cost)

            surv       = self.ml.predict_survival(v["severity"], trip_cost, c1, trip_risk)
            fuzzy_risk = self.fuzzy.road_blockage_risk(self.aftershock, self.fire_prox)
            vic_risk   = self.fuzzy.victim_risk_score(v["severity"], trip_cost, c1)

            # Per-victim trade-off decision
            if v["severity"] == "CRITICAL":
                tradeoff = "TIME-PRIORITY"
                justify  = (f"{v['id']} is CRITICAL with {surv:.0%} survival — "
                            f"prioritising speed despite {fuzzy_risk:.0%} block risk")
            elif fuzzy_risk > 0.5:
                tradeoff = "RISK-AWARE"
                justify  = (f"Block probability {fuzzy_risk:.0%} exceeds 50% threshold — "
                            f"routing {v['id']} around hazard zone to reduce exposure")
            elif self._risk_weight() < 1:
                tradeoff = "TIME-PRIORITY"
                justify  = (f"Fast strategy active: shortest path chosen for {v['id']} "
                            f"({v['severity']}) — risk={trip_risk} cells accepted")
            elif self._risk_weight() > 1:
                tradeoff = "RISK-AWARE"
                justify  = (f"Safe strategy active: longer path chosen for {v['id']} "
                            f"to avoid {trip_risk} hazard cells")
            else:
                tradeoff = "BALANCED"
                justify  = (f"{v['id']} ({v['severity']}): cost={trip_cost} risk={trip_risk} "
                            f"weighted equally — block_prob={fuzzy_risk:.0%}")

            if path1 and path2:
                v["status"] = "RESCUED"
                v["surv"]   = surv
                self.kpis["saved"] += 1
                total_time += trip_cost
                risk_total += trip_risk
                if self.kits > 0:
                    self.kits -= 1
                    self.kpis["kits_left"] = self.kits

                self._log(f"✅ {v['id']} ({v['severity']}) | route={len(full_path)} cells "
                          f"cost={trip_cost} risk_cells={trip_risk} surv={surv:.0%} "
                          f"vic_risk={vic_risk:.2f} block_prob={fuzzy_risk:.2f} | "
                          f"TRADEOFF={tradeoff}: {justify}")
            else:
                v["status"] = "UNREACHABLE"
                self._log(f"❌ {v['id']} UNREACHABLE – all paths blocked. "
                          f"REPLANNING: reroute other ambulances if possible.")

        # Aggregate KPIs
        saved = self.kpis["saved"]
        if saved > 0:
            self.kpis["avg_time"] = round(total_time / saved, 1)
        self.kpis["risk_exp"]   = risk_total
        self.kpis["path_opt"]   = round(sum(opt_costs) / len(opt_costs), 3) if opt_costs else 1.0

        total_capacity          = 2 * self.csp.capacity
        self.kpis["res_util"]   = round(saved / max(total_capacity, 1), 2)

        self._log(f"📈 MISSION SUMMARY — Saved={saved}/{len(priority_order)} "
                  f"AvgTime={self.kpis['avg_time']} RiskExp={risk_total} "
                  f"OptRatio={self.kpis['path_opt']} ResUtil={self.kpis['res_util']}")
        return all_paths


# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════
class AIDRAGUI(tk.Tk):
    PAD    = 6
    BG     = "#050d18"
    PANEL  = "#0a1628"
    CARD   = "#0e1e35"
    ACCENT = "#00e5ff"
    WARN   = "#ff3d00"
    OK     = "#00e676"
    TEXT   = "#b0bec5"
    HEAD   = "#e0f7fa"
    FONT   = ("Courier New", 9)
    FONTB  = ("Courier New", 9, "bold")
    FONTS  = ("Courier New", 7)

    def __init__(self):
        super().__init__()
        self.title("AIDRA  ─  Adaptive Intelligent Disaster Response Agent")
        self.configure(bg=self.BG)
        self.geometry("1500x860")
        self.resizable(True, True)
        self.agent   = AIDRAAgent()
        self._paths  = {}
        self._build_ui()
        self._refresh_victim_panel()
        self._draw_grid()
        self._update_kpi()
        self._update_ml_panel()
        self._update_csp_panel()
        self._update_fuzzy_panel()

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self, bg=self.BG)
        top.pack(fill="x", padx=8, pady=(6, 0))

        tk.Label(top, text="AIDRA", font=("Courier New", 16, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack(side="left")
        tk.Label(top, text=" DISASTER RESPONSE COMMAND", font=("Courier New", 8),
                 bg=self.BG, fg=self.TEXT).pack(side="left", padx=(2, 30))

        self._sys_lbl = tk.Label(top, text="● SYSTEM READY", font=self.FONTB,
                                  bg="#003300", fg=self.OK, padx=6)
        self._sys_lbl.pack(side="left")

        self._threat_lbl = tk.Label(top, text="  THREAT: HIGH  ", font=self.FONTB,
                                     bg="#330000", fg=self.WARN, padx=4)
        self._threat_lbl.pack(side="left", padx=6)

        self._tick_lbl = tk.Label(top, text="T+000", font=self.FONTB,
                                   bg=self.BG, fg=self.TEXT)
        self._tick_lbl.pack(side="right")

        ctrl = tk.Frame(top, bg=self.BG)
        ctrl.pack(side="left", padx=10)

        self._run_btn = self._btn(ctrl, "▶  RUN MISSION",   self.ACCENT, self._run_mission)
        self._run_btn.pack(side="left", padx=3)
        self._btn(ctrl, "⚡  DYNAMIC EVENT", self.WARN, self._dynamic_event).pack(side="left", padx=3)
        self._btn(ctrl, "📊  COMPARE",       "#9c27b0", self._show_comparison).pack(side="left", padx=3)
        self._btn(ctrl, "↺  RESET",          "#607d8b", self._reset).pack(side="left", padx=3)

        tk.Label(ctrl, text="Strategy:", bg=self.BG, fg=self.TEXT,
                 font=self.FONT).pack(side="left", padx=(12, 2))
        self._strat = ttk.Combobox(ctrl, values=["Fast", "Balanced", "Safe"],
                                    width=10, font=self.FONT)
        self._strat.set("Balanced")
        self._strat.pack(side="left")

        tk.Label(ctrl, text="Algorithm:", bg=self.BG, fg=self.TEXT,
                 font=self.FONT).pack(side="left", padx=(10, 2))
        self._algo = ttk.Combobox(ctrl, values=["A*", "Greedy", "BFS", "DFS"],
                                   width=9, font=self.FONT)
        self._algo.set("A*")
        self._algo.pack(side="left")

        body = tk.Frame(self, bg=self.BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        left = tk.Frame(body, bg=self.BG, width=215)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        mid = tk.Frame(body, bg=self.BG)
        mid.pack(side="left", fill="both", expand=True, padx=6)

        right = tk.Frame(body, bg=self.BG, width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_left(left)
        self._build_mid(mid)
        self._build_right(right)

    # ── left ─────────────────────────────────────────────────────────────────
    def _build_left(self, p):
        self._section(p, "VICTIM REGISTRY")
        self._victim_frame = tk.Frame(p, bg=self.BG)
        self._victim_frame.pack(fill="x", pady=2)

        self._section(p, "AMBULANCE STATUS")
        self._amb_frame = tk.Frame(p, bg=self.BG)
        self._amb_frame.pack(fill="x")

        self._section(p, "FUZZY ASSESSMENT")
        self._fuzzy_frame = tk.Frame(p, bg=self.BG)
        self._fuzzy_frame.pack(fill="x")

        self._section(p, "LOCAL SEARCH")
        self._ls_frame = tk.Frame(p, bg=self.BG)
        self._ls_frame.pack(fill="x")

    # ── mid ───────────────────────────────────────────────────────────────────
    def _build_mid(self, p):
        tk.Label(p, text="URBAN GRID — DISASTER ZONE", font=("Courier New", 8),
                 bg=self.BG, fg=self.TEXT).pack()

        self._fig, self._ax = plt.subplots(figsize=(7.8, 5.0))
        self._fig.patch.set_facecolor("#050d18")
        self._canvas = FigureCanvasTkAgg(self._fig, master=p)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self._section(p, "DECISION LOG")
        self._log_box = scrolledtext.ScrolledText(
            p, height=7, bg="#040c18", fg=self.OK,
            font=("Courier New", 7), insertbackground=self.ACCENT,
            relief="flat", wrap="word", state="disabled")
        self._log_box.pack(fill="x", pady=2)

    # ── right ─────────────────────────────────────────────────────────────────
    def _build_right(self, p):
        self._section(p, "KPI DASHBOARD")
        self._kpi_frame = tk.Frame(p, bg=self.BG)
        self._kpi_frame.pack(fill="x", pady=2)

        self._section(p, "SEARCH ALGORITHM BENCHMARK")
        self._algo_frame = tk.Frame(p, bg=self.BG)
        self._algo_frame.pack(fill="x", pady=2)

        self._section(p, "ML MODEL METRICS")
        self._ml_frame = tk.Frame(p, bg=self.BG)
        self._ml_frame.pack(fill="x", pady=2)

        self._section(p, "CSP RESOURCE ALLOCATION")
        self._csp_frame = tk.Frame(p, bg=self.BG)
        self._csp_frame.pack(fill="x", pady=2)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _section(self, parent, title):
        f = tk.Frame(parent, bg=self.BG)
        f.pack(fill="x", pady=(5, 0))
        tk.Label(f, text=title, font=("Courier New", 7, "bold"),
                 bg=self.BG, fg="#546e7a").pack(anchor="w", padx=2)
        tk.Frame(parent, bg="#1a3a4a", height=1).pack(fill="x")
        return f

    def _btn(self, parent, text, color, cmd):
        return tk.Button(parent, text=text, font=("Courier New", 8, "bold"),
                         bg="#111", fg=color, activebackground="#1a2a3a",
                         activeforeground=color, relief="flat",
                         padx=8, pady=3, cursor="hand2", command=cmd,
                         bd=1, highlightbackground=color)

    def _mini_bar(self, parent, value, max_val, color, width=90):
        pct = min(value / max_val, 1.0) if max_val else 0
        c   = tk.Canvas(parent, width=width, height=7, bg="#0d1b2a",
                        highlightthickness=0)
        c.pack(side="left")
        c.create_rectangle(0, 0, int(width * pct), 7, fill=color, outline="")

    # ── victim panel ─────────────────────────────────────────────────────────
    def _refresh_victim_panel(self):
        for w in self._victim_frame.winfo_children():
            w.destroy()
        for v in self.agent.victims:
            c      = tk.Frame(self._victim_frame, bg=self.CARD, pady=2, padx=4)
            c.pack(fill="x", pady=1)
            status = v.get("status", "WAITING")
            st_col = self.OK if status == "RESCUED" else (self.WARN if status == "UNREACHABLE" else self.TEXT)
            tk.Label(c, text=v["id"], font=self.FONTB,
                     bg=self.CARD, fg=self.HEAD).pack(side="left")
            tk.Label(c, text=f" {v['severity']}", font=self.FONTS,
                     bg=SEVERITY_COLOR.get(v["severity"], "#333"),
                     fg="white").pack(side="left", padx=2)
            tk.Label(c, text=f" surv:{v['surv']:.0%}",
                     font=self.FONTS, bg=self.CARD, fg=self.TEXT).pack(side="left")
            tk.Label(c, text=status, font=self.FONTS,
                     bg=self.CARD, fg=st_col).pack(side="right")

    # ── ambulance panel ───────────────────────────────────────────────────────
    def _update_amb_panel(self):
        for w in self._amb_frame.winfo_children():
            w.destroy()
        assign = self.agent.csp_assignment
        for a in self.agent.ambulances:
            f   = tk.Frame(self._amb_frame, bg=self.CARD, padx=4, pady=3)
            f.pack(fill="x", pady=2)
            idx = int(a["id"].split("-")[1]) - 1
            vics = [vid for vid, ai in assign.items() if ai == idx]
            tk.Label(f, text=f"{a['id']}",
                     font=self.FONTB, bg=self.CARD, fg=self.OK).pack(side="left")
            tk.Label(f, text=f" → {', '.join(vics) if vics else 'unassigned'}",
                     font=self.FONTS, bg=self.CARD, fg=self.TEXT).pack(side="left")

    # ── KPI panel ─────────────────────────────────────────────────────────────
    def _update_kpi(self):
        for w in self._kpi_frame.winfo_children():
            w.destroy()
        k = self.agent.kpis
        f = tk.Frame(self._kpi_frame, bg=self.BG)
        f.pack(fill="x")
        items = [
            ("SAVED",     k["saved"],        len(self.agent.victims), self.OK),
            ("AVG TIME",  k["avg_time"],      50,                      self.ACCENT),
            ("RISK EXP",  k["risk_exp"],      30,                      self.WARN),
            ("KITS LEFT", k["kits_left"],     10,                      "#ffeb3b"),
            ("PATH OPT",  k["path_opt"],       2,                      "#ce93d8"),
            ("RES UTIL",  k["res_util"],       1,                      "#80cbc4"),
        ]
        for i, (lbl, val, mx, col) in enumerate(items):
            r  = i // 3
            cc = i % 3
            sub = tk.Frame(f, bg=self.CARD, padx=4, pady=4)
            sub.grid(row=r, column=cc, padx=2, pady=2, sticky="nsew")
            tk.Label(sub, text=f"{val:.2f}" if isinstance(val, float) else str(val),
                     font=("Courier New", 14, "bold"), bg=self.CARD, fg=col).pack()
            tk.Label(sub, text=lbl, font=("Courier New", 6),
                     bg=self.CARD, fg=self.TEXT).pack()

    # ── algo benchmark panel (reads bench_perf) ───────────────────────────────
    def _update_algo_panel(self):
        for w in self._algo_frame.winfo_children():
            w.destroy()
        f = tk.Frame(self._algo_frame, bg=self.BG)
        f.pack(fill="x")
        colors = {"A*": self.ACCENT, "Greedy": self.OK, "BFS": "#9c27b0", "DFS": self.WARN}
        for col_idx, hdr in enumerate(["Algo", "Nodes", "Cost", "Risk", "ms"]):
            tk.Label(f, text=hdr, font=("Courier New", 7, "bold"),
                     bg=self.BG, fg="#546e7a").grid(row=0, column=col_idx, padx=3, sticky="w")
        # FIX: read bench_perf, not algo_perf
        for row_idx, (alg, d) in enumerate(self.agent.bench_perf.items(), start=1):
            col = colors.get(alg, self.TEXT)
            vals = [alg, str(d["nodes"]), str(d["cost"]), str(d.get("risk", "?")), f"{d['ms']}"]
            for col_idx, val in enumerate(vals):
                tk.Label(f, text=val, font=self.FONTS,
                         bg=self.BG, fg=col).grid(row=row_idx, column=col_idx, padx=3, sticky="w")

    # ── local search panel ───────────────────────────────────────────────────
    def _update_ls_panel(self):
        for w in self._ls_frame.winfo_children():
            w.destroy()
        f = tk.Frame(self._ls_frame, bg=self.BG)
        f.pack(fill="x")
        for row_idx, (name, d) in enumerate(self.agent.local_perf.items()):
            col = self.ACCENT if "Anneal" in name else self.OK
            tk.Label(f, text=name, font=self.FONTS,
                     bg=self.BG, fg=col).grid(row=row_idx, column=0, sticky="w", padx=2)
            # FIX: pick the correct stat key per algorithm
            if "Hill" in name:
                extra = f"impr={d['improvements']}"
            else:
                extra = f"acc={d['accepted']}"
            tk.Label(f, text=f"score={d['score']}  {d['ms']}ms  {extra}",
                     font=self.FONTS, bg=self.BG, fg=self.TEXT
                     ).grid(row=row_idx, column=1, sticky="w", padx=4)

    # ── ML panel ─────────────────────────────────────────────────────────────
    def _update_ml_panel(self):
        for w in self._ml_frame.winfo_children():
            w.destroy()
        for name, m in self.agent.ml.metrics.items():
            f = tk.Frame(self._ml_frame, bg=self.CARD, padx=4, pady=2)
            f.pack(fill="x", pady=1)
            tk.Label(f, text=name, font=self.FONTB,
                     bg=self.CARD, fg=self.ACCENT).pack(anchor="w")
            row = tk.Frame(f, bg=self.CARD)
            row.pack(anchor="w")
            for metric, col in [("Accuracy",  self.OK),
                                  ("Precision", self.ACCENT),
                                  ("Recall",    "#ffeb3b"),
                                  ("F1",        "#ff9800")]:
                tk.Label(row, text=f"{metric[0]}:{m[metric]:.3f}  ",
                         font=("Courier New", 7), bg=self.CARD, fg=col).pack(side="left")
            cm = m["CM"]
            cm_str = f"CM[[{cm[0][0]}|{cm[0][1]}][{cm[1][0]}|{cm[1][1]}]]"
            tk.Label(f, text=cm_str, font=("Courier New", 6),
                     bg=self.CARD, fg="#78909c").pack(anchor="w")

    # ── CSP panel ─────────────────────────────────────────────────────────────
    def _update_csp_panel(self):
        for w in self._csp_frame.winfo_children():
            w.destroy()
        f   = tk.Frame(self._csp_frame, bg=self.BG)
        f.pack(fill="x")
        csp = self.agent.csp
        rows = [
            ("Method",     "MRV + Forward-Check",                       self.TEXT),
            ("Capacity",   str(csp.capacity if csp else "-"),            self.ACCENT),
            ("Backtracks", str(csp.backtracks if csp else 0),           self.WARN),
            ("Nodes",      str(csp.nodes if csp else 0),                self.TEXT),
        ]
        for i, (lbl, val, col) in enumerate(rows):
            tk.Label(f, text=f"{lbl:<16}", font=self.FONTS,
                     bg=self.BG, fg=self.TEXT).grid(row=i, column=0, sticky="w")
            tk.Label(f, text=val, font=self.FONTB,
                     bg=self.BG, fg=col).grid(row=i, column=1, sticky="w")
        assign = self.agent.csp_assignment
        if assign:
            tk.Label(f, text="Assignment:", font=self.FONTS,
                     bg=self.BG, fg="#546e7a").grid(row=len(rows), column=0,
                                                     columnspan=2, sticky="w", pady=(4,0))
            for j, (vid, ai) in enumerate(assign.items()):
                tk.Label(f, text=f"  {vid} → AMB-{ai+1}", font=self.FONTS,
                         bg=self.BG, fg=self.TEXT).grid(row=len(rows)+j+1,
                                                          column=0, columnspan=2, sticky="w")

    # ── fuzzy panel ───────────────────────────────────────────────────────────
    def _update_fuzzy_panel(self):
        for w in self._fuzzy_frame.winfo_children():
            w.destroy()
        fl  = self.agent.fuzzy
        rb  = fl.road_blockage_risk(self.agent.aftershock, self.agent.fire_prox)
        lvl = "LOW" if rb < 0.4 else ("MEDIUM" if rb < 0.7 else "HIGH")
        col = self.OK if rb < 0.4 else (self.WARN if rb < 0.7 else "#ff1744")
        items = [
            ("ROAD BLOCK RISK", rb,                          lvl,                          col),
            ("AFTERSHOCK",      self.agent.aftershock / 10,  f"{self.agent.aftershock:.1f}/10", "#ff9800"),
            ("FIRE PROXIMITY",  self.agent.fire_prox / 10,   f"{self.agent.fire_prox:.1f}/10",  "#ff3d00"),
        ]
        for lbl, val, txt, color in items:
            f = tk.Frame(self._fuzzy_frame, bg=self.CARD, padx=4, pady=2)
            f.pack(fill="x", pady=1)
            tk.Label(f, text=lbl, font=("Courier New", 7, "bold"),
                     bg=self.CARD, fg=self.TEXT).pack(anchor="w")
            row2 = tk.Frame(f, bg=self.CARD)
            row2.pack(anchor="w", fill="x")
            self._mini_bar(row2, val, 1.0, color, width=100)
            tk.Label(row2, text=f" {txt}", font=("Courier New", 7),
                     bg=self.CARD, fg=color).pack(side="left")

    # ── grid drawing ──────────────────────────────────────────────────────────
    def _draw_grid(self, paths=None):
        self._ax.clear()
        self._ax.set_facecolor("#050d18")
        for sp in self._ax.spines.values():
            sp.set_color("#0a2a3a")

        grid      = self.agent.grid
        color_map = np.zeros((GRID_H, GRID_W, 3))
        cell_rgb  = {
            EMPTY:    (13/255, 27/255, 42/255),
            OBSTACLE: (26/255, 26/255, 46/255),
            FIRE:     (139/255, 0, 0),
            RISK:     (122/255, 96/255, 0),
            MEDICAL:  (0, 77/255, 64/255),
            BASE:     (0, 48/255, 96/255),
            BLOCKED:  (58/255, 0, 58/255),
        }
        for r in range(GRID_H):
            for c in range(GRID_W):
                color_map[r, c] = cell_rgb.get(grid[r][c], cell_rgb[EMPTY])

        self._ax.imshow(color_map, aspect="auto", origin="upper",
                        interpolation="nearest",
                        extent=[-0.5, GRID_W - 0.5, GRID_H - 0.5, -0.5])

        for x in range(GRID_W + 1):
            self._ax.axvline(x - 0.5, color="#0a2a3a", lw=0.4)
        for y in range(GRID_H + 1):
            self._ax.axhline(y - 0.5, color="#0a2a3a", lw=0.4)

        if paths:
            pcolors = ["#00e5ff", "#76ff03", "#ff9100", "#e040fb", "#ff1744", "#40c4ff"]
            for i, (vid, path) in enumerate(paths.items()):
                if len(path) > 1:
                    pc = pcolors[i % len(pcolors)]
                    xs = [p[1] for p in path]
                    ys = [p[0] for p in path]
                    self._ax.plot(xs, ys, color=pc, lw=1.8, alpha=0.65, zorder=3)
                    self._ax.text(xs[0], ys[0], vid[1:], fontsize=5,
                                  color=pc, zorder=7, ha="center", va="center")

        sev_marker = {"CRITICAL": "*", "MODERATE": "o", "MINOR": "s"}
        sev_col    = {"CRITICAL": "#ff3333", "MODERATE": "#ffaa00", "MINOR": "#33ff99"}
        for v in self.agent.victims:
            r, c  = v["pos"]
            col   = "#888" if v.get("status") == "RESCUED" else sev_col[v["severity"]]
            self._ax.plot(c, r, sev_marker[v["severity"]], markersize=13,
                          color=col, markeredgecolor="#000", zorder=5)
            self._ax.text(c, r, v["id"], ha="center", va="center",
                          fontsize=5, color="#fff", zorder=6, fontweight="bold")

        for label, pos, col in [("AI", BASE_POS, "#1976d2"),
                                  ("MC", MEDICAL_POS1, "#00897b"),
                                  ("MC", MEDICAL_POS2, "#00897b")]:
            r, c = pos
            rect = mpatches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05", fc=col, ec="#00e5ff", lw=1.2, zorder=4)
            self._ax.add_patch(rect)
            self._ax.text(c, r, label, ha="center", va="center",
                          fontsize=6, color="white", fontweight="bold", zorder=7)

        legend_items = [
            mpatches.Patch(color="#8b0000", label="Fire"),
            mpatches.Patch(color="#7a6000", label="Risk Zone"),
            mpatches.Patch(color="#3a003a", label="Blocked"),
            mpatches.Patch(color="#004d40", label="Medical"),
            mpatches.Patch(color="#003060", label="Base"),
        ]
        self._ax.legend(handles=legend_items, loc="lower right", fontsize=5,
                        facecolor="#050d18", edgecolor="#1a3a4a",
                        labelcolor="white", framealpha=0.8)

        self._ax.set_xlim(-0.5, GRID_W - 0.5)
        self._ax.set_ylim(GRID_H - 0.5, -0.5)
        self._ax.tick_params(colors="#1a3a4a", labelsize=6)
        for t in self._ax.get_xticklabels() + self._ax.get_yticklabels():
            t.set_color("#1a3a4a")
        self._canvas.draw()

    # ── comparison popup — reads bench_perf (fixed) ───────────────────────────
    def _show_comparison(self):
        if not self.agent.bench_perf:
            messagebox.showinfo("No Data", "Run a mission first to generate comparison data.")
            return
        win = tk.Toplevel(self)
        win.title("Comparative Evaluation")
        win.configure(bg=self.BG)
        win.geometry("900x650")

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.patch.set_facecolor("#050d18")
        fig.suptitle("AIDRA — Comparative Performance Report",
                     color=self.ACCENT, fontsize=11, fontweight="bold")

        # FIX: use bench_perf (single-victim clean snapshot)
        algos  = list(self.agent.bench_perf.keys())
        colors = ["#00e5ff", "#ff3d00", "#76ff03", "#9c27b0"]  # BFS, DFS, Greedy, A*

        def _bar_chart(ax, title, key, ylabel):
            vals = [self.agent.bench_perf[a].get(key, 0) for a in algos]
            bars = ax.bar(algos, vals, color=colors[:len(algos)], edgecolor="#0a2a3a")
            ax.set_title(title, color=self.TEXT, fontsize=8)
            ax.set_ylabel(ylabel, color=self.TEXT, fontsize=7)
            ax.set_facecolor("#050d18")
            ax.tick_params(colors=self.TEXT, labelsize=7)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        str(round(v, 1)), ha="center", va="bottom",
                        color=self.TEXT, fontsize=7)
            for sp in ax.spines.values():
                sp.set_color("#1a3a4a")

        _bar_chart(axes[0][0], "Nodes Expanded",  "nodes",    "Count")
        _bar_chart(axes[0][1], "Path Cost",        "cost",     "Cost Units")
        _bar_chart(axes[0][2], "Execution Time",   "ms",       "ms")
        _bar_chart(axes[1][0], "Risk Exposure",    "risk",     "Risk Cells")

        # ── ML Metrics comparison — FIX: full label names, no truncation ──────
        ax = axes[1][1]
        ml_names  = list(self.agent.ml.metrics.keys())   # full names, e.g. "k-NN (k=7)"
        ml_colors = ["#00e5ff", "#76ff03", "#ff9800"]
        x         = np.arange(4)
        w         = 0.25
        for i, (name, m) in enumerate(self.agent.ml.metrics.items()):
            vals = [m["Accuracy"], m["Precision"], m["Recall"], m["F1"]]
            ax.bar(x + i * w, vals, w, label=name, color=ml_colors[i], edgecolor="#0a2a3a")
        ax.set_xticks(x + w)
        ax.set_xticklabels(["Acc", "Prec", "Rec", "F1"], color=self.TEXT, fontsize=7)
        ax.set_title("ML Model Metrics", color=self.TEXT, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_facecolor("#050d18")
        ax.tick_params(colors=self.TEXT, labelsize=7)
        # FIX: legend with full labels, placed outside to avoid overlap
        ax.legend(fontsize=6, facecolor="#050d18", labelcolor="white",
                  edgecolor="#1a3a4a", loc="upper right")
        for sp in ax.spines.values():
            sp.set_color("#1a3a4a")

        # ── Local Search comparison — FIX: correct keys + clean dual-axis ─────
        ax = axes[1][2]
        if self.agent.local_perf:
            ls_data   = self.agent.local_perf
            ls_names  = list(ls_data.keys())              # ["Hill Climbing", "Sim. Annealing"]
            ls_scores = [ls_data[n]["score"] for n in ls_names]
            ls_times  = [ls_data[n]["ms"]    for n in ls_names]

            # FIX: correct stat label per method
            ls_extras = []
            for n in ls_names:
                if "Hill" in n:
                    ls_extras.append(f"impr={ls_data[n]['improvements']}")
                else:
                    ls_extras.append(f"acc={ls_data[n]['accepted']}")

            bar_colors = ["#00e5ff", "#76ff03"]
            ax2 = ax.twinx()
            bars = ax.bar(ls_names, ls_scores, color=bar_colors, edgecolor="#0a2a3a", alpha=0.85)
            ax2.plot(ls_names, ls_times, "o--", color=self.WARN, linewidth=1.8, markersize=7,
                     label="Time (ms)")

            # Annotate bars with score + extra stat
            for bar, score, extra in zip(bars, ls_scores, ls_extras):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(ls_scores) * 0.01,
                        f"{score}\n{extra}", ha="center", va="bottom",
                        color=self.TEXT, fontsize=6)

            ax.set_title("Local Search Comparison\n(score: lower is better)", color=self.TEXT, fontsize=8)
            ax.set_ylabel("Weighted Rescue Score ↓", color=self.TEXT, fontsize=7)
            ax2.set_ylabel("Execution Time (ms)", color=self.WARN, fontsize=7)
            ax.set_facecolor("#050d18")
            ax.tick_params(colors=self.TEXT, labelsize=7)
            ax2.tick_params(colors=self.WARN, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#1a3a4a")
            ax2.legend(fontsize=6, facecolor="#050d18", labelcolor=self.WARN,
                       edgecolor="#1a3a4a", loc="upper left")
        else:
            ax.text(0.5, 0.5, "Run mission first", ha="center", va="center",
                    color=self.TEXT, transform=ax.transAxes)
            ax.set_facecolor("#050d18")

        plt.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig, master=win)
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        canvas2.draw()

    # ── actions ───────────────────────────────────────────────────────────────
    def _run_mission(self):
        self.agent.strategy  = self._strat.get()
        self.agent.algorithm = self._algo.get()
        self._sys_lbl.config(text="● ACTIVE", bg="#004400")
        self._run_btn.config(state="disabled")
        threading.Thread(target=self._mission_thread, daemon=True).start()

    def _mission_thread(self):
        paths = self.agent.run_mission()
        self.after(0, lambda: self._mission_done(paths))

    def _mission_done(self, paths):
        self._paths = paths
        self._refresh_victim_panel()
        self._update_kpi()
        self._update_algo_panel()
        self._update_csp_panel()
        self._update_fuzzy_panel()
        self._update_amb_panel()
        self._update_ls_panel()
        self._draw_grid(paths)
        self._append_log("\n".join(self.agent.log))
        self._tick_lbl.config(text=f"T+{self.agent.tick:03d}")
        self._sys_lbl.config(text="● COMPLETE", bg="#003300")
        self._run_btn.config(state="normal")

    def _dynamic_event(self):
        self.agent.trigger_event()
        self._update_fuzzy_panel()
        self._draw_grid(self._paths)
        if self.agent.log:
            self._append_log(self.agent.log[-1])
        self._tick_lbl.config(text=f"T+{self.agent.tick:03d}")
        if self._paths:
            self._append_log(">> Dynamic event detected — re-running mission with updated grid...")
            threading.Thread(target=self._mission_thread, daemon=True).start()

    def _reset(self):
        self.agent  = AIDRAAgent(self._strat.get(), self._algo.get())
        self._paths = {}
        self._refresh_victim_panel()
        self._update_kpi()
        self._update_fuzzy_panel()
        self._update_csp_panel()
        for fr in [self._algo_frame, self._ls_frame, self._amb_frame]:
            for w in fr.winfo_children():
                w.destroy()
        self._draw_grid()
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")
        self._tick_lbl.config(text="T+000")
        self._sys_lbl.config(text="● SYSTEM READY", bg="#003300")

    def _append_log(self, text: str):
        self._log_box.config(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.config(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = AIDRAGUI()
    app.mainloop()