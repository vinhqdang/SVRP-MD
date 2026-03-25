# SVRP-MD: Developer Implementation Plan

**Project:** Stochastic VRP with Mixed/Signed Demands — Branch-and-Cut Solver  
**Author:** Quang-Vinh Dang, British University Vietnam  
**Status:** Active development  
**Stack:** Python (ALNS heuristic) + C++ (branch-and-cut) + CPLEX or Gurobi (LP/MIP solver)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Phase 0 — Environment Setup](#3-phase-0--environment-setup)
4. [Phase 1 — Data Layer](#4-phase-1--data-layer)
5. [Phase 2 — Exact Route Oracle (Algorithm 1)](#5-phase-2--exact-route-oracle-algorithm-1)
6. [Phase 3 — Lower Bound Engine](#6-phase-3--lower-bound-engine)
7. [Phase 4 — ALNS Warm-Start Heuristic](#7-phase-4--alns-warm-start-heuristic)
8. [Phase 5 — Branch-and-Cut Solver](#8-phase-5--branch-and-cut-solver)
9. [Phase 6 — Instance Generation](#9-phase-6--instance-generation)
10. [Phase 7 — Evaluation Pipeline](#10-phase-7--evaluation-pipeline)
11. [Testing Strategy](#11-testing-strategy)
12. [Performance Benchmarks and Acceptance Criteria](#12-performance-benchmarks-and-acceptance-criteria)
13. [Open Theoretical Tasks for the Developer](#13-open-theoretical-tasks-for-the-developer)
14. [Dependency Map](#14-dependency-map)

---

## 1. Project Overview

### What we are building

A two-stage solver for the **SVRP-MD** — a vehicle routing problem where customer demands are **signed real numbers**:

- `d_i > 0` → pickup (truck loads goods, running load increases)
- `d_i < 0` → delivery (truck unloads goods, running load decreases)

This breaks every assumption in the classical VRPSD literature. The running load along a route is a **random walk**, not a monotone process. The penalty is:

```
E[ Σ_k Σ_j max(0, L_j^k − Q) ]
```

where `L_j^k` is the running load after serving position `j` on route `k`. The full objective is:

```
min  total_distance + c_f * num_vehicles + c_p * E[trajectory_penalty]
```

### What survives from the VRPSD literature

| Component | Status | Source |
|---|---|---|
| Route cuts | **Always valid** | Ota-Fukasawa Theorem 2 |
| Set cuts (Jensen) | **Always valid** | Proposition 2 (convexity) |
| Superadditivity | **Fails** | Proposition 1 counterexample |
| Path cuts / E-cuts | **Invalid as-is** | Superadditivity required |
| PR-cuts (prefix-risk) | **Conditional** | Requires Theorem D proof |

### Four theorem targets (paper spine)

- **Theorem A** — Route cuts valid under position-based disaggregation *(proved)*
- **Theorem B** — Superadditivity fails *(proved — Proposition 1)*
- **Theorem C** — Jensen bound is a valid set-cut lower bound *(proved)*
- **Theorem D** — PR-cut activation function exists *(open — main theoretical task)*

---

## 2. Repository Structure

```
svrp_md/
│
├── data/
│   ├── raw/                    # Original CVRP/VRPSD benchmark instances
│   ├── generated/              # Signed-demand instances produced by Phase 6
│   └── scenarios/              # Scenario files (.npy or .json per instance)
│
├── src/
│   ├── core/
│   │   ├── instance.py         # Instance dataclass (graph, demands, scenarios)
│   │   ├── route.py            # Route dataclass + orientation logic
│   │   └── solution.py         # Solution dataclass (list of routes + cost)
│   │
│   ├── oracle/
│   │   └── route_eval.py       # Algorithm 1 — exact route oracle (Phase 2)
│   │
│   ├── bounds/
│   │   ├── jensen.py           # Jensen lower bound + partition DP (Phase 3)
│   │   └── dispersion.py       # Dispersion-aware bounds (Phase 3, research)
│   │
│   ├── alns/
│   │   ├── alns.py             # ALNS controller (Phase 4)
│   │   ├── destroy.py          # Destroy operators (Phase 4)
│   │   ├── repair.py           # Repair operators (Phase 4)
│   │   └── scoring.py          # Acceptance + scoring logic (Phase 4)
│   │
│   ├── bnc/                    # Branch-and-cut (Phase 5, C++ or Python+solver)
│   │   ├── master.py           # Master problem formulation
│   │   ├── cuts/
│   │   │   ├── route_cuts.py   # Route cut separation
│   │   │   ├── set_cuts.py     # Jensen set cut separation
│   │   │   └── pr_cuts.py      # PR-cut separation (conditional on Thm D)
│   │   └── callback.py         # Solver callback dispatcher
│   │
│   ├── instance_gen/
│   │   ├── generator.py        # Instance generation protocol (Phase 6)
│   │   └── metrics.py          # Instance-level metric computation
│   │
│   └── eval/
│       ├── runner.py           # Batch experiment runner (Phase 7)
│       ├── metrics.py          # Solution-level metrics
│       └── report.py           # Results aggregation and tables
│
├── tests/
│   ├── test_oracle.py
│   ├── test_bounds.py
│   ├── test_alns.py
│   ├── test_cuts.py
│   └── fixtures/               # Small hand-crafted test instances
│
├── experiments/
│   ├── configs/                # YAML experiment configs
│   └── results/                # Raw result CSVs + logs
│
├── notebooks/
│   └── analysis.ipynb          # Result visualisation
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 3. Phase 0 — Environment Setup

**Goal:** Reproducible environment, CI passing, dependency versions locked.

### 3.1 Python environment

```bash
conda create -n svrp_md python=3.11
conda activate svrp_md
pip install numpy scipy networkx pandas matplotlib pyyaml pytest tqdm
```

For ALNS framework:

```bash
pip install alns  # https://github.com/N-Wouda/ALNS
```

For solver interface (choose one):

```bash
pip install gurobipy          # Gurobi (preferred — free academic license)
# OR
pip install cplex             # CPLEX
# OR
pip install mip               # CBC fallback for open-source runs
```

For CVRPSEP (capacity cut separation):

```bash
# Clone and build CVRPSEP, then wrap via ctypes or subprocess
git clone https://github.com/lysgaard/cvrpsep
cd cvrpsep && make
```

### 3.2 Solver version pins

| Library | Min version | Notes |
|---|---|---|
| Python | 3.11 | f-strings, match syntax |
| NumPy | 1.26 | vectorised scenario ops |
| Gurobi | 11.0 | callback API stable |
| CPLEX | 22.1 | if using CPLEX |
| networkx | 3.2 | graph utilities |

### 3.3 CI configuration

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short
```

### 3.4 Acceptance criteria for Phase 0

- [ ] `pytest tests/` passes with 0 failures on a clean environment
- [ ] `python -c "import gurobipy; print(gurobipy.gurobi.version())"` prints a version
- [ ] `python -c "from src.core.instance import Instance"` imports without error

---

## 4. Phase 1 — Data Layer

**Goal:** A clean, typed data model that all downstream components consume.

### 4.1 `Instance` dataclass

```python
# src/core/instance.py
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Instance:
    name: str
    n_customers: int            # |V+|
    capacity: float             # Q
    initial_load: float         # L_0 (default 0.0)
    cost_penalty: float         # c_p  (penalty multiplier)
    cost_fleet: float           # c_f  (per-vehicle cost)
    
    # Graph
    distance: np.ndarray        # shape (n+1, n+1), index 0 = depot
    
    # Demands — shape (N_scenarios, n_customers)
    # demand[xi, i] = realisation of d_i under scenario xi
    demand: np.ndarray
    prob: np.ndarray            # shape (N_scenarios,), sum to 1.0
    mean_demand: np.ndarray     # shape (n_customers,)  pre-computed
    var_demand: np.ndarray      # shape (n_customers,)  pre-computed

    def __post_init__(self):
        assert self.demand.shape[0] == len(self.prob)
        assert abs(self.prob.sum() - 1.0) < 1e-9
        self.mean_demand = (self.prob[:, None] * self.demand).sum(axis=0)
        self.var_demand  = (self.prob[:, None] * self.demand**2).sum(axis=0) \
                           - self.mean_demand**2

    @property
    def n_scenarios(self): return len(self.prob)
    
    @property
    def n_nodes(self): return self.n_customers + 1  # includes depot
```

### 4.2 `Route` dataclass

```python
# src/core/route.py
from dataclasses import dataclass
from typing import List

@dataclass
class Route:
    customers: List[int]    # ordered list of customer indices (1-based)
    # Note: orientation is determined at evaluation time (best of two)
    
    def reverse(self) -> 'Route':
        return Route(self.customers[::-1])
    
    def __len__(self): return len(self.customers)
    def __iter__(self): return iter(self.customers)
```

### 4.3 `Solution` dataclass

```python
# src/core/solution.py
from dataclasses import dataclass, field
from typing import List
from .route import Route

@dataclass
class Solution:
    routes: List[Route]
    total_distance: float = 0.0
    expected_penalty: float = 0.0
    n_vehicles: int = 0
    
    # Set after evaluation
    objective: float = float('inf')
    
    def compute_objective(self, c_f: float, c_p: float):
        self.n_vehicles = len(self.routes)
        self.objective = self.total_distance + c_f * self.n_vehicles \
                         + c_p * self.expected_penalty
        return self.objective
```

### 4.4 Instance I/O

```python
# src/core/instance.py  (continued)
import json, numpy as np

def load_instance(path: str) -> Instance:
    """Load from JSON format (see Section 6 for schema)."""
    with open(path) as f:
        d = json.load(f)
    return Instance(
        name=d['name'],
        n_customers=d['n_customers'],
        capacity=d['capacity'],
        initial_load=d.get('initial_load', 0.0),
        cost_penalty=d['cost_penalty'],
        cost_fleet=d['cost_fleet'],
        distance=np.array(d['distance']),
        demand=np.array(d['demand']),     # shape (N_scenarios, n_customers)
        prob=np.array(d['prob']),
    )
```

### 4.5 Acceptance criteria for Phase 1

- [ ] Load all 270 Jabali instances (adapted) without error
- [ ] `instance.mean_demand` matches hand-computed values for 3 test instances
- [ ] `Instance.__post_init__` raises `AssertionError` on malformed probability vector
- [ ] Round-trip: `save(load(path)) == original` for all fields

---

## 5. Phase 2 — Exact Route Oracle (Algorithm 1)

**Goal:** Given a route and scenarios, compute the exact expected trajectory penalty. This is the single most important function in the codebase — everything else calls it.

### 5.1 Core implementation

```python
# src/oracle/route_eval.py
import numpy as np
from src.core.instance import Instance
from src.core.route import Route

def eval_route_directed(
    customers: list[int],
    instance: Instance,
    L0: float = None
) -> float:
    """
    Compute expected trajectory penalty for a DIRECTED route.
    
    Time: O(N * t) where N = n_scenarios, t = len(customers).
    
    Parameters
    ----------
    customers : ordered list of customer indices (1-based)
    instance  : Instance with demand[xi, i] and prob[xi]
    L0        : initial load (defaults to instance.initial_load)
    
    Returns
    -------
    float : E[ Σ_j max(0, L_j - Q) ]
    """
    if L0 is None:
        L0 = instance.initial_load
    
    Q    = instance.capacity
    D    = instance.demand[:, [c - 1 for c in customers]]  # (N, t)
    prob = instance.prob                                    # (N,)
    
    # Prefix sums of demands per scenario: shape (N, t)
    prefix = np.cumsum(D, axis=1)
    
    # Running load at each position: shape (N, t)
    load = L0 + prefix
    
    # Overload at each position: shape (N, t)
    overload = np.maximum(0.0, load - Q)
    
    # Per-scenario total penalty: shape (N,)
    penalty_per_scenario = overload.sum(axis=1)
    
    # Expected penalty
    return float(prob @ penalty_per_scenario)


def eval_route(route: Route, instance: Instance, L0: float = None) -> float:
    """
    Compute expected trajectory penalty for an UNDIRECTED route.
    Evaluates both orientations and returns the minimum.
    
    Time: O(N * t).
    """
    fwd = eval_route_directed(route.customers, instance, L0)
    bwd = eval_route_directed(route.customers[::-1], instance, L0)
    return min(fwd, bwd)


def eval_solution(solution, instance: Instance) -> float:
    """Compute total expected trajectory penalty across all routes."""
    return sum(eval_route(r, instance) for r in solution.routes)
```

### 5.2 Distance computation helper

```python
def route_distance(route: Route, instance: Instance) -> float:
    """Total travel distance for a route (including depot legs)."""
    seq = [0] + route.customers + [0]  # depot = index 0
    return sum(instance.distance[seq[i]][seq[i+1]] for i in range(len(seq)-1))
```

### 5.3 Unit tests

```python
# tests/test_oracle.py

def test_deterministic_single_customer():
    """Single pickup customer: penalty = max(0, L0 + d - Q)."""
    inst = make_trivial_instance(d=[[9.0]], prob=[1.0], Q=10, L0=3)
    route = Route([1])
    # L1 = 3 + 9 = 12 > 10 → penalty = 2
    assert abs(eval_route(route, inst) - 2.0) < 1e-9

def test_deterministic_delivery_then_pickup():
    """Proposition 1 counterexample: should give 1.0, not 7.0."""
    inst = make_trivial_instance(d=[[-6.0, 9.0]], prob=[1.0], Q=10, L0=8)
    route_fwd = Route([1, 2])   # delivery then pickup
    route_bwd = Route([2, 1])   # pickup then delivery
    assert abs(eval_route_directed([1, 2], inst, L0=8) - 1.0) < 1e-9
    assert abs(eval_route_directed([2, 1], inst, L0=8) - 7.0) < 1e-9
    assert abs(eval_route(route_fwd, inst, L0=8) - 1.0) < 1e-9  # takes min

def test_zero_penalty_all_deliveries():
    """Route of pure deliveries from L0=0 should have zero penalty."""
    inst = make_trivial_instance(d=[[-2.0, -3.0, -1.0]], prob=[1.0], Q=10, L0=0)
    route = Route([1, 2, 3])
    assert eval_route(route, inst) == 0.0

def test_scenario_averaging():
    """Two equally probable scenarios: 1.0 and 3.0 → expected 2.0."""
    inst = make_trivial_instance(d=[[11.0], [13.0]], prob=[0.5, 0.5], Q=10, L0=0)
    route = Route([1])
    assert abs(eval_route(route, inst) - 2.0) < 1e-9

def test_vectorised_matches_loop():
    """Vectorised implementation matches naive loop for random instance."""
    inst = make_random_instance(n=8, N=50, seed=42)
    route = Route(list(range(1, 9)))
    vec = eval_route_directed(route.customers, inst)
    loop_val = loop_eval(route.customers, inst)   # reference naive impl
    assert abs(vec - loop_val) < 1e-9
```

### 5.4 Performance target

| Instance size | Scenario count | Max eval time |
|---|---|---|
| t = 10 customers | N = 200 | < 1 ms |
| t = 30 customers | N = 200 | < 5 ms |
| t = 80 customers | N = 200 | < 20 ms |

Use `numpy` vectorisation. Do **not** use Python loops over scenarios.

### 5.5 Acceptance criteria for Phase 2

- [ ] All unit tests pass
- [ ] Proposition 1 counterexample verified exactly: fwd=1.0, bwd=7.0
- [ ] Performance targets met (benchmark with `pytest-benchmark`)
- [ ] Both orientations evaluated; minimum returned

---

## 6. Phase 3 — Lower Bound Engine

**Goal:** Compute valid lower bounds on the expected trajectory penalty for customer sets. These feed directly into the set-cut coefficients in the B&C.

### 6.1 Jensen single-route lower bound

```python
# src/bounds/jensen.py
import numpy as np
from src.core.instance import Instance

def jensen_bound_ordered(
    customers: list[int],
    instance: Instance,
    L0: float = None
) -> float:
    """
    Jensen lower bound for a FIXED ordering of customers.
    
    L_J(S, σ, L0) = Σ_j max(0, L0 + Σ_{h≤j} μ_{σ(h)} − Q)
    
    Valid by Jensen's inequality: max(0, E[L_j] - Q) ≤ E[max(0, L_j - Q)].
    
    Time: O(t).
    """
    if L0 is None:
        L0 = instance.initial_load
    Q    = instance.capacity
    mu   = instance.mean_demand[[c - 1 for c in customers]]  # (t,)
    
    prefix_means = np.cumsum(mu)          # E[L_j] - L0
    loads        = L0 + prefix_means      # E[L_j]
    overloads    = np.maximum(0.0, loads - Q)
    return float(overloads.sum())


def jensen_bound_set(
    customers: list[int],
    instance: Instance,
    L0: float = None
) -> tuple[float, list[int]]:
    """
    Best Jensen lower bound over all orderings of a customer set.
    
    Returns (bound_value, best_ordering).
    
    IMPORTANT: The 'deliveries-first' heuristic (sort by mean demand ascending)
    is a plausible minimiser but is NOT proved to be optimal.
    Use it as a heuristic and label it clearly in code and paper.
    
    For small |customers| ≤ 8, use brute-force enumeration.
    For larger sets, use the heuristic ordering.
    """
    if len(customers) <= 8:
        return _brute_force_min_jensen(customers, instance, L0)
    else:
        return _heuristic_min_jensen(customers, instance, L0)


def _heuristic_min_jensen(
    customers: list[int],
    instance: Instance,
    L0: float
) -> tuple[float, list[int]]:
    """
    Heuristic: sort customers by mean_demand ascending
    (deliveries first, then pickups).
    NOT proved optimal — treat as valid lower bound (any ordering gives one).
    """
    mu = instance.mean_demand[[c - 1 for c in customers]]
    order = np.argsort(mu)                            # ascending
    ordered = [customers[i] for i in order]
    bound = jensen_bound_ordered(ordered, instance, L0)
    return bound, ordered


def _brute_force_min_jensen(
    customers: list[int],
    instance: Instance,
    L0: float
) -> tuple[float, list[int]]:
    """Enumerate all permutations. Only for |customers| ≤ 8."""
    from itertools import permutations
    best_val = float('inf')
    best_ord = customers
    for perm in permutations(customers):
        val = jensen_bound_ordered(list(perm), instance, L0)
        if val < best_val:
            best_val, best_ord = val, list(perm)
    return best_val, best_ord
```

### 6.2 Multi-vehicle partition DP

```python
def jensen_bound_partition(
    customers: list[int],
    k_tilde: int,
    instance: Instance,
    L0: float = None
) -> float:
    """
    Partition-based Jensen lower bound for k_tilde vehicles.
    
    L¹_MD(S, k̃) = min over partitions {S_1,...,S_{k̃}} of Σ_r L¹_MD(S_r, 1)
    
    Uses a greedy partition heuristic (exact DP is combinatorial).
    This is a VALID lower bound regardless of partition quality.
    
    Implementation: greedy — assign customer with largest |μ_i| to the
    vehicle whose current Jensen bound is smallest (load-balancing heuristic).
    """
    if L0 is None:
        L0 = instance.initial_load
    if k_tilde == 1:
        return jensen_bound_set(customers, instance, L0)[0]
    
    # Greedy partition
    mu = instance.mean_demand[[c - 1 for c in customers]]
    
    # Sort by |mean demand| descending — large demands assigned first
    order = np.argsort(-np.abs(mu))
    sorted_customers = [customers[i] for i in order]
    
    buckets = [[] for _ in range(k_tilde)]
    bucket_loads = [L0] * k_tilde  # track mean running load end
    
    for c in sorted_customers:
        # Assign to bucket that minimises incremental Jensen bound
        best_bucket, best_delta = 0, float('inf')
        for b in range(k_tilde):
            delta = jensen_bound_ordered(buckets[b] + [c], instance, L0) \
                    - jensen_bound_ordered(buckets[b], instance, L0)
            if delta < best_delta:
                best_bucket, best_delta = b, delta
        buckets[best_bucket].append(c)
    
    return sum(jensen_bound_set(b, instance, L0)[0] for b in buckets if b)
```

### 6.3 Position-based disaggregation values

```python
def disaggregation_values(route: Route, instance: Instance) -> dict[int, float]:
    """
    Compute Q̂_MD(R, v_j) = E[max(0, L_j - Q)] for each position j.
    
    This IS the route-disjoint disaggregation that makes set cuts valid.
    Returns dict mapping customer_index -> disaggregated recourse value.
    """
    customers = route.customers
    Q    = instance.capacity
    L0   = instance.initial_load
    D    = instance.demand[:, [c - 1 for c in customers]]  # (N, t)
    prob = instance.prob
    
    prefix = np.cumsum(D, axis=1)
    loads  = L0 + prefix                               # (N, t)
    pos_penalties = np.maximum(0.0, loads - Q)         # (N, t)
    expected_pos  = prob @ pos_penalties               # (t,)
    
    return {customers[j]: float(expected_pos[j]) for j in range(len(customers))}
```

### 6.4 Unit tests

```python
# tests/test_bounds.py

def test_jensen_below_exact():
    """Jensen bound must always be ≤ exact expected penalty."""
    inst = make_random_instance(n=6, N=100, seed=0)
    for _ in range(20):
        customers = random_subset(inst)
        route = Route(customers)
        exact = eval_route(route, inst)
        lb, _ = jensen_bound_set(customers, inst)
        assert lb <= exact + 1e-9, f"Jensen {lb} > exact {exact}"

def test_jensen_proposition1():
    """Proposition 1 instance: ordering matters."""
    inst = make_trivial_instance(d=[[-6.0, 9.0]], prob=[1.0], Q=10, L0=8)
    # fwd ordering [1,2]: Jensen = max(0,2-10) + max(0,11-10) = 0+1 = 1
    assert abs(jensen_bound_ordered([1, 2], inst, L0=8) - 1.0) < 1e-9
    # bwd ordering [2,1]: Jensen = max(0,17-10) + max(0,11-10) = 7+1 = 8
    assert abs(jensen_bound_ordered([2, 1], inst, L0=8) - 8.0) < 1e-9

def test_multi_vehicle_bound_leq_single():
    """Multi-vehicle bound ≤ single-vehicle bound."""
    inst = make_random_instance(n=10, N=50, seed=1)
    customers = list(range(1, 11))
    lb1 = jensen_bound_partition(customers, 1, inst)
    lb2 = jensen_bound_partition(customers, 2, inst)
    assert lb2 <= lb1 + 1e-9

def test_disaggregation_sums_to_route_eval():
    """Σ_j Q̂(R, v_j) == Q_MD(R) for the best orientation."""
    inst = make_random_instance(n=5, N=30, seed=2)
    route = Route([1, 2, 3, 4, 5])
    vals = disaggregation_values(route, inst)
    assert abs(sum(vals.values()) - eval_route_directed(route.customers, inst)) < 1e-9
```

### 6.5 Acceptance criteria for Phase 3

- [ ] `jensen_bound ≤ eval_route` for 1000 random (route, instance) pairs
- [ ] `disaggregation_values` sums to exact route penalty for all test routes
- [ ] Brute-force and heuristic orderings agree for all `|S| ≤ 8` cases tested
- [ ] Multi-vehicle bound is monotone: k̃+1 vehicles ≤ k̃ vehicles

---

## 7. Phase 4 — ALNS Warm-Start Heuristic

**Goal:** Produce a high-quality feasible solution before B&C opens. The ALNS must understand trajectory-based penalty, not just distance.

### 7.1 ALNS controller

```python
# src/alns/alns.py
from alns import ALNS as ALNSBase
from alns.accept import SimulatedAnnealing
import numpy as np

def run_alns(
    instance,
    max_iterations: int = 5000,
    seed: int = 0,
    time_limit_s: float = 60.0
) -> 'Solution':
    """
    Run ALNS to get a warm-start incumbent.
    Uses fast Jensen surrogate for neighbourhood evaluation,
    exact oracle for incumbent acceptance.
    """
    rng = np.random.default_rng(seed)
    alns = ALNSBase(rng)
    
    # Register operators (see destroy.py, repair.py)
    alns.add_destroy_operator(worst_prefix_destroy)
    alns.add_destroy_operator(signed_cluster_destroy)
    alns.add_destroy_operator(peak_contributor_destroy)
    alns.add_destroy_operator(route_split_destroy)
    
    alns.add_repair_operator(trajectory_regret_repair)
    alns.add_repair_operator(delivery_first_repair)
    alns.add_repair_operator(cancellation_aware_repair)
    
    init_sol = greedy_initial_solution(instance)
    
    criterion = SimulatedAnnealing(
        start_temperature=... ,
        end_temperature=... ,
        step=...
    )
    
    result = alns.iterate(init_sol, criterion, max_iterations,
                          stop_condition=TimeLimit(time_limit_s))
    return result.best_state
```

### 7.2 Initial solution

```python
# src/alns/alns.py
def greedy_initial_solution(instance) -> Solution:
    """
    Nearest-neighbour greedy construction.
    Routes are split when mean expected load would exceed Q.
    Negative-demand customers assigned first when possible.
    """
    ...
```

### 7.3 Destroy operators

```python
# src/alns/destroy.py

def worst_prefix_destroy(solution, rng, n_remove=None):
    """
    Remove customers around the route position with highest expected overload.
    
    For each route, find j* = argmax_j E[L_j].
    Remove the customer at j* and its immediate neighbours.
    n_remove ~ U(q_min, q_max) where q in [0.1n, 0.3n].
    """
    ...

def signed_cluster_destroy(solution, rng, n_remove=None):
    """
    Remove a block of customers that has high |Σμ_i| but small |net μ|.
    These are high-cancellation clusters: removing them frees up
    trajectory flexibility for the repair phase.
    """
    ...

def peak_contributor_destroy(solution, rng, n_remove=None):
    """
    Remove the n_remove customers with largest positive mean demand μ_i.
    These are the customers most likely to cause trajectory spikes.
    """
    ...

def route_split_destroy(solution, rng, n_remove=None):
    """
    Split the route with highest expected penalty at position j*.
    Customers after j* become a partial solution for repair.
    """
    ...
```

### 7.4 Repair operators

```python
# src/alns/repair.py

def trajectory_regret_repair(solution, rng):
    """
    Insert unassigned customers greedily by minimising:
        Δdist + c_p * ΔL̂_MD(R)
    
    where ΔL̂_MD is the Jensen bound increment from inserting
    customer i at position p.
    
    Use jensen_bound_ordered (fast, O(t)) not eval_route (slow, O(Nt))
    for neighbourhood screening.
    """
    ...

def delivery_first_repair(solution, rng):
    """
    For each route being built, preferentially insert customers
    with μ_i < 0 (deliveries) before customers with μ_i > 0 (pickups).
    Falls back to trajectory_regret for ties.
    """
    ...

def cancellation_aware_repair(solution, rng):
    """
    Jointly insert paired (delivery, pickup) customers when their
    combined trajectory peak is lower than inserting each separately.
    Check: jensen_bound_ordered([..., d, p, ...]) vs individual inserts.
    """
    ...
```

### 7.5 Scoring and acceptance

```python
# src/alns/scoring.py

def evaluate_candidate(solution, instance, exact=False):
    """
    If exact=True:  use eval_route (Algorithm 1) — for incumbent acceptance.
    If exact=False: use jensen_bound_set — for neighbourhood screening.
    """
    if exact:
        penalty = sum(eval_route(r, instance) for r in solution.routes)
    else:
        penalty = sum(
            jensen_bound_set(r.customers, instance)[0]
            for r in solution.routes
        )
    dist = sum(route_distance(r, instance) for r in solution.routes)
    return dist + instance.cost_fleet * len(solution.routes) \
           + instance.cost_penalty * penalty
```

### 7.6 ALNS unit tests

```python
# tests/test_alns.py

def test_alns_feasibility():
    """All routes in ALNS output must cover each customer exactly once."""
    inst = load_test_instance()
    sol = run_alns(inst, max_iterations=100)
    covered = sorted(c for r in sol.routes for c in r.customers)
    assert covered == list(range(1, inst.n_customers + 1))

def test_alns_improves_over_greedy():
    """ALNS objective should be <= greedy initial objective."""
    inst = load_test_instance()
    greedy = greedy_initial_solution(inst)
    alns   = run_alns(inst, max_iterations=1000)
    assert alns.objective <= greedy.objective

def test_delivery_first_lowers_penalty():
    """delivery_first_repair should produce lower mean-trajectory peak."""
    ...
```

### 7.7 Acceptance criteria for Phase 4

- [ ] ALNS respects customer coverage (each customer in exactly one route)
- [ ] ALNS objective ≤ greedy initial objective after 1000 iterations on all test instances
- [ ] `delivery_first_repair` outperforms `trajectory_regret_repair` on instances with ≥ 30% deliveries (verify on 10 instances)
- [ ] Wall-clock time ≤ 60 s for 80-customer instances

---

## 8. Phase 5 — Branch-and-Cut Solver

**Goal:** Exact solver producing provably optimal solutions. Built on top of Gurobi or CPLEX callbacks.

### 8.1 Master problem formulation

```python
# src/bnc/master.py
import gurobipy as gp
from gurobipy import GRB

def build_master(instance, n_vehicles_max: int) -> gp.Model:
    """
    Edge-flow formulation.
    x[i,j] ∈ {0,1,2} — number of times edge {i,j} is traversed.
    θ[i]    ≥ 0       — disaggregated recourse for customer i.
    
    Objective:
        min Σ c_ij x_ij + c_f * Σ z_m * m + c_p * Σ θ_i
    
    Subject to:
        Degree constraints (each customer visited exactly once)
        Rounded capacity inequalities (lazy — added via CVRPSEP callback)
        θ[i] ≥ 0 for all i
    """
    m = gp.Model("SVRP_MD")
    n = instance.n_customers
    
    # Edge variables
    x = {}
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            x[i, j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=2,
                                name=f"x_{i}_{j}")
    
    # Recourse variables (disaggregated by customer)
    theta = {i: m.addVar(lb=0.0, name=f"theta_{i}")
             for i in range(1, n + 1)}
    
    # Degree constraints
    for i in range(1, n + 1):
        m.addConstr(
            gp.quicksum(x[min(i,j), max(i,j)]
                        for j in range(n + 1) if j != i) == 2,
            name=f"degree_{i}"
        )
    
    # Depot degree = 2 * n_vehicles
    # (handled lazily via fleet size variable or fixed)
    
    # Objective
    dist_cost = gp.quicksum(
        instance.distance[i][j] * x[min(i,j), max(i,j)]
        for i in range(n + 1) for j in range(i + 1, n + 1)
    )
    recourse_cost = instance.cost_penalty * gp.quicksum(theta.values())
    
    m.setObjective(dist_cost + recourse_cost, GRB.MINIMIZE)
    m.update()
    
    return m, x, theta
```

### 8.2 Callback dispatcher

```python
# src/bnc/callback.py
import gurobipy as gp
from gurobipy import GRB

class SolverCallback(gp.Callback):
    def __init__(self, instance, x_vars, theta_vars, x_map):
        super().__init__()
        self.instance  = instance
        self.x_vars    = x_vars
        self.theta_vars = theta_vars
        self.x_map     = x_map   # maps (i,j) -> Gurobi var
    
    def callback(self):
        if self.where == GRB.Callback.MIPNODE:
            # At fractional LP nodes
            x_val = {k: self.cbGetNodeRel(v) for k, v in self.x_vars.items()}
            self._separate_rci(x_val, fractional=True)
            self._separate_jensen_set_cuts(x_val)
            # self._separate_pr_cuts(x_val)  # activate after Theorem D
        
        elif self.where == GRB.Callback.MIPSOL:
            # At integer nodes
            x_val = {k: self.cbGetSolution(v) for k, v in self.x_vars.items()}
            routes = self._extract_routes(x_val)
            self._separate_route_cuts(routes, x_val)
            self._separate_rci(x_val, fractional=False)
    
    def _separate_rci(self, x_val, fractional):
        """Call CVRPSEP, add violated RCIs as lazy constraints."""
        ...
    
    def _separate_route_cuts(self, routes, x_val):
        """
        For each route R in the current integer solution:
          - Compute Q_MD(R) exactly via eval_route
          - Compute activation W(x; X(R)) = Σ_{e in R\δ(0)} (x_e - 1)
          - If θ(V+(R)) < Q_MD(R) * W: add the cut
        """
        ...
    
    def _separate_jensen_set_cuts(self, x_val):
        """
        Use block-cut tree heuristic to identify candidate sets S.
        For each S with activation > 0:
          - Compute L¹_MD(S, k̃) via jensen_bound_partition
          - Compute W_P(x; X(S, k̃)) = 1 + x(S) - |S| + k̃
          - If θ(S) < L * W: add the cut
        """
        ...
    
    def _extract_routes(self, x_val):
        """Recover route list from integer x solution."""
        ...
```

### 8.3 Route cut structure

```python
# src/bnc/cuts/route_cuts.py

def add_route_cut(model, route, Q_md, x_vars, theta_vars, instance):
    """
    Route cut:
        Σ_{i in V+(R)} θ_i  ≥  Q_MD(R) * W(x; X(R))
    
    where W(x; X(R)) = 1 + Σ_{e in E(R) \ δ(0)} (x_e - 1)
                     = Σ_{e in E(R) \ δ(0)} x_e  - |V+(R)| + 2
    
    This is an OPTIMALITY CUT: tight at the current solution x̄.
    """
    customers = route.customers
    n_cust    = len(customers)
    
    # Edges internal to route (not depot edges)
    route_edges = [(min(customers[j], customers[j+1]),
                    max(customers[j], customers[j+1]))
                   for j in range(n_cust - 1)]
    
    lhs = gp.quicksum(theta_vars[c] for c in customers)
    rhs_activation = (gp.quicksum(x_vars[e] for e in route_edges)
                      - n_cust + 2)
    
    model.cbLazy(lhs >= Q_md * rhs_activation)
```

### 8.4 Jensen set cut structure

```python
# src/bnc/cuts/set_cuts.py

def add_jensen_set_cut(model, S, k_tilde, L_md, x_vars, theta_vars):
    """
    Jensen set cut:
        Σ_{i in S} θ_i  ≥  L¹_MD(S, k̃) * W_P(x; X(S, k̃))
    
    where W_P(x; X(S, k̃)) = 1 + x(S) - |S| + k̃
          x(S) = Σ_{e = {i,j}: i,j in S} x_e
    
    This is a LOWER BOUNDING FUNCTIONAL: active whenever S is
    covered by exactly k̃ paths in the solution.
    """
    lhs = gp.quicksum(theta_vars[c] for c in S)
    
    x_S = gp.quicksum(
        x_vars[min(i,j), max(i,j)]
        for i in S for j in S if i < j
    )
    
    activation = 1 + x_S - len(S) + k_tilde
    model.cbLazy(lhs >= L_md * activation)
```

### 8.5 Main solver entry point

```python
# src/bnc/master.py

def solve(
    instance,
    warm_start: 'Solution' = None,
    time_limit_s: float = 3600.0,
    verbose: bool = False
) -> dict:
    """
    Full branch-and-cut solve.
    
    Returns dict with:
        solution      : best Solution found
        objective     : objective value
        gap           : optimality gap (%)
        n_nodes       : B&B nodes explored
        solve_time_s  : wall-clock time
        cuts_added    : dict with cut counts by type
        proved_optimal: bool
    """
    model, x_vars, theta_vars = build_master(instance)
    
    if warm_start:
        _inject_warm_start(model, warm_start, x_vars, theta_vars)
    
    cb = SolverCallback(instance, x_vars, theta_vars, ...)
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = time_limit_s
    model.Params.Threads = 1   # single-threaded for reproducibility
    
    model.optimize(cb)
    
    return {
        'solution':       _extract_solution(model, x_vars),
        'objective':      model.ObjVal,
        'gap':            model.MIPGap * 100,
        'n_nodes':        int(model.NodeCount),
        'solve_time_s':   model.Runtime,
        'cuts_added':     cb.cut_counts,
        'proved_optimal': model.Status == GRB.OPTIMAL
    }
```

### 8.6 B&C unit tests

```python
# tests/test_cuts.py

def test_route_cut_validity():
    """Route cut must not cut off optimal solution on small instance."""
    inst = load_tiny_instance()   # 3 customers, known optimal
    # Solve, check that route cuts were not overly aggressive
    result = solve(inst, time_limit_s=10)
    assert result['proved_optimal']

def test_jensen_cut_validity():
    """Jensen set cut LHS must be ≥ RHS at any integer feasible solution."""
    # Build an instance, solve ALNS, verify cuts hold
    ...

def test_disaggregation_theta_sum():
    """At optimality, Σ θ_i should equal Σ_R Q_MD(R)."""
    ...
```

### 8.7 Acceptance criteria for Phase 5

- [ ] Solves all 3-customer test instances to provable optimality in < 1 s
- [ ] Solves Jabali 40-customer instances (adapted) within 1-hour time limit
- [ ] No route cut ever removes a feasible solution (verified by checking θ feasibility)
- [ ] Jensen set cuts never yield θ(S) < 0 at integer nodes
- [ ] Cut counts logged per type per run

---

## 9. Phase 6 — Instance Generation

**Goal:** Produce a systematic benchmark suite for SVRP-MD. Adapt existing VRPSD instances and generate new ones.

### 9.1 Generator

```python
# src/instance_gen/generator.py
import numpy as np
from src.core.instance import Instance

def adapt_jabali_instance(
    jabali_path: str,
    delivery_ratio: float = 0.3,
    n_scenarios: int = 200,
    seed: int = 0
) -> Instance:
    """
    Convert a Jabali et al. (2014) VRPSD instance to SVRP-MD format.
    
    Steps:
    1. Load original instance (normal distribution, all positive demands).
    2. Flip sign of delivery_ratio * n customers (chosen at random).
    3. Preserve absolute mean demand scale.
    4. Sample n_scenarios scenarios from signed normal distributions.
    
    Parameters
    ----------
    delivery_ratio : fraction of customers that are deliveries (μ < 0)
    """
    rng = np.random.default_rng(seed)
    base = _load_jabali(jabali_path)    # returns dict with means, stds, dist
    
    n = base['n_customers']
    means = base['means'].copy()
    stds  = base['stds'].copy()
    
    # Flip signs of delivery_ratio fraction
    n_deliveries = int(np.round(delivery_ratio * n))
    delivery_idx = rng.choice(n, n_deliveries, replace=False)
    means[delivery_idx] *= -1
    
    # Sample scenarios
    demand = np.zeros((n_scenarios, n))
    for i in range(n):
        demand[:, i] = rng.normal(loc=means[i], scale=stds[i],
                                   size=n_scenarios)
    
    prob = np.ones(n_scenarios) / n_scenarios
    
    return Instance(
        name=f"{base['name']}_dr{int(delivery_ratio*100)}_s{seed}",
        n_customers=n,
        capacity=base['capacity'],
        initial_load=0.0,
        cost_penalty=1.0,
        cost_fleet=0.0,
        distance=base['distance'],
        demand=demand,
        prob=prob,
    )
```

### 9.2 Instance parameter grid

```python
INSTANCE_GRID = {
    'base_sets':       ['jabali_40', 'jabali_50', 'jabali_60',
                        'jabali_70', 'jabali_80'],
    'delivery_ratio':  [0.1, 0.3, 0.5],
    'n_scenarios':     [100, 200],
    'seeds':           [0, 1, 2],               # 3 replicates per config
    'initial_load':    [0.0, 'half_Q'],         # L0 = 0 and L0 = Q/2
}
# Total: 5 × 3 × 2 × 3 × 2 = 180 instances
```

### 9.3 Instance-level metrics

```python
# src/instance_gen/metrics.py

def compute_instance_metrics(instance: Instance, solution=None) -> dict:
    """
    Metrics to report per instance in the computational study.
    All can be computed without a solution (pass solution=None).
    """
    mu = instance.mean_demand
    Q  = instance.capacity
    L0 = instance.initial_load
    
    metrics = {
        'n_customers':      instance.n_customers,
        'n_scenarios':      instance.n_scenarios,
        'delivery_ratio':   (mu < 0).mean(),
        'net_demand':       mu.sum(),
        'mean_abs_demand':  np.abs(mu).mean(),
        'demand_variance':  instance.var_demand.mean(),
        
        # Trajectory sensitivity: how much does ordering matter?
        # Compare best vs worst 2-customer ordering
        'ordering_sensitivity': _ordering_sensitivity(instance),
    }
    
    if solution:
        metrics.update(_solution_metrics(solution, instance))
    
    return metrics

def _solution_metrics(solution, instance) -> dict:
    """Per-solution metrics for the results table."""
    route_penalties = [eval_route(r, instance) for r in solution.routes]
    route_peaks = [
        max(instance.initial_load + np.cumsum(
                instance.mean_demand[[c-1 for c in r.customers]]))
        for r in solution.routes
    ]
    
    # Position distribution of overflow (unique to SVRP-MD)
    overflow_by_position = _overflow_position_distribution(solution, instance)
    
    return {
        'total_distance':       solution.total_distance,
        'expected_penalty':     solution.expected_penalty,
        'objective':            solution.objective,
        'n_vehicles':           solution.n_vehicles,
        'avg_route_penalty':    np.mean(route_penalties),
        'max_route_peak_mean':  max(route_peaks),
        'p_any_overflow':       _prob_any_overflow(solution, instance),
        'overflow_position':    overflow_by_position,  # list of P(L_j > Q)
    }
```

### 9.4 Acceptance criteria for Phase 6

- [ ] All 180 instances generated without error
- [ ] `delivery_ratio` in output matches requested ratio ± 1/n
- [ ] `prob.sum() == 1.0` for all generated instances
- [ ] Jabali original instances and adapted SVRP-MD instances have same distance matrices

---

## 10. Phase 7 — Evaluation Pipeline

**Goal:** Run all experiments, collect results, produce paper-ready tables.

### 10.1 Experiment configurations

```yaml
# experiments/configs/ablation.yaml
name: "Algorithm ablation"
instances: "data/generated/jabali_adapted/*.json"
time_limit_s: 3600
algorithms:
  - name: "ALNS_only"
    alns_iterations: 50000
    branch_and_cut: false
  - name: "BnC_route_cuts_only"
    alns_warmstart: true
    cuts: [route]
  - name: "BnC_route_jensen"
    alns_warmstart: true
    cuts: [route, jensen_set]
  - name: "BnC_full"
    alns_warmstart: true
    cuts: [route, jensen_set, pr_cuts]   # activate after Theorem D
metrics:
  - opt_count           # instances solved to optimality
  - avg_time_s          # average solve time (solved only)
  - avg_gap_pct         # average gap (unsolved only)
  - avg_nodes           # average B&B nodes
  - cuts_route          # average route cuts added
  - cuts_jensen         # average Jensen set cuts added
```

```yaml
# experiments/configs/hypotheses.yaml
name: "Hypothesis tests"
instances: "data/generated/**/*.json"
algorithms: ["BnC_route_jensen"]
group_by:
  - delivery_ratio      # H1, H4
  - demand_variance     # H4
  - n_customers         # H5
  - initial_load        # H2
```

### 10.2 Batch runner

```python
# src/eval/runner.py
import yaml, json, time
from pathlib import Path

def run_experiment(config_path: str, output_dir: str):
    cfg = yaml.safe_load(open(config_path))
    results = []
    
    for inst_path in sorted(Path('.').glob(cfg['instances'])):
        inst = load_instance(str(inst_path))
        inst_metrics = compute_instance_metrics(inst)
        
        for alg_cfg in cfg['algorithms']:
            print(f"  [{alg_cfg['name']}] {inst.name}")
            
            warm = None
            if alg_cfg.get('alns_warmstart'):
                warm = run_alns(inst, max_iterations=alg_cfg.get('alns_iterations', 5000))
            
            if alg_cfg.get('branch_and_cut', True):
                result = solve(inst, warm_start=warm,
                               time_limit_s=cfg['time_limit_s'],
                               cuts=alg_cfg.get('cuts', ['route', 'jensen_set']))
            else:
                result = {'solution': warm, 'objective': warm.objective,
                          'gap': None, 'proved_optimal': False,
                          'solve_time_s': ..., 'n_nodes': 0}
            
            row = {
                'instance': inst.name,
                'algorithm': alg_cfg['name'],
                **inst_metrics,
                **result,
                **compute_instance_metrics(inst, result['solution'])
            }
            results.append(row)
    
    # Save raw results
    import pandas as pd
    df = pd.DataFrame(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{cfg['name']}.csv", index=False)
    print(f"Saved {len(results)} rows to {out}")
```

### 10.3 Metrics and KPIs

#### Primary metrics (paper tables)

| Metric | Column name | Unit | Description |
|---|---|---|---|
| Instances solved | `opt_count` | count | Proved optimal within time limit |
| Average solve time | `avg_time_s` | seconds | Solved instances only |
| Average gap | `avg_gap_pct` | % | Unsolved instances only |
| B&B nodes | `avg_nodes` | count | Solved instances only |

#### Secondary metrics (ablation and hypothesis tables)

| Metric | Column name | Unit | Description |
|---|---|---|---|
| Route cuts added | `cuts_route` | count | Per run average |
| Jensen set cuts | `cuts_jensen` | count | Per run average |
| LP bound at root | `root_lb` | value | Lower bound before branching |
| Root gap | `root_gap_pct` | % | Gap closed before first branch |
| Ordering sensitivity | `order_sens` | value | Best vs worst 2-customer ordering |
| Overflow position | `overflow_pos_j` | vector | P(L_j > Q) per position |
| ALNS objective | `alns_obj` | value | Warm-start quality |
| ALNS time | `alns_time_s` | seconds | Time in warm-start phase |

### 10.4 Hypothesis test structure

Each hypothesis maps to specific metric comparisons:

```python
HYPOTHESIS_TESTS = {
    'H1': {
        'claim': 'Ordering matters: best vs worst ordering differ significantly',
        'metric': 'ordering_sensitivity',
        'test':   'mean > threshold (e.g. > 10% of Q_MD)',
        'group':  'delivery_ratio',
    },
    'H2': {
        'claim': 'Jensen bounds improve LP relaxation vs route cuts only',
        'metric': 'root_gap_pct',
        'test':   'BnC_route_jensen.root_gap < BnC_route_cuts_only.root_gap',
        'aggregation': 'paired t-test across instances',
    },
    'H3': {
        'claim': 'Trajectory-aware ALNS > distance-only ALNS on high-delivery instances',
        'metric': 'alns_obj',
        'test':   'trajectory_alns.alns_obj < baseline_alns.alns_obj',
        'filter': 'delivery_ratio >= 0.3',
    },
    'H4': {
        'claim': 'Cut benefit grows with delivery ratio and variance',
        'metric': 'opt_count',
        'test':   'Spearman correlation(opt_count, delivery_ratio) > 0',
    },
    'H5': {
        'claim': 'B&C competitive on few long routes vs many short routes',
        'metric': 'opt_count',
        'test':   'opt_count higher when n_customers/n_vehicles >= threshold',
    },
}
```

### 10.5 Paper table templates

```python
# src/eval/report.py

def make_main_table(results_csv: str) -> str:
    """
    Produce Table 2 (paper): algorithm comparison across instance groups.
    
    Columns: Instance group | n | BnC_route | BnC_route+Jensen | BnC_full
    Each cell: Opt/n  |  avg_time_s  |  avg_gap%
    """
    ...

def make_ablation_table(results_csv: str) -> str:
    """
    Produce Table 3 (paper): cut contribution ablation.
    Rows = instance sizes, columns = algorithm variants.
    """
    ...

def make_hypothesis_table(results_csv: str) -> str:
    """
    Produce Table 4 (paper): hypothesis test results.
    Rows = H1..H5, columns = metric | test result | p-value.
    """
    ...
```

### 10.6 Acceptance criteria for Phase 7

- [ ] `run_experiment(ablation.yaml)` completes on 180 instances without crashes
- [ ] All 5 hypotheses have a corresponding result column in the output CSV
- [ ] `opt_count` + unsolved count = total instances for each algorithm
- [ ] `avg_time_s` only computed over solved instances (no contamination from time-limit hits)
- [ ] All result CSVs reproducible with the same seed and config

---

## 11. Testing Strategy

### 11.1 Test categories

| Category | Location | Run on |
|---|---|---|
| Unit tests | `tests/test_*.py` | Every commit |
| Integration tests | `tests/test_integration.py` | Every PR |
| Regression tests | `tests/test_regression.py` | Before paper submission |
| Performance benchmarks | `tests/bench_*.py` | Weekly |

### 11.2 Regression test instances

Maintain a small set of instances with known optimal values for regression testing. Start with:

- **Tiny-1**: 3 customers, 1 vehicle, all pickups, known optimal = `X`
- **Tiny-2**: 4 customers, 1 vehicle, 2 pickups + 2 deliveries, Proposition 1 structure
- **Small-1**: 10 customers, 2 vehicles, adapted from Jabali, solved to optimality manually
- **Prop1**: The Proposition 1 counterexample instance (Q=10, L0=8, d=[-6, +9])

```python
# tests/test_regression.py
REGRESSION_INSTANCES = {
    'tiny_1':  {'path': 'tests/fixtures/tiny_1.json', 'optimal': 12.34},
    'tiny_2':  {'path': 'tests/fixtures/tiny_2.json', 'optimal': 8.76},
    'prop1':   {'path': 'tests/fixtures/prop1.json',  'optimal': 1.0},
}

def test_regression_optimal_values():
    for name, spec in REGRESSION_INSTANCES.items():
        inst   = load_instance(spec['path'])
        result = solve(inst, time_limit_s=30)
        assert result['proved_optimal'], f"{name} not solved to optimality"
        assert abs(result['objective'] - spec['optimal']) < 1e-4, \
               f"{name}: expected {spec['optimal']}, got {result['objective']}"
```

### 11.3 Mathematical invariant tests

Run these on every instance in the generated benchmark:

```python
def check_mathematical_invariants(instance, solution, result):
    """Run after every solve. Raise AssertionError if violated."""
    
    # 1. Jensen bound ≤ exact penalty for every route
    for r in solution.routes:
        lb, _ = jensen_bound_set(r.customers, instance)
        exact = eval_route(r, instance)
        assert lb <= exact + 1e-6, f"Jensen violated: {lb} > {exact}"
    
    # 2. Disaggregation sums to route penalty
    for r in solution.routes:
        vals = disaggregation_values(r, instance)
        exact = eval_route_directed(r.customers, instance)
        assert abs(sum(vals.values()) - exact) < 1e-6
    
    # 3. θ ≥ 0 for all customers (from solver)
    # 4. Objective = distance + c_f * vehicles + c_p * penalty
    recomputed = solution.total_distance \
                 + instance.cost_fleet * solution.n_vehicles \
                 + instance.cost_penalty * solution.expected_penalty
    assert abs(recomputed - result['objective']) < 1e-4
```

---

## 12. Performance Benchmarks and Acceptance Criteria

### 12.1 Phase completion gates

Each phase must pass its gate before the next phase begins.

| Phase | Gate condition |
|---|---|
| Phase 0 | CI green, all imports work |
| Phase 1 | 270 instances load cleanly, round-trip serialisation passes |
| Phase 2 | Proposition 1 verified exactly, performance targets met |
| Phase 3 | Jensen ≤ exact for 1000 random pairs, disaggregation invariant holds |
| Phase 4 | ALNS improves over greedy, feasibility holds, ≤ 60 s for 80-customer |
| Phase 5 | Tiny instances solved optimally, no cut ever removes feasible solution |
| Phase 6 | 180 instances generated, all metrics computable |
| Phase 7 | All hypothesis results present, tables reproducible |

### 12.2 Target performance profile (based on Legault et al. comparisons)

These are targets to aim for. Adjust after Phase 7 based on actual results.

| Instance group | Target opt rate | Target avg time | Max gap on unsolved |
|---|---|---|---|
| 40 customers, 2 vehicles | ≥ 90% | ≤ 120 s | ≤ 5% |
| 50 customers, 2 vehicles | ≥ 80% | ≤ 300 s | ≤ 8% |
| 60 customers, 2 vehicles | ≥ 65% | ≤ 600 s | ≤ 12% |
| 40 customers, 3 vehicles | ≥ 70% | ≤ 300 s | ≤ 10% |
| 50 customers, 3 vehicles | ≥ 50% | ≤ 600 s | ≤ 15% |

### 12.3 Baseline comparisons

Implement and run two baselines for every instance:

- **Baseline A** — ALNS only (no B&C). Reports best feasible objective and solve time.
- **Baseline B** — Route cuts only (no Jensen set cuts). Measures Jensen cut contribution.

The paper's main claim is: `BnC_full` dominates both baselines in opt rate and average solve time.

---

## 13. Open Theoretical Tasks for the Developer

These are items where the theory is incomplete and the developer should **flag** rather than implement speculatively.

### 13.1 Theorem D — PR-cut activation function (CRITICAL)

**Status:** Open. Do not implement PR-cuts until this is resolved.

**What needs to happen:** Construct an explicit affine function `W(x; X_τ(S, k̃))` that equals 1 when the solution routes S with orderings in `E_τ(S)`, and ≤ 0 otherwise.

**Developer task:** When implementing `_separate_pr_cuts` in the callback, stub it out:

```python
def _separate_pr_cuts(self, x_val):
    """
    PR-cut separation — STUBBED pending Theorem D proof.
    TODO: Replace with real activation function once proved.
    See Section 3.5 of the algorithm design document.
    """
    raise NotImplementedError(
        "PR-cuts require Theorem D (activation function proof). "
        "Contact Quang-Vinh before implementing."
    )
```

### 13.2 Ordering conjecture for Jensen minimiser

**Status:** Conjecture only. The "deliveries-first" heuristic is plausible but unproved.

**Developer task:** In `jensen_bound_set`, the heuristic is labelled clearly. Add a brute-force verification flag:

```python
def jensen_bound_set(customers, instance, L0=None, verify=False):
    bound, ordering = _heuristic_min_jensen(customers, instance, L0)
    if verify and len(customers) <= 8:
        bf_bound, _ = _brute_force_min_jensen(customers, instance, L0)
        assert abs(bound - bf_bound) < 1e-9 or bound > bf_bound, \
            "Heuristic ordering is NOT optimal for this case — report!"
    return bound, ordering
```

Run with `verify=True` on all small instances during testing. Log any case where heuristic ≠ brute-force.

### 13.3 Second-order (dispersion-aware) bounds

**Status:** Research direction. Do not implement as a theorem.

**Developer task:** Implement as an **optional module** that can be switched on via config:

```yaml
# experiments/configs/dispersion_research.yaml
cuts:
  - route
  - jensen_set
  - dispersion_bound:   # EXPERIMENTAL — may not be valid
      type: scenario_clustering
      n_clusters: 10
```

Always label results from this as `EXPERIMENTAL` in the output CSV.

---

## 14. Dependency Map

The following diagram shows which phases depend on which. Do not start a phase until all its dependencies are complete and their acceptance criteria are met.

```
Phase 0 (Environment)
    │
    ▼
Phase 1 (Data Layer)
    │
    ├──► Phase 6 (Instance Generation)
    │
    ▼
Phase 2 (Route Oracle)
    │
    ├──► Phase 3 (Lower Bounds)  ──► Phase 5 (B&C)  ──► Phase 7 (Evaluation)
    │                                     ▲
    └──► Phase 4 (ALNS)  ─────────────────┘
```

### 14.1 Critical path

The critical path to a working solver is:

```
Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 5 (route cuts only) → Phase 7 (partial)
```

This gives a running system with ALNS + route-cut B&C and partial results. Then add:

```
Phase 3 (Jensen bounds) → Phase 5 (Jensen set cuts) → Phase 7 (full)
```

PR-cuts (from Phase 5) follow only after Theorem D is proved.

### 14.2 Parallel work opportunities

- Phase 6 (instance generation) can run in parallel with Phase 3 and Phase 4 — it only needs Phase 1.
- Phase 7 evaluation infrastructure can be built in parallel with Phase 5.
- The ALNS (Phase 4) can begin as soon as Phase 2 is complete — it does not depend on Phase 3.

---

*Last updated: working draft — integrate with SVRP_MD_v2.docx for full theoretical context.*
