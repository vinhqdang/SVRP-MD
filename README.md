# SVRP-MD (Stochastic Vehicle Routing Problem with Mixed Demands)

This repository contains the full source code for modeling, evaluating, and optimally solving the **Stochastic Vehicle Routing Problem with Mixed Demands (SVRP-MD)**. 

SVRP-MD involves planning routes for a fleet of capacity-constrained vehicles to serve a set of geographically dispersed customers. In this uniquely complex framework, customer demands are both stochastic (represented as a set of discrete scenarios with probabilities) and **mixed-sign** (pickups have positive demand; deliveries have negative demand). Our objective is to minimize the total expected routing cost, defined as the deterministic distance traversed plus the probabilistically weighted expected recourse penalties that result if a vehicle exceeds its physical capacity mid-route.

## Our Solver: STORM ⚡
The **STORM** (Stochastic Trajectory Optimization for Routing with Mixed-demands) algorithm is a state-of-the-art Exact Branch-and-Cut solver designed from the ground up for SVRP-MD. 
- **Exact Trajectory Anticipation:** Accurately evaluates the expected recourse penalties of every proposed route across all discrete scenarios, distinguishing between symmetrical traversal directions due to the non-commutative nature of mixed pickups and deliveries.
- **ALNS Warm-Starting:** Employs an Adaptive Large Neighborhood Search (ALNS) heuristic with specialized trajectory-regret repair operators to identify high-quality initial upper bounds quickly.
- **Jensen Bounds & Disaggregation:** To accelerate convergence, STORM utilizes powerful Jensen lower bounds derived analytically for customer sets, breaking down penalty expectations linearly to add dynamic **Jensen Set Cuts** to the Branch-and-Cut tree, drastically pruning the search space.

## Baseline Approach: Robust CVRP
For comprehensive evaluation, this repository also ships a strict exact baseline algorithm adopted from *The Robust Capacitated Vehicle Routing Problem Under Demand Uncertainty* (Gounaris et al., 2013). This solver:
- Avoids penalty evaluations entirely by restricting solutions only to routes that securely guarantee 100% capacity feasibility across *every single possible scenario*. 
- Enforces worst-case constraints lazily via dynamic capacity elimination cuts (`baseline_robust.py`).
While computationally much faster on small instances, it produces exceedingly conservative, expensive decisions with sub-optimal real-world expected costs.

## Directory Structure
- `src/core/`: Contains the foundational `Instance`, `Route`, and `Solution` data schemas.
- `src/oracle/`: The penalty simulation oracle (`eval_route`) utilized to determine exact subset recourse costs.
- `src/bounds/`: Mathematical lower bound engines leveraging the specialized `jensen_bound_set` formulas.
- `src/alns/`: Destructive and restorative repair heuristics operating the ALNS.
- `src/bnc/`: Core B&C optimization engines housing the master problem for STORM (`master.py`), the callbacks (`callback.py`), and the `baseline_robust.py` implementation.
- `src/instance_gen/` & `src/eval/`: Extensive instance generator code (to adapt CVRP datasets to mixed-sign stochastic instances) and the pipeline configuration runner.
- `scripts/`: Large scale execution helpers and instance generators.
- `experiments/configs/`: YAML setups for automated hypothesis ablation testing.

## Running Experiments
To test STORM against the baseline on randomly generated large-scale test datasets:

1. **Activate the Environment:**
   Ensure you have all dependencies (`numpy, gurobipy, scipy, networkx, pandas, alns, pytest, pyyaml`) installed in a Python 3.13 environment.

2. **Generate Test Instances:**
   ```bash
   python scripts/generate_large_scale.py
   ```
   *Generates robust testing schemas varying customer counts (e.g., 10, 12, 14 nodes) and stochastic delivery ratios (20% - 40%).*

3. **Evaluate the Pipeline:**
   ```bash
   python run_large_scale.py
   ```
   *Runs Gurobi B&C to solve the generated `.json` files via both algorithms, comparing computation times and Expected Penalties objectively according to the parameters explicitly described inside `experiments/configs/ablation.yaml`.*

## Reproducibility
All Python logic is thoroughly unit-tested using `pytest`. The primary solver runs on **Gurobi Optimizer** (valid license required). Ensure `gurobipy` is authenticated.
