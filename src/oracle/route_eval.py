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
    """
    fwd = eval_route_directed(route.customers, instance, L0)
    bwd = eval_route_directed(route.customers[::-1], instance, L0)
    return min(fwd, bwd)


def eval_solution(solution, instance: Instance) -> float:
    """Compute total expected trajectory penalty across all routes."""
    return sum(eval_route(r, instance) for r in solution.routes)

def route_distance(route: Route, instance: Instance) -> float:
    """Total travel distance for a route (including depot legs)."""
    seq = [0] + route.customers + [0]  # depot = index 0
    return sum(instance.distance[seq[i]][seq[i+1]] for i in range(len(seq)-1))
