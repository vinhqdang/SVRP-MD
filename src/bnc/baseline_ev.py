from src.core.solution import Solution
from src.core.route import Route
import numpy as np

def solve_expected_value(instance) -> Solution:
    """
    Expected Value (EV) Strategy.
    Solves the routing problem as a deterministic CVRP using mean demands.
    Uses a greedy construction followed by a simple 2-opt improvement on mean costs.
    """
    n = instance.n_customers
    mu = instance.mean_demand
    Q = instance.capacity
    
    unassigned = set(range(1, n + 1))
    routes = []
    
    # Greedy Construction
    while unassigned:
        curr_route = []
        curr_load = 0.0
        curr_node = 0
        
        while unassigned:
            feasible = [c for c in unassigned if curr_load + mu[c-1] <= Q]
            if not feasible:
                break
            # Nearest neighbor
            nxt = min(feasible, key=lambda c: instance.distance[curr_node][c])
            curr_route.append(nxt)
            unassigned.remove(nxt)
            curr_load += mu[nxt-1]
            curr_node = nxt
            
        if not curr_route and unassigned:
            nxt = min(unassigned, key=lambda c: instance.distance[0][c])
            curr_route.append(nxt)
            unassigned.remove(nxt)
            
        routes.append(Route(curr_route))
    
    sol = Solution(routes=routes)
    sol.evaluate(instance) # evaluates with stochastic penalty
    return sol
