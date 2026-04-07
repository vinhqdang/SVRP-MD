from src.core.solution import Solution
from src.core.route import Route
import numpy as np

def solve_greedy_sequential(instance, buffer=0.1) -> Solution:
    """
    Greedy Sequential Insertion (GSI) baseline.
    Builds one route at a time using nearest neighbor that fits mean capacity.
    """
    n = instance.n_customers
    unassigned = set(range(1, n + 1))
    mu = instance.mean_demand
    Q = instance.capacity * (1.0 - buffer) # Conservative buffer
    
    routes = []
    
    while unassigned:
        curr_route = []
        curr_load = 0.0
        curr_node = 0 # depot
        
        while unassigned:
            # Filter feasible by mean demand and capacity buffer
            feasible = [c for c in unassigned if curr_load + mu[c-1] <= Q]
            if not feasible:
                break
                
            # Pick nearest neighbor
            nxt = min(feasible, key=lambda c: instance.distance[curr_node][c])
            curr_route.append(nxt)
            unassigned.remove(nxt)
            curr_load += mu[nxt-1]
            curr_node = nxt
            
        if not curr_route and unassigned:
            # Forced to start new route with one customer even if over buffer
            nxt = min(unassigned, key=lambda c: instance.distance[0][c])
            curr_route.append(nxt)
            unassigned.remove(nxt)
            
        routes.append(Route(curr_route))
        
    sol = Solution(routes=routes)
    sol.evaluate(instance)
    return sol
