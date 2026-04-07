from src.core.solution import Solution
from src.core.route import Route
import numpy as np

def solve_tsp_split(instance, fill_rate=0.9) -> Solution:
    """
    TSP-First, Route-Second (TFRS) baseline.
    1. Generate a TSP tour using a greedy heuristic.
    2. Split the tour into routes based on capacity.
    """
    n = instance.n_customers
    mu = instance.mean_demand
    Q = instance.capacity * fill_rate
    
    # 1. Simple Greedy TSP tour
    unvisited = set(range(1, n + 1))
    tour = []
    curr = 0 # depot
    while unvisited:
        nxt = min(unvisited, key=lambda x: instance.distance[curr][x])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
        
    # 2. Split the tour
    routes = []
    curr_route = []
    curr_load = 0.0
    
    for c in tour:
        if curr_load + mu[c-1] <= Q:
            curr_route.append(c)
            curr_load += mu[c-1]
        else:
            if curr_route:
                routes.append(Route(curr_route))
            curr_route = [c]
            curr_load = mu[c-1]
            
    if curr_route:
        routes.append(Route(curr_route))
        
    sol = Solution(routes=routes)
    sol.evaluate(instance)
    return sol
