from alns import ALNS as ALNSBase
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations
from alns.select import RandomSelect
import numpy as np

from src.core.route import Route
from src.core.solution import Solution
from src.alns.scoring import evaluate_candidate
from src.alns.destroy import worst_prefix_destroy, signed_cluster_destroy, peak_contributor_destroy, route_split_destroy
from src.alns.repair import trajectory_regret_repair, delivery_first_repair, cancellation_aware_repair

class ALNSSolution:
    def __init__(self, routes, instance):
        self.routes = routes
        self.instance = instance
        self.unassigned = []
        
    def objective(self):
        dummy = Solution(routes=self.routes)
        return evaluate_candidate(dummy, self.instance, exact=False)

def greedy_initial_solution(instance) -> ALNSSolution:
    unassigned = set(range(1, instance.n_customers + 1))
    routes = []
    mu = instance.mean_demand
    
    order = sorted(list(unassigned), key=lambda x: mu[x-1])
    unassigned = set(order)
    
    while unassigned:
        curr_load = instance.initial_load
        curr_node = 0
        route_custs = []
        
        while unassigned:
            feasible = [c for c in unassigned if curr_load + mu[c-1] <= instance.capacity]
            if not feasible:
                break
            feasible.sort(key=lambda c: instance.distance[curr_node][c])
            nxt = feasible[0]
            route_custs.append(nxt)
            unassigned.remove(nxt)
            curr_load += mu[nxt-1]
            curr_node = nxt
            
        if not route_custs:
            nxt = list(unassigned)[0]
            route_custs.append(nxt)
            unassigned.remove(nxt)
            
        routes.append(Route(route_custs))
        
    return ALNSSolution(routes=routes, instance=instance)

def run_alns(
    instance,
    max_iterations: int = 50,
    seed: int = 0,
    time_limit_s: float = 60.0
) -> ALNSSolution:
    rng = np.random.default_rng(seed)
    alns = ALNSBase(rng)
    
    alns.add_destroy_operator(worst_prefix_destroy)
    alns.add_destroy_operator(signed_cluster_destroy)
    alns.add_destroy_operator(peak_contributor_destroy)
    alns.add_destroy_operator(route_split_destroy)
    
    alns.add_repair_operator(trajectory_regret_repair)
    alns.add_repair_operator(delivery_first_repair)
    alns.add_repair_operator(cancellation_aware_repair)
    
    init_sol = greedy_initial_solution(instance)
    
    criterion = SimulatedAnnealing(
        start_temperature=100.0,
        end_temperature=1.0,
        step=0.999,
        method="exponential"
    )
    
    stop_crit = MaxIterations(max_iterations)
    select_crit = RandomSelect(len(alns.destroy_operators), len(alns.repair_operators))

    result = alns.iterate(init_sol, select_crit, criterion, stop_crit)
    
    best = result.best_state
    final_sol = Solution(routes=best.routes)
    final_sol.compute_objective(instance.cost_fleet, instance.cost_penalty)
    return final_sol
