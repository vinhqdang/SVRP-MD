import copy
from src.core.route import Route

def trajectory_regret_repair(solution, rng, **kwargs):
    sol = copy.deepcopy(solution)
    instance = sol.instance
    
    rng.shuffle(sol.unassigned)
    
    for c in sol.unassigned:
        best_r = None
        best_p = None
        best_delta = float('inf')
        
        for r_idx, r in enumerate(sol.routes):
            for p in range(len(r.customers) + 1):
                prev_c = r.customers[p-1] if p > 0 else 0
                next_c = r.customers[p] if p < len(r.customers) else 0
                delta = instance.distance[prev_c][c] + instance.distance[c][next_c] - instance.distance[prev_c][next_c]
                if delta < best_delta:
                    best_delta = delta
                    best_r = r_idx
                    best_p = p
                    
        if best_r is not None:
            sol.routes[best_r].customers.insert(best_p, c)
        else:
            sol.routes.append(Route([c]))
            
    sol.unassigned = []
    return sol

delivery_first_repair = trajectory_regret_repair
cancellation_aware_repair = trajectory_regret_repair
