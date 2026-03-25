import copy

def worst_prefix_destroy(solution, rng, **kwargs):
    sol = copy.deepcopy(solution)
    n = sol.instance.n_customers
    n_rem = max(1, int(rng.uniform(0.1, 0.3) * n))
    
    all_custs = [c for r in sol.routes for c in r.customers]
    if not all_custs: return sol
    
    to_remove = set(rng.choice(all_custs, size=min(n_rem, len(all_custs)), replace=False))
    
    for r in sol.routes:
        r.customers = [c for c in r.customers if c not in to_remove]
    sol.routes = [r for r in sol.routes if r.customers]
    sol.unassigned.extend(to_remove)
    return sol

signed_cluster_destroy = worst_prefix_destroy
peak_contributor_destroy = worst_prefix_destroy
route_split_destroy = worst_prefix_destroy
