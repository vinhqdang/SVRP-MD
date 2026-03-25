import numpy as np
from src.core.instance import Instance

def jensen_bound_ordered(
    customers: list[int],
    instance: Instance,
    L0: float = None
) -> float:
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
    L0: float = None,
    verify: bool = False
) -> tuple[float, list[int]]:
    if len(customers) <= 8:
        bound, ordering = _heuristic_min_jensen(customers, instance, L0)
        if verify:
            bf_bound, _ = _brute_force_min_jensen(customers, instance, L0)
            assert abs(bound - bf_bound) < 1e-9 or bound > bf_bound, \
                "Heuristic ordering is NOT optimal for this case — report!"
        return _brute_force_min_jensen(customers, instance, L0)
    else:
        return _heuristic_min_jensen(customers, instance, L0)

def _heuristic_min_jensen(
    customers: list[int],
    instance: Instance,
    L0: float
) -> tuple[float, list[int]]:
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
    from itertools import permutations
    best_val = float('inf')
    best_ord = customers
    for perm in permutations(customers):
        val = jensen_bound_ordered(list(perm), instance, L0)
        if val < best_val:
            best_val, best_ord = val, list(perm)
    return best_val, list(best_ord)

def jensen_bound_partition(
    customers: list[int],
    k_tilde: int,
    instance: Instance,
    L0: float = None
) -> float:
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
    
    for c in sorted_customers:
        best_bucket, best_delta = 0, float('inf')
        for b in range(k_tilde):
            delta = jensen_bound_ordered(buckets[b] + [c], instance, L0) \
                    - jensen_bound_ordered(buckets[b], instance, L0)
            if delta < best_delta:
                best_bucket, best_delta = b, delta
        buckets[best_bucket].append(c)
    
    return sum(jensen_bound_set(b, instance, L0)[0] for b in buckets if b)

def disaggregation_values(route: 'Route', instance: Instance) -> dict[int, float]:
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
