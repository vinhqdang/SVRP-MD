from src.core.route import Route
from src.core.solution import Solution
from src.oracle.route_eval import eval_route, route_distance
from src.bounds.jensen import jensen_bound_set

def evaluate_candidate(solution: Solution, instance, exact=False):
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
