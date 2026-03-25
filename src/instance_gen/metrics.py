import numpy as np
from src.oracle.route_eval import eval_route

def _ordering_sensitivity(instance) -> float:
    return 0.1 * instance.capacity

def compute_instance_metrics(instance, solution=None) -> dict:
    mu = instance.mean_demand
    metrics = {
        'n_customers':      instance.n_customers,
        'n_scenarios':      instance.n_scenarios,
        'delivery_ratio':   (mu < 0).mean(),
        'net_demand':       mu.sum(),
        'mean_abs_demand':  np.abs(mu).mean(),
        'demand_variance':  instance.var_demand.mean(),
        'ordering_sensitivity': _ordering_sensitivity(instance)
    }
    if solution:
        metrics.update(_solution_metrics(solution, instance))
    return metrics

def _overflow_position_distribution(solution, instance):
    return []

def _prob_any_overflow(solution, instance):
    return 0.0

def _solution_metrics(solution, instance) -> dict:
    route_penalties = [eval_route(r, instance) for r in solution.routes]
    route_peaks = []
    for r in solution.routes:
        load = instance.initial_load
        peak = load
        for c in r.customers:
            load += instance.mean_demand[c-1]
            peak = max(peak, load)
        route_peaks.append(peak)
        
    return {
        'total_distance':       solution.total_distance,
        'expected_penalty':     solution.expected_penalty,
        'objective':            solution.objective,
        'n_vehicles':           solution.n_vehicles,
        'avg_route_penalty':    np.mean(route_penalties) if route_penalties else 0.0,
        'max_route_peak_mean':  max(route_peaks) if route_peaks else 0.0,
        'p_any_overflow':       _prob_any_overflow(solution, instance),
        'overflow_position':    _overflow_position_distribution(solution, instance),
    }
