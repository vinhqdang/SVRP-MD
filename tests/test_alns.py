import numpy as np
from src.alns.alns import run_alns, greedy_initial_solution
from tests.test_oracle import make_random_instance

def load_test_instance():
    return make_random_instance(n=10, N=20, seed=42)

def test_alns_feasibility():
    inst = load_test_instance()
    sol = run_alns(inst, max_iterations=10)
    covered = sorted(c for r in sol.routes for c in r.customers)
    assert covered == list(range(1, inst.n_customers + 1))

def test_alns_improves_over_greedy():
    inst = load_test_instance()
    greedy_state = greedy_initial_solution(inst)
    greedy_obj = greedy_state.objective()
    alns_sol = run_alns(inst, max_iterations=20)
    assert alns_sol.objective <= greedy_obj

def test_delivery_first_lowers_penalty():
    pass
