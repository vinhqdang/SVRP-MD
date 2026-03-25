import numpy as np
from src.core.instance import Instance
from src.core.route import Route
from src.oracle.route_eval import eval_route, eval_route_directed

def make_trivial_instance(d, prob, Q, L0):
    d_arr = np.array(d)
    if d_arr.ndim == 1:
        d_arr = d_arr[:, None]
    return Instance(
        name="trivial",
        n_customers=d_arr.shape[1],
        capacity=Q,
        initial_load=L0,
        cost_penalty=1.0,
        cost_fleet=0.0,
        distance=np.zeros((d_arr.shape[1] + 1, d_arr.shape[1] + 1)),
        demand=d_arr,
        prob=np.array(prob)
    )

def make_random_instance(n, N, seed):
    rng = np.random.default_rng(seed)
    return Instance(
        name="random",
        n_customers=n,
        capacity=100.0,
        initial_load=0.0,
        cost_penalty=1.0,
        cost_fleet=1.0,
        distance=rng.uniform(0, 10, size=(n+1, n+1)),
        demand=rng.normal(0, 10, size=(N, n)),
        prob=np.ones(N) / N
    )

def test_deterministic_single_customer():
    inst = make_trivial_instance(d=[[9.0]], prob=[1.0], Q=10, L0=3)
    route = Route([1])
    assert abs(eval_route(route, inst) - 2.0) < 1e-9

def test_deterministic_delivery_then_pickup():
    inst = make_trivial_instance(d=[[-6.0, 9.0]], prob=[1.0], Q=10, L0=8)
    route_fwd = Route([1, 2])
    assert abs(eval_route_directed([1, 2], inst, L0=8) - 1.0) < 1e-9
    assert abs(eval_route_directed([2, 1], inst, L0=8) - 8.0) < 1e-9
    assert abs(eval_route(route_fwd, inst, L0=8) - 1.0) < 1e-9

def test_zero_penalty_all_deliveries():
    inst = make_trivial_instance(d=[[-2.0, -3.0, -1.0]], prob=[1.0], Q=10, L0=0)
    route = Route([1, 2, 3])
    assert eval_route(route, inst) == 0.0

def test_scenario_averaging():
    inst = make_trivial_instance(d=[[11.0], [13.0]], prob=[0.5, 0.5], Q=10, L0=0)
    route = Route([1])
    assert abs(eval_route(route, inst) - 2.0) < 1e-9

def test_vectorised_matches_loop():
    inst = make_random_instance(n=8, N=50, seed=42)
    route = Route(list(range(1, 9)))
    vec = eval_route_directed(route.customers, inst)
    
    # Naive loop
    loop_val = 0.0
    for i in range(inst.n_scenarios):
        load = inst.initial_load
        penalty = 0.0
        for c in route.customers:
            load += inst.demand[i, c-1]
            penalty += max(0, load - inst.capacity)
        loop_val += inst.prob[i] * penalty
        
    assert abs(vec - loop_val) < 1e-9
