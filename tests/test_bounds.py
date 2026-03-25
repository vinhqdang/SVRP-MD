import numpy as np
from src.core.instance import Instance
from src.core.route import Route
from src.oracle.route_eval import eval_route, eval_route_directed
from src.bounds.jensen import jensen_bound_set, jensen_bound_ordered, jensen_bound_partition, disaggregation_values

from tests.test_oracle import make_trivial_instance, make_random_instance

def random_subset(inst):
    rng = np.random.default_rng(0)
    k = rng.integers(1, inst.n_customers + 1)
    return rng.choice(np.arange(1, inst.n_customers + 1), size=k, replace=False).tolist()

def test_jensen_below_exact():
    inst = make_random_instance(n=6, N=100, seed=0)
    for _ in range(20):
        customers = random_subset(inst)
        route = Route(customers)
        exact = eval_route(route, inst)
        lb, _ = jensen_bound_set(customers, inst)
        assert lb <= exact + 1e-9, f"Jensen {lb} > exact {exact}"

def test_jensen_proposition1():
    inst = make_trivial_instance(d=[[-6.0, 9.0]], prob=[1.0], Q=10, L0=8)
    assert abs(jensen_bound_ordered([1, 2], inst, L0=8) - 1.0) < 1e-9
    assert abs(jensen_bound_ordered([2, 1], inst, L0=8) - 8.0) < 1e-9

def test_multi_vehicle_bound_leq_single():
    inst = make_random_instance(n=10, N=50, seed=1)
    customers = list(range(1, 11))
    lb1 = jensen_bound_partition(customers, 1, inst)
    lb2 = jensen_bound_partition(customers, 2, inst)
    assert lb2 <= lb1 + 1e-9

def test_disaggregation_sums_to_route_eval():
    inst = make_random_instance(n=5, N=30, seed=2)
    route = Route([1, 2, 3, 4, 5])
    vals = disaggregation_values(route, inst)
    assert abs(sum(vals.values()) - eval_route_directed(route.customers, inst)) < 1e-9
