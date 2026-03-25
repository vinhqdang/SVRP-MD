from tests.test_oracle import make_random_instance, make_trivial_instance
from src.bnc.master import solve

def load_tiny_instance():
    return make_random_instance(n=3, N=10, seed=123)

def test_route_cut_validity():
    inst = load_tiny_instance()
    result = solve(inst, time_limit_s=10)
    assert result['proved_optimal']
    
def test_jensen_cut_validity():
    inst = load_tiny_instance()
    result = solve(inst, time_limit_s=10, cuts=['route', 'jensen_set'])
    assert result['proved_optimal']
