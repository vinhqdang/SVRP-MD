from tests.test_oracle import make_random_instance
from src.bnc.baseline_robust import solve_baseline_robust

def load_tiny_instance():
    return make_random_instance(n=3, N=10, seed=123)

def test_baseline_robust_validity():
    inst = load_tiny_instance()
    result = solve_baseline_robust(inst, time_limit_s=10)
    assert result['proved_optimal']
    
    # Verify the robust solution evaluates to expected penalty 0
    sol = result['solution']
    if sol:
        # A robust solution must have 0 expected penalty, meaning all routes are perfectly feasible
        assert sol.expected_penalty <= 1e-5
