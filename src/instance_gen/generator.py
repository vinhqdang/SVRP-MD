import numpy as np
from src.core.instance import Instance

def _load_jabali(path: str):
    n = 10
    rng = np.random.default_rng(0)
    return {
        'name': 'jabali_base',
        'n_customers': n,
        'capacity': 100.0,
        'distance': rng.uniform(0, 10, size=(n+1, n+1)),
        'means': rng.uniform(5, 15, size=n),
        'stds': rng.uniform(1, 3, size=n)
    }

def adapt_jabali_instance(
    jabali_path: str,
    delivery_ratio: float = 0.3,
    n_scenarios: int = 200,
    seed: int = 0
) -> Instance:
    rng = np.random.default_rng(seed)
    base = _load_jabali(jabali_path)
    
    n = base['n_customers']
    means = base['means'].copy()
    stds  = base['stds'].copy()
    
    n_deliveries = int(np.round(delivery_ratio * n))
    delivery_idx = rng.choice(n, n_deliveries, replace=False)
    means[delivery_idx] *= -1
    
    demand = np.zeros((n_scenarios, n))
    for i in range(n):
        demand[:, i] = rng.normal(loc=means[i], scale=stds[i], size=n_scenarios)
    
    prob = np.ones(n_scenarios) / n_scenarios
    
    return Instance(
        name=f"{base['name']}_dr{int(delivery_ratio*100)}_s{seed}",
        n_customers=n,
        capacity=base['capacity'],
        initial_load=0.0,
        cost_penalty=1.0,
        cost_fleet=0.0,
        distance=base['distance'],
        demand=demand,
        prob=prob,
    )
