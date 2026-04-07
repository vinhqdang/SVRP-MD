import numpy as np
from src.core.instance import Instance

def generate_spatial_instance(
    n_customers: int,
    distribution: str = 'uniform',
    delivery_ratio: float = 0.5,
    n_scenarios: int = 50,
    capacity: float = 100.0,
    seed: int = 42
) -> Instance:
    """
    Generate a large-scale SVRP-MD instance with specific spatial and demand patterns.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate Locations (Nodes)
    # depot = (50, 50)
    coords = np.zeros((n_customers + 1, 2))
    coords[0] = [50.0, 50.0]
    
    if distribution == 'uniform':
        coords[1:] = rng.uniform(0.0, 100.0, size=(n_customers, 2))
    
    elif distribution == 'circular':
        # Nodes placed on concentric rings
        for i in range(1, n_customers + 1):
            ring = rng.integers(1, 6)  # Up to 5 rings
            radius = ring * 10.0 + rng.uniform(-2, 2)
            angle = rng.uniform(0, 2 * np.pi)
            coords[i] = [50.0 + radius * np.cos(angle), 50.0 + radius * np.sin(angle)]
            
    elif distribution == 'clustered':
        # 5 clusters
        n_clusters = 5
        centers = rng.uniform(20.0, 80.0, size=(n_clusters, 2))
        for i in range(1, n_customers + 1):
            c_idx = rng.integers(0, n_clusters)
            coords[i] = centers[c_idx] + rng.normal(0, 5.0, size=2)
            
    # 2. Distance Matrix (Euclidean)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # 3. Generate Demands
    means = rng.uniform(5, 15, size=n_customers)
    n_deliveries = int(np.round(delivery_ratio * n_customers))
    delivery_idx = rng.choice(n_customers, n_deliveries, replace=False)
    means[delivery_idx] *= -1
    
    stds = rng.uniform(1, 4, size=n_customers)
    
    demand = np.zeros((n_scenarios, n_customers))
    for i in range(n_customers):
        demand[:, i] = rng.normal(loc=means[i], scale=stds[i], size=n_scenarios)
        
    prob = np.ones(n_scenarios) / n_scenarios
    
    return Instance(
        name=f"auto_{distribution}_n{n_customers}_dr{int(delivery_ratio*100)}_s{seed}",
        n_customers=n_customers,
        capacity=capacity,
        initial_load=0.0,
        cost_penalty=2.0,
        cost_fleet=50.0,
        distance=dist,
        demand=demand,
        prob=prob
    )
