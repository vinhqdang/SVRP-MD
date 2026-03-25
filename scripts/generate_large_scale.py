import os
import json
import numpy as np
from tests.test_oracle import make_random_instance

def generate_instances():
    os.makedirs('data/generated/large_scale', exist_ok=True)
    instance_count = 0
    
    for n in [10, 12, 14]:
        for dr in [0.2, 0.4]:
            for seed in range(3): 
                inst = make_random_instance(n=n, N=50, seed=seed)
                rng = np.random.default_rng(seed)
                n_del = int(np.round(dr * n))
                del_idx = rng.choice(n, n_del, replace=False)
                
                means = inst.mean_demand.copy()
                stds = np.sqrt(inst.var_demand)
                means[del_idx] = -np.abs(means[del_idx])
                
                demand = np.zeros((50, n))
                for i in range(n):
                    demand[:, i] = rng.normal(loc=means[i], scale=stds[i], size=50)
                inst.demand = demand
                inst.__post_init__() 
                
                # Force tight settings to distinguish robust vs stochastic expected penalty!
                inst.capacity = float(np.sum(np.abs(means)) * 0.4)
                inst.cost_fleet = 100.0  # Make extra vehicles very expensive
                inst.cost_penalty = 1.0  # Make recourse somewhat cheaper than a whole new vehicle
                
                inst.name = f"inst_n{n}_dr{int(dr*100)}_{seed}"
                
                data = {
                    'name': inst.name,
                    'n_customers': inst.n_customers,
                    'capacity': inst.capacity,
                    'initial_load': inst.initial_load,
                    'cost_penalty': inst.cost_penalty,
                    'cost_fleet': inst.cost_fleet,
                    'distance': inst.distance.tolist(),
                    'demand': inst.demand.tolist(),
                    'prob': inst.prob.tolist()
                }
                with open(f'data/generated/large_scale/{inst.name}.json', 'w') as f:
                    json.dump(data, f)
                instance_count += 1
    print(f"Generated {instance_count} instances.")

if __name__ == '__main__':
    generate_instances()
