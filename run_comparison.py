import os
import yaml
import json
import numpy as np
import pandas as pd
from src.core.instance import Instance
from tests.test_oracle import make_random_instance
from src.eval.runner import run_experiment

def save_test_instances():
    os.makedirs('data/test_instances', exist_ok=True)
    for i in range(3):
        # 8 customers, 50 scenarios
        inst = make_random_instance(n=8, N=50, seed=42+i)
        inst.name = f"test_inst_{i}"
        
        # Save as JSON (handling numpy arrays)
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
        with open(f'data/test_instances/inst_{i}.json', 'w') as f:
            json.dump(data, f)

def build_config():
    cfg = {
        'name': 'baseline_comparison',
        'instances': 'data/test_instances/*.json',
        'time_limit_s': 60,
        'algorithms': [
            {'name': 'baseline_robust', 'branch_and_cut': True},
            {'name': 'svrp_exact', 'branch_and_cut': True, 'cuts': ['route', 'jensen_set']}
        ]
    }
    with open('experiment_cfg.yaml', 'w') as f:
        yaml.dump(cfg, f)

if __name__ == "__main__":
    print("Saving test instances...")
    save_test_instances()
    print("Building config...")
    build_config()
    print("Running experiment...")
    run_experiment('experiment_cfg.yaml', 'results')
    
    print("Reading results...")
    df = pd.read_csv('results/baseline_comparison.csv')
    print(df[['instance', 'algorithm', 'objective', 'expected_penalty', 'solve_time_s', 'proved_optimal']])
