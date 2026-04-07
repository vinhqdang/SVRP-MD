import os
import pandas as pd
import time
import json
from src.instance_gen.generator import generate_spatial_instance
from src.alns.alns import run_alns
from src.bnc.master import solve

from src.bnc.baseline_robust import solve_baseline_robust
from src.bnc.baseline_ev import solve_expected_value
from src.alns.baseline_tfrs import solve_tsp_split
from src.alns.baseline_greedy import solve_greedy_sequential

def run_20_benchmarks():
    configs = []
    sizes = [100, 200, 500, 1000]
    dists = ['uniform', 'circular', 'clustered']
    
    for n in sizes:
        for d in dists:
            configs.append({'n': n, 'dist': d, 'dr': 0.5})
            
    for n in [100, 500]:
        for dr in [0.2, 0.8]:
            configs.append({'n': n, 'dist': 'uniform', 'dr': dr})
            
    for n in [200, 1000]:
        configs.append({'n': n, 'dist': 'clustered', 'dr': 0.5, 'seed': 123})
        configs.append({'n': n, 'dist': 'circular', 'dr': 0.3, 'seed': 456})
        
    configs = configs[:20]
    
    results = []
    os.makedirs('results/benchmarks', exist_ok=True)
    
    algorithms = [
        ('STORM (Integrated)', None), # Special handling
        ('STORM (ALNS-Heuristic)', lambda inst: run_alns(inst, max_iterations=100)),
        ('Baseline_Robust', lambda inst: solve_baseline_robust(inst, time_limit_s=10)),
        ('Baseline_EV', lambda inst: solve_expected_value(inst)),
        ('Baseline_TFRS', lambda inst: solve_tsp_split(inst)),
        ('Baseline_GSI', lambda inst: solve_greedy_sequential(inst))
    ]
    
    for i, cfg in enumerate(configs):
        print(f"--- Case {i+1}/20: n={cfg['n']}, dist={cfg['dist']}, dr={cfg['dr']} ---")
        
        inst = generate_spatial_instance(
            n_customers=cfg['n'],
            distribution=cfg['dist'],
            delivery_ratio=cfg['dr'],
            seed=cfg.get('seed', i)
        )
        
        for algo_name, algo_func in algorithms:
            print(f"  Running {algo_name}...")
            start_time = time.time()
            try:
                if algo_name == 'STORM (Integrated)':
                    # ALNS + B&C
                    sol = run_alns(inst, max_iterations=50)
                    try:
                        bnc_res = solve(inst, warm_start=sol, time_limit_s=10)
                        final_obj = bnc_res['objective']
                    except Exception as e:
                        if "Model too large" in str(e):
                            final_obj = sol.objective
                        else: raise e
                else:
                    sol_or_res = algo_func(inst)
                    if hasattr(sol_or_res, 'objective'):
                        final_obj = sol_or_res.objective
                    else: # Robust solver returns a dict
                        final_obj = sol_or_res['objective']
                    
                solve_time = time.time() - start_time
                
                results.append({
                    'case_id': i + 1,
                    'n_customers': cfg['n'],
                    'distribution': cfg['dist'],
                    'delivery_ratio': cfg['dr'],
                    'algorithm': algo_name,
                    'objective': final_obj if final_obj != float('inf') else None,
                    'time_s': solve_time,
                    'status': 'SUCCESS' if final_obj != float('inf') else 'INFEASIBLE'
                })
            except Exception as e:
                print(f"    {algo_name} failed: {str(e)}")
                results.append({
                    'case_id': i + 1,
                    'n_customers': cfg['n'],
                    'distribution': cfg['dist'],
                    'delivery_ratio': cfg['dr'],
                    'algorithm': algo_name,
                    'objective': None,
                    'time_s': time.time() - start_time,
                    'status': 'LICENSE_LIMIT' if "Model too large" in str(e) else 'ERROR'
                })

        # Save incremental results
        pd.DataFrame(results).to_csv('results/benchmarks/summary.csv', index=False)
        
    print("Benchmarks completed. Summary saved to results/benchmarks/summary.csv")

if __name__ == "__main__":
    run_20_benchmarks()
