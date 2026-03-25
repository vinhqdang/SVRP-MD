import yaml, json, time
from pathlib import Path
from src.core.instance import load_instance
from src.instance_gen.metrics import compute_instance_metrics
from src.alns.alns import run_alns
from src.bnc.master import solve
import pandas as pd

def run_experiment(config_path: str, output_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    results = []
    
    for inst_path in sorted(Path('.').glob(cfg['instances'])):
        inst = load_instance(str(inst_path))
        inst_metrics = compute_instance_metrics(inst)
        
        for alg_cfg in cfg['algorithms']:
            warm = None
            if alg_cfg.get('alns_warmstart'):
                warm = run_alns(inst, max_iterations=alg_cfg.get('alns_iterations', 50))
                
            if alg_cfg.get('branch_and_cut', True):
                result = solve(inst, warm_start=warm,
                               time_limit_s=cfg.get('time_limit_s', 3600),
                               cuts=alg_cfg.get('cuts', ['route', 'jensen_set']))
            else:
                result = {'solution': warm, 'objective': warm.objective,
                          'gap': None, 'proved_optimal': False,
                          'solve_time_s': 0, 'n_nodes': 0, 'cuts_added': {}}
            
            row = {
                'instance': inst.name,
                'algorithm': alg_cfg['name'],
                **inst_metrics,
                **result,
            }
            if result.get('solution'):
                row.update(compute_instance_metrics(inst, result['solution']))
                del row['solution']
            else:
                del row['solution']
                
            results.append(row)
            
    df = pd.DataFrame(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{cfg['name']}.csv", index=False)
