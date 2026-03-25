import os
import pandas as pd
from src.eval.runner import run_experiment

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    print("Running large scale evaluation with STORM...")
    run_experiment('experiments/configs/ablation.yaml', 'results')
    
    df = pd.read_csv('results/ablation_study.csv')
    
    summary = df.groupby('algorithm').agg({
        'objective': 'mean',
        'solve_time_s': 'mean',
        'proved_optimal': 'mean',
        'expected_penalty': 'mean'
    }).reset_index()
    
    print("\n--- STORM LARGE SCALE EVALUATION RESULTS ---")
    print(summary.to_markdown(index=False))
