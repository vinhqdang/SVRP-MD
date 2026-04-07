import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def generate_report_charts():
    df = pd.read_csv('results/benchmarks/summary.csv')
    os.makedirs('results/benchmarks/plots', exist_ok=True)
    
    # Define consistent colors for algorithms
    colors = {
        'STORM (Integrated)': '#2ecc71', # Green
        'STORM (ALNS-Heuristic)': '#27ae60', # Dark Green
        'Baseline_Robust': '#e74c3c', # Red
        'Baseline_EV': '#3498db',    # Blue
        'Baseline_TFRS': '#f1c40f',  # Yellow
        'Baseline_GSI': '#9b59b6'    # Purple
    }

    # 1. Objective Value by Algorithm and Scale
    plt.figure(figsize=(14, 7))
    subset = df.groupby(['n_customers', 'algorithm'])['objective'].mean().unstack()
    # Explicitly re-index columns to ensure all 6 are present in the defined order
    all_algos = ['STORM (Integrated)', 'STORM (ALNS-Heuristic)', 'Baseline_Robust', 'Baseline_EV', 'Baseline_TFRS', 'Baseline_GSI']
    subset = subset.reindex(columns=[c for c in all_algos if c in subset.columns or c in colors])
    
    subset.plot(kind='bar', ax=plt.gca(), width=0.8, color=[colors.get(x, '#333') for x in subset.columns])
    
    plt.ylabel('Avg Objective Value (Expected Cost)')
    plt.xlabel('Network Scale (n_customers)')
    plt.title('Algorithm Comparison: Solution Quality vs Network Scale')
    plt.xticks(rotation=0)
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/benchmarks/plots/dist_improvement.png')
    plt.close()

    # 2. Solve Time Comparison
    plt.figure(figsize=(14, 7))
    time_subset = df.groupby(['n_customers', 'algorithm'])['time_s'].mean().unstack()
    time_subset = time_subset.reindex(columns=subset.columns) # Sync algorithms
    
    time_subset.plot(kind='bar', ax=plt.gca(), width=0.8, logy=True, color=[colors.get(x, '#333') for x in time_subset.columns])
    plt.ylabel('Avg Solve Time (s) - Log Scale')
    plt.xlabel('Network Scale (n_customers)')
    plt.title('Computational Efficiency Comparison')
    plt.xticks(rotation=0)
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/benchmarks/plots/time_vs_scale.png')
    plt.close()
    
    print("Charts generated in results/benchmarks/plots/")

if __name__ == "__main__":
    generate_report_charts()
