import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    os.makedirs('results/plots', exist_ok=True)
    df = pd.read_csv('results/ablation_study.csv')
    
    # Color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. Average Solve Time by Algorithm
    summary_time = df.groupby('algorithm')['solve_time_s'].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    plt.bar(summary_time['algorithm'], summary_time['solve_time_s'], color=colors)
    plt.title('Average Solve Time by Algorithm (Log Scale)', fontsize=14)
    plt.ylabel('Solve Time (s)')
    plt.yscale('log') # Log scale because time difference might be massive
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/plots/solve_time_comparison.png', dpi=300)
    plt.close()
    
    # 2. Objective Value Comparison across instances
    import numpy as np
    
    # Temporarily cap infinite objectives (infeasible instances) to a constant above the max finite value so it renders
    max_finite = df.loc[df['objective'] != np.inf, 'objective'].max()
    plot_df = df.copy()
    plot_df.loc[plot_df['objective'] == np.inf, 'objective'] = max_finite * 1.1
    
    pivot_df = plot_df.pivot(index='instance', columns='algorithm', values='objective')
    
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind='bar', ax=plt.gca(), width=0.8, color=colors)
    plt.title('Objective Value by Instance and Algorithm', fontsize=14)
    plt.ylabel('Objective Function (Distance + Expected Penalty)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/plots/objective_comparison.png', dpi=300)
    plt.close()
    
    print("Plots successfully generated in results/plots/ directory.")

if __name__ == '__main__':
    generate_plots()
