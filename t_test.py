#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results():
    """Load all results from the directories"""
    algorithms = ['ga', 'pso', 'aco']
    results = {algo: [] for algo in algorithms}

    # Find all directories and load results
    for algo in algorithms:
        for dir_path in Path('results').glob(f'{algo}_*_*'):
            try:
                summary_path = dir_path / 'run_summary.json'
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        data = json.load(f)
                        results[algo].append({
                            'final_score': data['final_score'],
                            'runtime_s': data['runtime_s'],
                            'iterations': data['iterations'],
                            'directory': str(dir_path)
                        })
            except Exception as e:
                print(f"Error reading {summary_path}: {e}")

    return results


def perform_statistical_analysis(results):
    """Perform t-tests and descriptive statistics"""
    # Extract final scores
    ga_scores = [r['final_score'] for r in results['ga']]
    pso_scores = [r['final_score'] for r in results['pso']]
    aco_scores = [r['final_score'] for r in results['aco']]

    # Perform t-tests
    t_ga_pso, p_ga_pso = stats.ttest_ind(ga_scores, pso_scores)
    t_ga_aco, p_ga_aco = stats.ttest_ind(ga_scores, aco_scores)
    t_pso_aco, p_pso_aco = stats.ttest_ind(pso_scores, aco_scores)

    # Calculate descriptive statistics
    stats_data = {
        'GA': {
            'mean': np.mean(ga_scores),
            'std': np.std(ga_scores),
            'min': np.min(ga_scores),
            'max': np.max(ga_scores),
            'count': len(ga_scores)
        },
        'PSO': {
            'mean': np.mean(pso_scores),
            'std': np.std(pso_scores),
            'min': np.min(pso_scores),
            'max': np.max(pso_scores),
            'count': len(pso_scores)
        },
        'ACO': {
            'mean': np.mean(aco_scores),
            'std': np.std(aco_scores),
            'min': np.min(aco_scores),
            'max': np.max(aco_scores),
            'count': len(aco_scores)
        }
    }

    return {
        'scores': {'GA': ga_scores, 'PSO': pso_scores, 'ACO': aco_scores},
        't_tests': {
            'GA_vs_PSO': {'t_statistic': t_ga_pso, 'p_value': p_ga_pso},
            'GA_vs_ACO': {'t_statistic': t_ga_aco, 'p_value': p_ga_aco},
            'PSO_vs_ACO': {'t_statistic': t_pso_aco, 'p_value': p_pso_aco}
        },
        'descriptive_stats': stats_data
    }


def create_visualizations(analysis_results, output_dir='analysis_results'):
    """Create comprehensive visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    scores = analysis_results['scores']
    stats_data = analysis_results['descriptive_stats']

    # Set style for better presentation
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))

    # 1. Box plot comparison
    plt.subplot(2, 3, 1)
    data_to_plot = [scores['GA'], scores['PSO'], scores['ACO']]
    box_plot = plt.boxplot(data_to_plot, labels=[
                           'GA', 'PSO', 'ACO'], patch_artist=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Distribution of Final Scores by Algorithm',
              fontsize=14, fontweight='bold')
    plt.ylabel('Final Score (Lower is Better)')
    plt.grid(True, alpha=0.3)

    # 2. Violin plot
    plt.subplot(2, 3, 2)
    violin_data = []
    labels = []
    for algo in ['GA', 'PSO', 'ACO']:
        violin_data.extend(scores[algo])
        labels.extend([algo] * len(scores[algo]))

    df_violin = pd.DataFrame({'Algorithm': labels, 'Score': violin_data})
    sns.violinplot(x='Algorithm', y='Score', data=df_violin, palette=colors)
    plt.title('Score Distribution Density', fontsize=14, fontweight='bold')
    plt.ylabel('Final Score')
    plt.grid(True, alpha=0.3)

    # 3. Mean comparison with error bars
    plt.subplot(2, 3, 3)
    algorithms = list(stats_data.keys())
    means = [stats_data[algo]['mean'] for algo in algorithms]
    stds = [stats_data[algo]['std'] for algo in algorithms]

    bars = plt.bar(algorithms, means, yerr=stds,
                   capsize=10, color=colors, alpha=0.7)
    plt.title('Mean Scores with Standard Deviation',
              fontsize=14, fontweight='bold')
    plt.ylabel('Mean Final Score')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[0]/10,
                 f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')

    # 4. Cumulative distribution function
    plt.subplot(2, 3, 4)
    for algo, color in zip(['GA', 'PSO', 'ACO'], colors):
        sorted_scores = np.sort(scores[algo])
        y_vals = np.arange(len(sorted_scores)) / float(len(sorted_scores) - 1)
        plt.plot(sorted_scores, y_vals, label=algo, color=color, linewidth=2)

    plt.title('Cumulative Distribution Function',
              fontsize=14, fontweight='bold')
    plt.xlabel('Final Score')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Statistical significance chart
    plt.subplot(2, 3, 5)
    p_values = [
        analysis_results['t_tests']['GA_vs_PSO']['p_value'],
        analysis_results['t_tests']['GA_vs_ACO']['p_value'],
        analysis_results['t_tests']['PSO_vs_ACO']['p_value']
    ]
    comparisons = ['GA vs PSO', 'GA vs ACO', 'PSO vs ACO']

    bars = plt.bar(comparisons, p_values, color=[
                   'skyblue', 'lightcoral', 'lightgreen'])
    plt.axhline(y=0.05, color='red', linestyle='--',
                label='p=0.05 significance level')
    plt.title('Statistical Significance (p-values)',
              fontsize=14, fontweight='bold')
    plt.ylabel('p-value')
    plt.xticks(rotation=45)
    plt.legend()

    # Add p-value labels
    for bar, p_val in zip(bars, p_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'p={p_val:.4f}', ha='center', va='bottom', fontsize=10)

    # 6. Runtime comparison
    plt.subplot(2, 3, 6)
    runtime_data = []
    runtime_labels = []
    for algo in ['ga', 'pso', 'aco']:
        runtimes = [r['runtime_s'] for r in results[algo]]
        runtime_data.extend(runtimes)
        runtime_labels.extend([algo.upper()] * len(runtimes))

    df_runtime = pd.DataFrame(
        {'Algorithm': runtime_labels, 'Runtime': runtime_data})
    sns.boxplot(x='Algorithm', y='Runtime', data=df_runtime, palette=colors)
    plt.title('Algorithm Runtime Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def print_statistical_summary(analysis_results):
    """Print detailed statistical summary"""
    print("=" * 70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS RESULTS")
    print("=" * 70)

    # Descriptive statistics
    print("\nDESCRIPTIVE STATISTICS:")
    print("-" * 50)
    stats_data = analysis_results['descriptive_stats']
    for algo in ['GA', 'PSO', 'ACO']:
        stats = stats_data[algo]
        print(f"{algo}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std:  {stats['std']:.2f}")
        print(f"  Min:  {stats['min']:.2f}")
        print(f"  Max:  {stats['max']:.2f}")
        print(f"  N:    {stats['count']}")
        print()

    # T-test results
    print("STATISTICAL SIGNIFICANCE (T-TESTS):")
    print("-" * 50)
    t_tests = analysis_results['t_tests']
    for comparison, result in t_tests.items():
        p_value = result['p_value']
        significance = "*** SIGNIFICANT ***" if p_value < 0.05 else "Not significant"
        print(f"{comparison}:")
        print(f"  t-statistic: {result['t_statistic']:.4f}")
        print(f"  p-value:     {p_value:.6f} {significance}")
        print()

    # Conclusion
    print("CONCLUSION:")
    print("-" * 50)
    ga_mean = stats_data['GA']['mean']
    pso_mean = stats_data['PSO']['mean']
    aco_mean = stats_data['ACO']['mean']

    best_algo = min([('GA', ga_mean), ('PSO', pso_mean),
                    ('ACO', aco_mean)], key=lambda x: x[1])

    print(
        f"Best performing algorithm: {best_algo[0]} (mean score: {best_algo[1]:.2f})")

    if t_tests['GA_vs_PSO']['p_value'] < 0.05:
        if ga_mean < pso_mean:
            print("GA is statistically significantly better than PSO")
        else:
            print("PSO is statistically significantly better than GA")

    if t_tests['GA_vs_ACO']['p_value'] < 0.05:
        if ga_mean < aco_mean:
            print("GA is statistically significantly better than ACO")
        else:
            print("ACO is statistically significantly better than GA")


def save_detailed_results(analysis_results, output_dir='analysis_results'):
    """Save detailed results to CSV and JSON files"""
    # Save scores to CSV
    scores_df = pd.DataFrame(analysis_results['scores'])
    scores_df.to_csv(f'{output_dir}/all_scores.csv', index=False)

    # Save descriptive statistics to CSV
    stats_df = pd.DataFrame(analysis_results['descriptive_stats']).T
    stats_df.to_csv(f'{output_dir}/descriptive_statistics.csv')

    # Save t-test results to JSON
    with open(f'{output_dir}/t_test_results.json', 'w') as f:
        json.dump(analysis_results['t_tests'], f, indent=2)

    print(f"\nDetailed results saved to '{output_dir}' directory")


if __name__ == "__main__":
    # Load all results
    print("Loading results from directories...")
    results = load_results()

    # Check if we have enough data
    for algo, data in results.items():
        print(f"{algo.upper()}: {len(data)} runs")

    if all(len(data) >= 2 for data in results.values()):
        # Perform statistical analysis
        print("\nPerforming statistical analysis...")
        analysis_results = perform_statistical_analysis(results)

        # Print summary
        print_statistical_summary(analysis_results)

        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(analysis_results)

        # Save detailed results
        save_detailed_results(analysis_results)

    else:
        print("Error: Not enough data for statistical analysis. Need at least 2 runs per algorithm.")
