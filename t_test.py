#!/usr/bin/env python3
import os
import json
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# =============== IO / LOADING =================

def load_results(base_dir='results'):
    """Load all results from the directories."""
    algorithms = ['ga', 'pso', 'aco']
    results = {algo: [] for algo in algorithms}

    for algo in algorithms:
        for dir_path in Path(base_dir).glob(f'{algo}_*'):
            try:
                summary_path = dir_path / 'run_summary.json'
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        data = json.load(f)
                        results[algo].append({
                            'final_score': float(data['final_score']),
                            'runtime_s': float(data['runtime_s']),
                            'iterations': int(data['iterations']),
                            'directory': str(dir_path)
                        })
            except Exception as e:
                print(f"Error reading {summary_path}: {e}")

    return results


# =============== STATS HELPERS =================

def _sample_stats(x):
    x = np.asarray(x, dtype=float)
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        min=float(np.min(x)),
        max=float(np.max(x)),
        count=int(len(x)),
    )


def _cohen_d(a, b):
    """Cohen's d using pooled SD (unbiased for equal variances)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
    # if both variances are zero, effect size is purely difference / ~0 -> undefined
    if sa2 == 0 and sb2 == 0:
        return float('nan')
    sp = sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2)
              ) if (na + nb - 2) > 0 else float('nan')
    if sp == 0:
        return float('inf') if np.mean(a) != np.mean(b) else 0.0
    return (np.mean(a) - np.mean(b)) / sp


def _hedges_g(a, b):
    """Hedges' g with small-sample correction."""
    d = _cohen_d(a, b)
    if not isfinite(d):
        return d
    na, nb = len(a), len(b)
    df = na + nb - 2
    if df <= 0:
        return float('nan')
    J = 1 - (3 / (4 * df - 1))  # correction factor
    return d * J


def _safe_ttest(a, b):
    """
    Welch's t-test. If both groups are constant (std=0), returns (nan, nan).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.std(a, ddof=1) if len(a) > 1 else 0.0
    sb = np.std(b, ddof=1) if len(b) > 1 else 0.0
    if sa == 0 and sb == 0:
        return float('nan'), float('nan')
    return stats.ttest_ind(a, b, equal_var=False)


# =============== ANALYSIS =================

def perform_statistical_analysis(results):
    """Perform Welch t-tests, descriptive statistics, effect sizes."""
    ga_scores = [r['final_score'] for r in results['ga']]
    pso_scores = [r['final_score'] for r in results['pso']]
    aco_scores = [r['final_score'] for r in results['aco']]

    # Welch t-tests
    t_ga_pso, p_ga_pso = _safe_ttest(ga_scores, pso_scores)
    t_ga_aco, p_ga_aco = _safe_ttest(ga_scores, aco_scores)
    t_pso_aco, p_pso_aco = _safe_ttest(pso_scores, aco_scores)

    # Effect sizes
    d_ga_pso = _cohen_d(ga_scores, pso_scores)
    g_ga_pso = _hedges_g(ga_scores, pso_scores)
    d_ga_aco = _cohen_d(ga_scores, aco_scores)
    g_ga_aco = _hedges_g(ga_scores, aco_scores)
    d_pso_aco = _cohen_d(pso_scores, aco_scores)
    g_pso_aco = _hedges_g(pso_scores, aco_scores)

    stats_data = {
        'GA':  _sample_stats(ga_scores),
        'PSO': _sample_stats(pso_scores),
        'ACO': _sample_stats(aco_scores),
    }

    t_tests = {
        'GA_vs_PSO': {'t_statistic': float(t_ga_pso), 'p_value': float(p_ga_pso), 'cohen_d': float(d_ga_pso), 'hedges_g': float(g_ga_pso)},
        'GA_vs_ACO': {'t_statistic': float(t_ga_aco), 'p_value': float(p_ga_aco), 'cohen_d': float(d_ga_aco), 'hedges_g': float(g_ga_aco)},
        'PSO_vs_ACO': {'t_statistic': float(t_pso_aco), 'p_value': float(p_pso_aco), 'cohen_d': float(d_pso_aco), 'hedges_g': float(g_pso_aco)},
    }

    return {
        'scores': {'GA': ga_scores, 'PSO': pso_scores, 'ACO': aco_scores},
        't_tests': t_tests,
        'descriptive_stats': stats_data
    }


# =============== VISUALIZATION =================

def create_visualizations(analysis_results, raw_results, output_dir='analysis_results'):
    """
    Create comprehensive visualizations:
      (1) Box + strip (raw points) of scores
      (2) Violin density
      (3) Mean ± 95% CI bars
      (4) ECDF
      (5) -log10(p) with α=0.05 line
      (6) Runtime boxplot
    """
    os.makedirs(output_dir, exist_ok=True)
    scores = analysis_results['scores']
    stats_data = analysis_results['descriptive_stats']
    ttests = analysis_results['t_tests']

    # Prepare dataframes
    df_scores = pd.DataFrame({
        'Algorithm': np.concatenate([
            np.repeat('GA',  len(scores['GA'])),
            np.repeat('PSO', len(scores['PSO'])),
            np.repeat('ACO', len(scores['ACO'])),
        ]),
        'Score': np.concatenate([scores['GA'], scores['PSO'], scores['ACO']])
    })

    runtime_data = []
    for algo_key in ['ga', 'pso', 'aco']:
        runtime_data.extend([{'Algorithm': algo_key.upper(
        ), 'Runtime': r['runtime_s']} for r in raw_results[algo_key]])
    df_runtime = pd.DataFrame(runtime_data)

    # Visual style
    sns.set_context("talk")
    sns.set_style("whitegrid")
    colors = ['#4C78A8', '#72B7B2', '#F58518']  # consistent palette

    fig = plt.figure(figsize=(21, 15))

    # 1. Box + strip
    ax1 = plt.subplot(2, 3, 1)
    sns.boxplot(data=df_scores, x='Algorithm', y='Score',
                palette=colors, showfliers=False, ax=ax1)
    sns.stripplot(data=df_scores, x='Algorithm', y='Score',
                  color='black', alpha=0.5, jitter=0.15, ax=ax1)
    ax1.set_title('Final Scores: Box + Raw Points')
    ax1.set_ylabel('Final Score (lower = better)')

    # 2. Violin
    ax2 = plt.subplot(2, 3, 2)
    sns.violinplot(data=df_scores, x='Algorithm', y='Score',
                   palette=colors, inner='box', ax=ax2)
    ax2.set_title('Score Distribution (Violin)')
    ax2.set_ylabel('Final Score')

    # 3. Mean ± 95% CI
    ax3 = plt.subplot(2, 3, 3)
    algos = list(stats_data.keys())
    means = np.array([stats_data[a]['mean'] for a in algos], dtype=float)
    stds = np.array([stats_data[a]['std'] for a in algos], dtype=float)
    ns = np.array([stats_data[a]['count'] for a in algos], dtype=int)
    se = stds / np.sqrt(np.maximum(ns, 1))
    ci95 = 1.96 * se  # approx 95% CI

    bars = ax3.bar(algos, means, yerr=ci95, capsize=8,
                   alpha=0.85, color=colors)
    for bar, mean in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_title('Mean Final Score ± 95% CI')
    ax3.set_ylabel('Mean Final Score')

    # 4. ECDF
    ax4 = plt.subplot(2, 3, 4)
    for algo, c in zip(['GA', 'PSO', 'ACO'], colors):
        x = np.sort(scores[algo])
        y = np.arange(1, len(x)+1) / len(x)
        ax4.plot(x, y, label=algo, linewidth=2.0, color=c)
    ax4.set_title('Empirical CDF of Final Scores')
    ax4.set_xlabel('Final Score')
    ax4.set_ylabel('Cumulative Probability')
    ax4.legend()

    # 5. -log10(p) bar with 0.05 line
    ax5 = plt.subplot(2, 3, 5)
    comparisons = ['GA vs PSO', 'GA vs ACO', 'PSO vs ACO']
    p_values = [
        ttests['GA_vs_PSO']['p_value'],
        ttests['GA_vs_ACO']['p_value'],
        ttests['PSO_vs_ACO']['p_value'],
    ]
    # handle nan and very small numbers
    safe_p = [p if isfinite(p) and p > 0 else np.nan for p in p_values]
    logp = [-np.log10(p) if isfinite(p) and p >
            0 else np.nan for p in p_values]

    bars = ax5.bar(comparisons, logp, color=[
                   '#6baed6', '#fd8d3c', '#74c476'], alpha=0.9)
    # line for alpha=0.05 => -log10(0.05) ~ 1.3010
    ax5.axhline(y=-np.log10(0.05), color='red',
                linestyle='--', label='α = 0.05')
    for bar, p in zip(bars, p_values):
        label = 'nan' if (not isfinite(p) or p <= 0) else f"p={p:.2e}"
        ax5.text(bar.get_x() + bar.get_width()/2, (bar.get_height() if isfinite(bar.get_height()) else 0) + 0.1,
                 label, ha='center', va='bottom', fontsize=10)
    ax5.set_title('Significance: -log10(p)')
    ax5.set_ylabel('-log10(p)')
    ax5.legend()

    # 6. Runtime boxplot
    ax6 = plt.subplot(2, 3, 6)
    if not df_runtime.empty:
        sns.boxplot(data=df_runtime, x='Algorithm', y='Runtime',
                    palette=colors, showfliers=False, ax=ax6)
        sns.stripplot(data=df_runtime, x='Algorithm', y='Runtime',
                      color='black', alpha=0.5, jitter=0.15, ax=ax6)
    ax6.set_title('Runtime Distribution')
    ax6.set_ylabel('Runtime (seconds)')

    plt.tight_layout()
    out = f'{output_dir}/comprehensive_analysis.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {out}")
    plt.close(fig)


# =============== REPORTING =================

def print_statistical_summary(analysis_results):
    """Print detailed statistical summary."""
    print("=" * 75)
    print("COMPREHENSIVE STATISTICAL ANALYSIS RESULTS")
    print("=" * 75)

    print("\nDESCRIPTIVE STATISTICS:")
    print("-" * 50)
    stats_data = analysis_results['descriptive_stats']
    for algo in ['GA', 'PSO', 'ACO']:
        s = stats_data[algo]
        print(f"{algo}:")
        print(f"  Mean: {s['mean']:.2f}")
        print(f"  Std:  {s['std']:.2f}")
        print(f"  Min:  {s['min']:.2f}")
        print(f"  Max:  {s['max']:.2f}")
        print(f"  N:    {s['count']}")
        print()

    print("STATISTICAL SIGNIFICANCE (Welch's t-tests):")
    print("-" * 50)
    for comparison, res in analysis_results['t_tests'].items():
        tval = res['t_statistic']
        pval = res['p_value']
        dval = res['cohen_d']
        gval = res['hedges_g']
        p_str = f"{pval:.3e}" if isfinite(pval) else "nan"
        t_str = f"{tval:.4f}" if isfinite(tval) else "nan"
        d_str = f"{dval:.3f}" if isfinite(dval) else "nan"
        g_str = f"{gval:.3f}" if isfinite(gval) else "nan"
        signif = ("*** SIGNIFICANT ***" if isfinite(pval) and pval < 0.05 else
                  "Not significant or undefined")
        print(f"{comparison}:")
        print(f"  t-statistic: {t_str}")
        print(f"  p-value:     {p_str}  {signif}")
        print(f"  effect size: Cohen's d = {d_str}, Hedges' g = {g_str}")
        print()

    print("CONCLUSION:")
    print("-" * 50)
    ga_mean = stats_data['GA']['mean']
    pso_mean = stats_data['PSO']['mean']
    aco_mean = stats_data['ACO']['mean']

    best_algo = min([('GA', ga_mean), ('PSO', pso_mean),
                    ('ACO', aco_mean)], key=lambda x: x[1])
    print(
        f"Best performing algorithm (lower is better): {best_algo[0]} (mean score: {best_algo[1]:.2f})")

    # Pairwise verbal conclusions (only where p < 0.05 and defined)
    tests = analysis_results['t_tests']
    if isfinite(tests['GA_vs_PSO']['p_value']) and tests['GA_vs_PSO']['p_value'] < 0.05:
        print("GA vs PSO: statistically significant difference.")
    if isfinite(tests['GA_vs_ACO']['p_value']) and tests['GA_vs_ACO']['p_value'] < 0.05:
        print("GA vs ACO: statistically significant difference.")
    if isfinite(tests['PSO_vs_ACO']['p_value']) and tests['PSO_vs_ACO']['p_value'] < 0.05:
        print("PSO vs ACO: statistically significant difference.")


def save_detailed_results(analysis_results, output_dir='analysis_results'):
    """Save detailed results to CSV and JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # Scores
    scores_df = pd.DataFrame(analysis_results['scores'])
    scores_path = f'{output_dir}/all_scores.csv'
    scores_df.to_csv(scores_path, index=False)

    # Descriptive stats
    stats_df = pd.DataFrame(analysis_results['descriptive_stats']).T
    desc_path = f'{output_dir}/descriptive_statistics.csv'
    stats_df.to_csv(desc_path)

    # T-test + effect sizes
    with open(f'{output_dir}/t_test_results.json', 'w') as f:
        json.dump(analysis_results['t_tests'], f, indent=2)

    print(f"\nSaved: {scores_path}")
    print(f"Saved: {desc_path}")
    print(f"Saved: {output_dir}/t_test_results.json")


# =============== MAIN =================

if __name__ == "__main__":
    print("Loading results from directories...")
    results = load_results()

    for algo, data in results.items():
        print(f"{algo.upper()}: {len(data)} runs")

    # Need at least 2 runs per algorithm for variance estimates / plots to be meaningful
    if all(len(data) >= 2 for data in results.values()):
        print("\nPerforming statistical analysis...")
        analysis_results = perform_statistical_analysis(results)

        print_statistical_summary(analysis_results)

        print("\nCreating visualizations...")
        create_visualizations(analysis_results, raw_results=results)

        save_detailed_results(analysis_results)
    else:
        print("Error: Not enough data for statistical analysis. Need at least 2 runs per algorithm.")
