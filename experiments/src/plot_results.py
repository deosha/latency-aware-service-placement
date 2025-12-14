"""
Visualization and plotting utilities for experimental results.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# Use publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 12
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF


class ResultsPlotter:
    """Creates publication-quality plots from experimental results."""

    def __init__(self, results_file: str = "../results/all_results.json",
                 output_dir: str = "../plots"):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)

        self.df = self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for result in self.results:
            problem_name = result['problem_name']
            n = result['num_services']
            k = result['num_locations']

            for algo_name, algo_results in result['algorithms'].items():
                if 'error' in algo_results:
                    continue

                row = {
                    'problem': problem_name,
                    'n_services': n,
                    'k_locations': k,
                    'algorithm': algo_name,
                    'objective': algo_results['objective_value'],
                    'solve_time': algo_results['solve_time'],
                    'status': algo_results['solver_status'],
                    'gap': algo_results.get('gap', 0.0),
                    'critical_path': algo_results['critical_path_latency'],
                    'cpu_util': algo_results['avg_cpu_utilization'],
                    'mem_util': algo_results['avg_memory_utilization'],
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def plot_scalability(self, save_pdf: bool = True):
        """Plot scalability: objective value and solve time vs problem size."""
        scalability_df = self.df[self.df['problem'].str.contains('scalability')]

        if scalability_df.empty:
            print("No scalability results found")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Plot objective value
        for algo in scalability_df['algorithm'].unique():
            algo_df = scalability_df[scalability_df['algorithm'] == algo]
            algo_df = algo_df.sort_values('n_services')
            ax1.plot(algo_df['n_services'], algo_df['objective'],
                    marker='o', label=algo, linewidth=2, markersize=6)

        ax1.set_xlabel('Number of Services')
        ax1.set_ylabel('Total Weighted Latency (ms)')
        ax1.set_title('(a) Solution Quality vs Problem Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot solve time
        for algo in scalability_df['algorithm'].unique():
            algo_df = scalability_df[scalability_df['algorithm'] == algo]
            algo_df = algo_df.sort_values('n_services')
            ax2.plot(algo_df['n_services'], algo_df['solve_time'],
                    marker='s', label=algo, linewidth=2, markersize=6)

        ax2.set_xlabel('Number of Services')
        ax2.set_ylabel('Solve Time (seconds)')
        ax2.set_title('(b) Computational Time vs Problem Size')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_pdf:
            plt.savefig(self.output_dir / 'scalability.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(self.output_dir / 'scalability.png', bbox_inches='tight', dpi=300)
            print(f"✓ Saved scalability plot")

        plt.close()

    def plot_algorithm_comparison(self, save_pdf: bool = True):
        """Bar chart comparing algorithms across metrics."""
        # Aggregate metrics by algorithm
        agg_df = self.df.groupby('algorithm').agg({
            'objective': 'mean',
            'solve_time': 'mean',
            'cpu_util': 'mean',
            'mem_util': 'mean'
        }).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Objective value
        ax = axes[0, 0]
        ax.bar(agg_df['algorithm'], agg_df['objective'], color='steelblue', alpha=0.7)
        ax.set_ylabel('Avg Total Latency (ms)')
        ax.set_title('(a) Average Solution Quality')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Solve time
        ax = axes[0, 1]
        ax.bar(agg_df['algorithm'], agg_df['solve_time'], color='coral', alpha=0.7)
        ax.set_ylabel('Avg Solve Time (s)')
        ax.set_title('(b) Average Computational Time')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # CPU utilization
        ax = axes[1, 0]
        ax.bar(agg_df['algorithm'], agg_df['cpu_util'], color='green', alpha=0.7)
        ax.set_ylabel('CPU Utilization (%)')
        ax.set_title('(c) Average CPU Utilization')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Memory utilization
        ax = axes[1, 1]
        ax.bar(agg_df['algorithm'], agg_df['mem_util'], color='purple', alpha=0.7)
        ax.set_ylabel('Memory Utilization (%)')
        ax.set_title('(d) Average Memory Utilization')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_pdf:
            plt.savefig(self.output_dir / 'algorithm_comparison.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(self.output_dir / 'algorithm_comparison.png', bbox_inches='tight', dpi=300)
            print(f"✓ Saved algorithm comparison plot")

        plt.close()

    def plot_dag_structure_comparison(self, save_pdf: bool = True):
        """Compare performance across DAG structures."""
        dag_df = self.df[self.df['problem'].str.contains('dag_')]

        if dag_df.empty:
            print("No DAG structure results found")
            return

        # Extract DAG type from problem name
        dag_df['dag_type'] = dag_df['problem'].str.replace('dag_', '')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Group by DAG type and algorithm
        dag_types = dag_df['dag_type'].unique()
        algorithms = dag_df['algorithm'].unique()

        x = np.arange(len(dag_types))
        width = 0.15
        offset = -(len(algorithms) - 1) * width / 2

        # Plot objective values
        for i, algo in enumerate(algorithms):
            algo_df = dag_df[dag_df['algorithm'] == algo]
            values = [algo_df[algo_df['dag_type'] == dt]['objective'].values[0]
                     if len(algo_df[algo_df['dag_type'] == dt]) > 0 else 0
                     for dt in dag_types]
            ax1.bar(x + offset + i * width, values, width, label=algo, alpha=0.7)

        ax1.set_xlabel('DAG Structure')
        ax1.set_ylabel('Total Weighted Latency (ms)')
        ax1.set_title('(a) Solution Quality by DAG Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(dag_types)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Plot solve times
        for i, algo in enumerate(algorithms):
            algo_df = dag_df[dag_df['algorithm'] == algo]
            values = [algo_df[algo_df['dag_type'] == dt]['solve_time'].values[0]
                     if len(algo_df[algo_df['dag_type'] == dt]) > 0 else 0
                     for dt in dag_types]
            ax2.bar(x + offset + i * width, values, width, label=algo, alpha=0.7)

        ax2.set_xlabel('DAG Structure')
        ax2.set_ylabel('Solve Time (s)')
        ax2.set_title('(b) Computational Time by DAG Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(dag_types)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_pdf:
            plt.savefig(self.output_dir / 'dag_comparison.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(self.output_dir / 'dag_comparison.png', bbox_inches='tight', dpi=300)
            print(f"✓ Saved DAG comparison plot")

        plt.close()

    def plot_topology_comparison(self, save_pdf: bool = True):
        """Compare performance across network topologies."""
        topo_df = self.df[self.df['problem'].str.contains('topology_')]

        if topo_df.empty:
            print("No topology results found")
            return

        topo_df['topology'] = topo_df['problem'].str.replace('topology_', '')

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        topologies = topo_df['topology'].unique()
        algorithms = topo_df['algorithm'].unique()

        x = np.arange(len(topologies))
        width = 0.15
        offset = -(len(algorithms) - 1) * width / 2

        for i, algo in enumerate(algorithms):
            algo_df = topo_df[topo_df['algorithm'] == algo]
            values = [algo_df[algo_df['topology'] == tp]['objective'].values[0]
                     if len(algo_df[algo_df['topology'] == tp]) > 0 else 0
                     for tp in topologies]
            ax.bar(x + offset + i * width, values, width, label=algo, alpha=0.7)

        ax.set_xlabel('Network Topology')
        ax.set_ylabel('Total Weighted Latency (ms)')
        ax.set_title('Solution Quality by Network Topology')
        ax.set_xticks(x)
        ax.set_xticklabels(topologies)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_pdf:
            plt.savefig(self.output_dir / 'topology_comparison.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(self.output_dir / 'topology_comparison.png', bbox_inches='tight', dpi=300)
            print(f"✓ Saved topology comparison plot")

        plt.close()

    def plot_improvement_over_baseline(self, baseline: str = "FirstFit", save_pdf: bool = True):
        """Plot improvement of each algorithm over baseline."""
        if baseline not in self.df['algorithm'].values:
            print(f"Baseline {baseline} not found")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        problems = self.df['problem'].unique()
        algorithms = [a for a in self.df['algorithm'].unique() if a != baseline]

        improvements = {algo: [] for algo in algorithms}

        for problem in problems:
            problem_df = self.df[self.df['problem'] == problem]
            if baseline not in problem_df['algorithm'].values:
                continue

            baseline_obj = problem_df[problem_df['algorithm'] == baseline]['objective'].values[0]

            for algo in algorithms:
                if algo in problem_df['algorithm'].values:
                    algo_obj = problem_df[problem_df['algorithm'] == algo]['objective'].values[0]
                    improvement = (baseline_obj - algo_obj) / baseline_obj * 100
                    improvements[algo].append(improvement)

        # Plot as box plots
        data = [improvements[algo] for algo in algorithms]
        bp = ax.boxplot(data, labels=algorithms, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label=f'{baseline} baseline')
        ax.set_ylabel(f'Improvement over {baseline} (%)')
        ax.set_title(f'Latency Improvement Relative to {baseline}')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_pdf:
            plt.savefig(self.output_dir / f'improvement_over_{baseline}.pdf',
                       bbox_inches='tight', dpi=300)
            plt.savefig(self.output_dir / f'improvement_over_{baseline}.png',
                       bbox_inches='tight', dpi=300)
            print(f"✓ Saved improvement plot")

        plt.close()

    def create_all_plots(self):
        """Generate all plots."""
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60 + "\n")

        self.plot_scalability()
        self.plot_algorithm_comparison()
        self.plot_dag_structure_comparison()
        self.plot_topology_comparison()
        self.plot_improvement_over_baseline(baseline="Greedy")

        print(f"\n✓ All plots saved to {self.output_dir}")


def main():
    """Main plotting entry point."""
    plotter = ResultsPlotter(
        results_file="../results/all_results.json",
        output_dir="../plots"
    )
    plotter.create_all_plots()


if __name__ == "__main__":
    main()
