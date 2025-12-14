"""
Main experiment runner for service placement evaluation.
"""

import os
import json
import time
from typing import List, Dict
import numpy as np
import pandas as pd
from pathlib import Path

from problem import PlacementProblem, PlacementSolution
from benchmarks import BenchmarkGenerator, create_smart_factory_benchmark
from baselines import (RandomPlacement, GreedyPlacement, FirstFitPlacement, CloudOnlyPlacement)

# Import MILP solver only if Gurobi is available
try:
    from milp_solver import MILPSolver
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("WARNING: Gurobi not available, MILP solver disabled")


class ExperimentRunner:
    """Runs experiments and collects results."""

    def __init__(self, output_dir: str = "../results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_single_experiment(self,
                            problem: PlacementProblem,
                            problem_name: str,
                            algorithms: List[str] = None) -> Dict:
        """
        Run all algorithms on a single problem instance.

        Args:
            problem: Problem instance
            problem_name: Name for this instance
            algorithms: List of algorithm names to run

        Returns:
            Dictionary with results
        """
        if algorithms is None:
            algorithms = ["MILP", "Greedy", "FirstFit", "Random", "CloudOnly"]

        print(f"\n{'='*60}")
        print(f"Running experiment: {problem_name}")
        print(f"Services: {problem.num_services}, Locations: {problem.num_locations}")
        print(f"{'='*60}")

        results = {
            'problem_name': problem_name,
            'num_services': problem.num_services,
            'num_locations': problem.num_locations,
            'num_edges': len(problem.edge_frequencies),
            'algorithms': {}
        }

        # Run each algorithm
        for algo_name in algorithms:
            print(f"\nRunning {algo_name}...", end=" ", flush=True)

            try:
                if algo_name == "MILP" and GUROBI_AVAILABLE:
                    solver = MILPSolver(problem, time_limit=300, mip_gap=0.01)
                    solution = solver.solve()
                    stats = solver.get_solver_stats()
                elif algo_name == "Greedy":
                    solver = GreedyPlacement(problem)
                    solution = solver.solve()
                    stats = {}
                elif algo_name == "FirstFit":
                    solver = FirstFitPlacement(problem)
                    solution = solver.solve()
                    stats = {}
                elif algo_name == "Random":
                    solver = RandomPlacement(problem, seed=42)
                    solution = solver.solve(num_trials=1000)
                    stats = {}
                elif algo_name == "CloudOnly":
                    solver = CloudOnlyPlacement(problem)
                    solution = solver.solve()
                    stats = {}
                else:
                    print(f"SKIPPED (not available)")
                    continue

                # Collect metrics
                critical_path = solution.compute_critical_path_latency()
                utilization = solution.get_resource_utilization()
                validation = solution.validate_constraints()

                avg_cpu_util = np.mean([u['cpu_util_pct'] for u in utilization.values()])
                avg_mem_util = np.mean([u['memory_util_pct'] for u in utilization.values()])

                results['algorithms'][algo_name] = {
                    'objective_value': solution.objective_value,
                    'solve_time': solution.solve_time,
                    'solver_status': solution.solver_status,
                    'gap': solution.gap,
                    'critical_path_latency': critical_path,
                    'avg_cpu_utilization': avg_cpu_util,
                    'avg_memory_utilization': avg_mem_util,
                    'valid': validation['valid'],
                    'violations': validation['violations'],
                    'solver_stats': stats,
                    'placement': solution.placement
                }

                print(f"✓ Obj={solution.objective_value:.2f}, Time={solution.solve_time:.2f}s")

            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                results['algorithms'][algo_name] = {
                    'error': str(e)
                }

        self.results.append(results)
        return results

    def run_scalability_experiments(self, max_services: int = 50, step: int = 10):
        """
        Run scalability experiments varying number of services.

        Args:
            max_services: Maximum number of services
            step: Step size for service count
        """
        print("\n" + "="*80)
        print("SCALABILITY EXPERIMENTS")
        print("="*80)

        generator = BenchmarkGenerator(seed=42)

        for num_services in range(10, max_services + 1, step):
            num_locations = min(10, max(5, num_services // 5))

            problem = generator.generate_problem(
                num_services=num_services,
                num_locations=num_locations,
                dag_type="random",
                topology_type="hierarchical",
                resource_dist="uniform"
            )

            problem_name = f"scalability_n{num_services}_k{num_locations}"
            self.run_single_experiment(problem, problem_name)

    def run_dag_structure_experiments(self):
        """Compare performance across different DAG structures."""
        print("\n" + "="*80)
        print("DAG STRUCTURE EXPERIMENTS")
        print("="*80)

        generator = BenchmarkGenerator(seed=42)
        dag_types = ["chain", "tree", "diamond", "random"]

        for dag_type in dag_types:
            problem = generator.generate_problem(
                num_services=20,
                num_locations=8,
                dag_type=dag_type,
                topology_type="hierarchical",
                resource_dist="uniform"
            )

            problem_name = f"dag_{dag_type}"
            self.run_single_experiment(problem, problem_name)

    def run_topology_experiments(self):
        """Compare performance across different network topologies."""
        print("\n" + "="*80)
        print("TOPOLOGY EXPERIMENTS")
        print("="*80)

        generator = BenchmarkGenerator(seed=42)
        topologies = ["hierarchical", "mesh", "hybrid"]

        for topo_type in topologies:
            problem = generator.generate_problem(
                num_services=20,
                num_locations=8,
                dag_type="random",
                topology_type=topo_type,
                resource_dist="uniform"
            )

            problem_name = f"topology_{topo_type}"
            self.run_single_experiment(problem, problem_name)

    def run_smart_factory_experiment(self):
        """Run the smart factory benchmark from the paper."""
        print("\n" + "="*80)
        print("SMART FACTORY BENCHMARK")
        print("="*80)

        problem = create_smart_factory_benchmark()
        self.run_single_experiment(problem, "smart_factory")

    def save_results(self, filename: str = "results.json"):
        """Save all results to JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ Results saved to {output_file}")

    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame with summary of all results."""
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

        df = pd.DataFrame(rows)
        return df


def main():
    """Main experiment entry point."""
    runner = ExperimentRunner(output_dir="../results")

    # Run all experiment suites
    print("\n" + "="*80)
    print("LATENCY-AWARE SERVICE PLACEMENT EXPERIMENTS")
    print("="*80)

    # 1. Smart factory benchmark
    runner.run_smart_factory_experiment()

    # 2. Scalability experiments
    runner.run_scalability_experiments(max_services=50, step=10)

    # 3. DAG structure experiments
    runner.run_dag_structure_experiments()

    # 4. Topology experiments
    runner.run_topology_experiments()

    # Save results
    runner.save_results("all_results.json")

    # Create summary
    df = runner.create_summary_dataframe()
    df.to_csv(runner.output_dir / "summary.csv", index=False)
    print("\n✓ Summary saved to results/summary.csv")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  Avg Objective: {algo_df['objective'].mean():.2f}")
        print(f"  Avg Solve Time: {algo_df['solve_time'].mean():.3f}s")
        print(f"  Success Rate: {(algo_df['status'] == 'optimal').sum() / len(algo_df) * 100:.1f}%")

    # Compare MILP vs baselines
    if 'MILP' in df['algorithm'].values and 'Greedy' in df['algorithm'].values:
        print("\n" + "="*80)
        print("MILP vs GREEDY COMPARISON")
        print("="*80)

        for problem in df['problem'].unique():
            problem_df = df[df['problem'] == problem]
            if 'MILP' in problem_df['algorithm'].values and 'Greedy' in problem_df['algorithm'].values:
                milp_obj = problem_df[problem_df['algorithm'] == 'MILP']['objective'].values[0]
                greedy_obj = problem_df[problem_df['algorithm'] == 'Greedy']['objective'].values[0]
                improvement = (greedy_obj - milp_obj) / greedy_obj * 100
                print(f"{problem}: {improvement:.1f}% improvement")


if __name__ == "__main__":
    main()
