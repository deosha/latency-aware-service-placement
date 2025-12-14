# Latency-Aware Service Placement Experiments

This directory contains the implementation and experimental evaluation for the paper:
**"Latency-Aware Service Placement on the Fog-Edge-Cloud Continuum via Integer Programming"**

## Directory Structure

```
experiments/
├── src/
│   ├── problem.py           # Data structures for problem instances
│   ├── milp_solver.py       # Gurobi MILP solver implementation
│   ├── baselines.py         # Baseline algorithms (Greedy, Random, FirstFit, CloudOnly)
│   ├── benchmarks.py        # Synthetic and real benchmark generators
│   ├── run_experiments.py   # Main experiment runner
│   └── plot_results.py      # Plotting and visualization
├── benchmarks/              # Saved benchmark instances
├── results/                 # Experimental results (JSON, CSV)
├── plots/                   # Generated plots (PDF, PNG)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Gurobi Optimizer** (free academic license available at https://www.gurobi.com/academia/)

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify Gurobi installation
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

## Usage

### Running Experiments

```bash
cd src
python run_experiments.py
```

This will:
1. Run the smart factory benchmark from the paper
2. Run scalability experiments (10-50 services)
3. Run DAG structure experiments (chain, tree, diamond, random)
4. Run topology experiments (hierarchical, mesh, hybrid)
5. Save results to `../results/all_results.json`

### Generating Plots

```bash
cd src
python plot_results.py
```

This will generate publication-quality plots in `../plots/`:
- `scalability.pdf` - Solution quality and solve time vs problem size
- `algorithm_comparison.pdf` - Algorithm comparison across metrics
- `dag_comparison.pdf` - Performance by DAG structure
- `topology_comparison.pdf` - Performance by network topology
- `improvement_over_Greedy.pdf` - Improvement over baseline

### Using Individual Components

```python
from problem import PlacementProblem
from benchmarks import create_smart_factory_benchmark
from milp_solver import solve_milp
from baselines import GreedyPlacement

# Load benchmark
problem = create_smart_factory_benchmark()

# Solve with MILP
milp_solution = solve_milp(problem, time_limit=300)
print(f"MILP Objective: {milp_solution.objective_value:.2f}")

# Solve with Greedy
greedy_solver = GreedyPlacement(problem)
greedy_solution = greedy_solver.solve()
print(f"Greedy Objective: {greedy_solution.objective_value:.2f}")

# Compare
improvement = (greedy_solution.objective_value - milp_solution.objective_value) / greedy_solution.objective_value * 100
print(f"MILP Improvement: {improvement:.1f}%")
```

### Creating Custom Benchmarks

```python
from benchmarks import BenchmarkGenerator

generator = BenchmarkGenerator(seed=42)

# Generate custom problem
problem = generator.generate_problem(
    num_services=30,
    num_locations=10,
    dag_type="random",       # or "chain", "tree", "diamond"
    topology_type="hierarchical",  # or "mesh", "hybrid"
    resource_dist="realistic",     # or "uniform", "skewed"
    anti_affinity_prob=0.1,
    replication_prob=0.1
)
```

## Algorithms Implemented

1. **MILP (Gurobi)** - Optimal solution via mixed-integer linear programming
2. **Greedy** - Greedy heuristic in topological order
3. **FirstFit** - First-fit heuristic in topological order
4. **Random** - Best of 1000 random feasible placements
5. **CloudOnly** - Place all services at cloud (baseline)

## Experimental Design

### Scalability Experiments
- Variable: Number of services (10, 20, 30, 40, 50)
- Fixed: 5-10 locations, random DAG, hierarchical topology
- Metrics: Objective value, solve time, optimality gap

### DAG Structure Experiments
- Variable: DAG structure (chain, tree, diamond, random)
- Fixed: 20 services, 8 locations
- Metrics: Objective value, solve time

### Topology Experiments
- Variable: Network topology (hierarchical, mesh, hybrid)
- Fixed: 20 services, 8 locations, random DAG
- Metrics: Objective value, latency distribution

### Smart Factory Benchmark
- Realistic scenario from paper (Section 4.1)
- 5 services, 5 locations, linear pipeline
- Validates implementation against paper results

## Expected Results

Based on the paper's theoretical analysis:

- **MILP** should achieve optimal or near-optimal solutions (MIP gap < 1%)
- **Greedy** should achieve 20-40% worse than MILP on average
- **MILP solve time** grows exponentially with problem size
- **Greedy solve time** remains under 1 second for all instances
- **Resource utilization** should be higher for MILP (better packing)

## Troubleshooting

### Gurobi License Error
```
GurobiError: Model too large for size-limited license
```
Solution: Apply for free academic license at https://www.gurobi.com/academia/

### Memory Error on Large Instances
```
MemoryError: Unable to allocate array
```
Solution: Reduce `max_services` in `run_scalability_experiments()` or increase time limit

### No Feasible Solution Found
```
RuntimeError: Solver failed with status: infeasible
```
Solution: Check resource capacity settings in benchmark generator, increase location capacities

## Citation

If you use this code, please cite:

```bibtex
@article{latency_aware_placement_2024,
  title={Latency-Aware Service Placement on the Fog-Edge-Cloud Continuum via Integer Programming},
  author={[Authors]},
  journal={Journal of Cloud Computing},
  year={2024}
}
```

## License

[License information]
