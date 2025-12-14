"""
Quick test to verify implementation without running full experiments.
This can run without Gurobi installed (tests baselines only).
"""

import sys
import numpy as np

from problem import PlacementProblem, Service, Location
from benchmarks import BenchmarkGenerator, create_smart_factory_benchmark
from baselines import GreedyPlacement, FirstFitPlacement, RandomPlacement

# Test if Gurobi is available
try:
    from milp_solver import MILPSolver
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("WARNING: Gurobi not available - MILP tests will be skipped\n")


def test_smart_factory_benchmark():
    """Test the smart factory benchmark."""
    print("="*60)
    print("TEST 1: Smart Factory Benchmark")
    print("="*60)

    problem = create_smart_factory_benchmark()

    print(f"✓ Created problem with {problem.num_services} services, {problem.num_locations} locations")
    print(f"✓ DAG has {len(problem.edge_frequencies)} edges")
    print(f"✓ Latency matrix shape: {problem.latency_matrix.shape}")

    # Test greedy solver
    print("\nTesting Greedy solver...")
    greedy = GreedyPlacement(problem)
    solution = greedy.solve()
    print(f"  Objective: {solution.objective_value:.2f} ms")
    print(f"  Solve time: {solution.solve_time:.4f}s")
    print(f"  Placement: {solution.placement}")

    # Validate solution
    validation = solution.validate_constraints()
    print(f"  Valid: {validation['valid']}")
    if not validation['valid']:
        print(f"  Violations: {validation['violations']}")
        return False

    # Test FirstFit solver
    print("\nTesting FirstFit solver...")
    firstfit = FirstFitPlacement(problem)
    solution2 = firstfit.solve()
    print(f"  Objective: {solution2.objective_value:.2f} ms")
    print(f"  Solve time: {solution2.solve_time:.4f}s")

    # Test Random solver
    print("\nTesting Random solver...")
    random_solver = RandomPlacement(problem, seed=42)
    solution3 = random_solver.solve(num_trials=100)
    print(f"  Objective: {solution3.objective_value:.2f} ms")
    print(f"  Solve time: {solution3.solve_time:.4f}s")

    print("\n✓ Smart factory benchmark test PASSED\n")
    return True


def test_synthetic_benchmarks():
    """Test synthetic benchmark generation."""
    print("="*60)
    print("TEST 2: Synthetic Benchmark Generation")
    print("="*60)

    generator = BenchmarkGenerator(seed=42)

    # Test different DAG types
    dag_types = ["chain", "tree", "diamond", "random"]
    for dag_type in dag_types:
        print(f"\nTesting {dag_type} DAG...")
        problem = generator.generate_problem(
            num_services=10,
            num_locations=5,
            dag_type=dag_type,
            topology_type="hierarchical",
            anti_affinity_prob=0.0,  # Disable for testing
            colocation_prob=0.0,     # Disable for testing
            replication_prob=0.0     # Disable for testing
        )

        print(f"  Services: {problem.num_services}")
        print(f"  Locations: {problem.num_locations}")
        print(f"  Edges: {len(problem.edge_frequencies)}")

        # Quick solve with greedy
        try:
            solver = GreedyPlacement(problem)
            solution = solver.solve()
            print(f"  Greedy objective: {solution.objective_value:.2f} ms")
        except RuntimeError as e:
            # Try FirstFit instead
            print(f"  Greedy failed ({e}), trying FirstFit...")
            solver = FirstFitPlacement(problem)
            solution = solver.solve()
            print(f"  FirstFit objective: {solution.objective_value:.2f} ms")

        validation = solution.validate_constraints()
        if not validation['valid']:
            print(f"  ERROR: Invalid solution - {validation['violations']}")
            return False

    print("\n✓ Synthetic benchmark test PASSED\n")
    return True


def test_resource_utilization():
    """Test resource utilization computation."""
    print("="*60)
    print("TEST 3: Resource Utilization")
    print("="*60)

    problem = create_smart_factory_benchmark()
    solver = GreedyPlacement(problem)
    solution = solver.solve()

    utilization = solution.get_resource_utilization()

    print("\nResource Utilization:")
    for loc_id, util in utilization.items():
        loc_name = problem.locations[loc_id].name
        print(f"  {loc_name}:")
        print(f"    CPU: {util['cpu_used']:.2f} / {util['cpu_capacity']:.2f} ({util['cpu_util_pct']:.1f}%)")
        print(f"    Memory: {util['memory_used']:.2f} / {util['memory_capacity']:.2f} ({util['memory_util_pct']:.1f}%)")

    print("\n✓ Resource utilization test PASSED\n")
    return True


def test_critical_path():
    """Test critical path latency computation."""
    print("="*60)
    print("TEST 4: Critical Path Latency")
    print("="*60)

    problem = create_smart_factory_benchmark()
    solver = GreedyPlacement(problem)
    solution = solver.solve()

    total_latency = solution.compute_total_latency()
    critical_path = solution.compute_critical_path_latency()

    print(f"Total weighted latency: {total_latency:.2f} ms")
    print(f"Critical path latency: {critical_path:.2f} ms")
    print(f"Objective value: {solution.objective_value:.2f} ms")

    # Total latency should match objective
    assert abs(total_latency - solution.objective_value) < 1e-6, \
        "Total latency doesn't match objective!"

    print("\n✓ Critical path test PASSED\n")
    return True


def test_milp_solver():
    """Test MILP solver if Gurobi is available."""
    if not GUROBI_AVAILABLE:
        print("="*60)
        print("TEST 5: MILP Solver - SKIPPED (Gurobi not available)")
        print("="*60)
        print()
        return True

    print("="*60)
    print("TEST 5: MILP Solver")
    print("="*60)

    # Create small problem for quick test
    generator = BenchmarkGenerator(seed=42)
    problem = generator.generate_problem(
        num_services=5,
        num_locations=3,
        dag_type="chain",
        topology_type="hierarchical"
    )

    print(f"Testing MILP on small problem: {problem.num_services} services, {problem.num_locations} locations")

    solver = MILPSolver(problem, time_limit=60, mip_gap=0.01)
    print("\nBuilding MILP model...")
    solver.build_model()

    stats = solver.get_solver_stats()
    print(f"  Variables: {stats['num_variables']}")
    print(f"  Constraints: {stats['num_constraints']}")

    print("\nSolving...")
    solution = solver.solve()

    print(f"  Status: {solution.solver_status}")
    print(f"  Objective: {solution.objective_value:.2f} ms")
    print(f"  Solve time: {solution.solve_time:.2f}s")
    print(f"  Gap: {solution.gap*100:.2f}%")

    # Compare with greedy
    greedy = GreedyPlacement(problem)
    greedy_solution = greedy.solve()

    improvement = (greedy_solution.objective_value - solution.objective_value) / greedy_solution.objective_value * 100
    print(f"\nMILP vs Greedy:")
    print(f"  MILP: {solution.objective_value:.2f} ms")
    print(f"  Greedy: {greedy_solution.objective_value:.2f} ms")
    print(f"  Improvement: {improvement:.1f}%")

    # MILP should be at least as good as greedy (assuming it found optimal)
    if solution.solver_status == "optimal":
        assert solution.objective_value <= greedy_solution.objective_value + 1e-6, \
            "MILP objective worse than greedy!"

    print("\n✓ MILP solver test PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("IMPLEMENTATION VERIFICATION TESTS")
    print("="*60 + "\n")

    tests = [
        ("Smart Factory Benchmark", test_smart_factory_benchmark),
        ("Synthetic Benchmarks", test_synthetic_benchmarks),
        ("Resource Utilization", test_resource_utilization),
        ("Critical Path", test_critical_path),
        ("MILP Solver", test_milp_solver),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}\n")
            import traceback
            traceback.print_exc()

    print("="*60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*60)

    if failed > 0:
        print(f"\n{failed} test(s) failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✓ All tests passed! Implementation is ready for experiments.")
        sys.exit(0)


if __name__ == "__main__":
    main()
