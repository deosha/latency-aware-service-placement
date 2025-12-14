"""
Baseline algorithms for comparison with MILP.
"""

import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import random

from problem import PlacementProblem, PlacementSolution


class RandomPlacement:
    """Random placement baseline."""

    def __init__(self, problem: PlacementProblem, seed: int = 42):
        self.problem = problem
        self.seed = seed

    def solve(self, num_trials: int = 100) -> PlacementSolution:
        """
        Try random placements and return best feasible one.

        Args:
            num_trials: Number of random trials

        Returns:
            Best feasible solution found
        """
        start_time = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)

        best_solution = None
        best_objective = float('inf')

        for trial in range(num_trials):
            # Generate random placement
            placement = {}
            for service in self.problem.services:
                placement[service.id] = random.randint(0, self.problem.num_locations - 1)

            # Check feasibility
            if not self._is_feasible(placement):
                continue

            # Compute objective
            obj_value = self._compute_objective(placement)

            if obj_value < best_objective:
                best_objective = obj_value
                best_solution = placement

        solve_time = time.time() - start_time

        if best_solution is None:
            raise RuntimeError("No feasible random placement found")

        return PlacementSolution(
            problem=self.problem,
            placement=best_solution,
            objective_value=best_objective,
            solve_time=solve_time,
            solver_status="feasible"
        )

    def _is_feasible(self, placement: Dict[int, int]) -> bool:
        """Check if placement satisfies resource constraints."""
        # Check resource capacity
        cpu_usage = [0.0] * self.problem.num_locations
        mem_usage = [0.0] * self.problem.num_locations

        for service in self.problem.services:
            loc_id = placement[service.id]
            cpu_usage[loc_id] += service.cpu_demand
            mem_usage[loc_id] += service.memory_demand

        for i, location in enumerate(self.problem.locations):
            if cpu_usage[i] > location.cpu_capacity + 1e-6:
                return False
            if mem_usage[i] > location.memory_capacity + 1e-6:
                return False

        # Check anti-affinity
        for (s1, s2) in self.problem.anti_affinity:
            if placement.get(s1) == placement.get(s2):
                return False

        # Check colocation
        for (s1, s2) in self.problem.colocation:
            if placement.get(s1) != placement.get(s2):
                return False

        return True

    def _compute_objective(self, placement: Dict[int, int]) -> float:
        """Compute objective value for placement."""
        total = 0.0
        for (src_id, tgt_id), freq in self.problem.edge_frequencies.items():
            src_loc = placement[src_id]
            tgt_loc = placement[tgt_id]
            latency = self.problem.get_latency(src_loc, tgt_loc)
            total += freq * latency
        return total


class GreedyPlacement:
    """Greedy placement based on topological order."""

    def __init__(self, problem: PlacementProblem):
        self.problem = problem

    def solve(self) -> PlacementSolution:
        """
        Greedy placement: process services in topological order,
        place each at location minimizing incremental latency.

        Returns:
            PlacementSolution
        """
        start_time = time.time()

        # Get topological order of services
        topo_order = list(nx.topological_sort(self.problem.service_dag))

        placement = {}
        cpu_usage = [0.0] * self.problem.num_locations
        mem_usage = [0.0] * self.problem.num_locations

        for service_id in topo_order:
            service = self.problem.get_service_by_id(service_id)

            # Find best feasible location for this service
            best_loc = None
            best_cost = float('inf')

            for loc_id, location in enumerate(self.problem.locations):
                # Check resource feasibility
                if cpu_usage[loc_id] + service.cpu_demand > location.cpu_capacity + 1e-6:
                    continue
                if mem_usage[loc_id] + service.memory_demand > location.memory_capacity + 1e-6:
                    continue

                # Check anti-affinity
                feasible = True
                for (s1, s2) in self.problem.anti_affinity:
                    if s1 == service_id and s2 in placement and placement[s2] == loc_id:
                        feasible = False
                        break
                    if s2 == service_id and s1 in placement and placement[s1] == loc_id:
                        feasible = False
                        break
                if not feasible:
                    continue

                # Check colocation
                for (s1, s2) in self.problem.colocation:
                    if s1 == service_id and s2 in placement and placement[s2] != loc_id:
                        feasible = False
                        break
                    if s2 == service_id and s1 in placement and placement[s1] != loc_id:
                        feasible = False
                        break
                if not feasible:
                    continue

                # Compute incremental cost
                cost = self._compute_incremental_cost(service_id, loc_id, placement)

                if cost < best_cost:
                    best_cost = cost
                    best_loc = loc_id

            if best_loc is None:
                raise RuntimeError(f"No feasible location found for service {service_id}")

            # Place service
            placement[service_id] = best_loc
            cpu_usage[best_loc] += service.cpu_demand
            mem_usage[best_loc] += service.memory_demand

        solve_time = time.time() - start_time

        # Compute total objective
        obj_value = 0.0
        for (src_id, tgt_id), freq in self.problem.edge_frequencies.items():
            src_loc = placement[src_id]
            tgt_loc = placement[tgt_id]
            latency = self.problem.get_latency(src_loc, tgt_loc)
            obj_value += freq * latency

        return PlacementSolution(
            problem=self.problem,
            placement=placement,
            objective_value=obj_value,
            solve_time=solve_time,
            solver_status="feasible"
        )

    def _compute_incremental_cost(self, service_id: int, loc_id: int,
                                  current_placement: Dict[int, int]) -> float:
        """Compute incremental latency cost of placing service at location."""
        cost = 0.0

        # Cost from predecessors
        for pred_id in self.problem.service_dag.predecessors(service_id):
            if pred_id in current_placement:
                pred_loc = current_placement[pred_id]
                freq = self.problem.get_edge_frequency(pred_id, service_id)
                latency = self.problem.get_latency(pred_loc, loc_id)
                cost += freq * latency

        # Estimated cost to successors (assume they'll be at same location)
        for succ_id in self.problem.service_dag.successors(service_id):
            freq = self.problem.get_edge_frequency(service_id, succ_id)
            # Assume successor at same location (optimistic)
            latency = 0.0
            cost += freq * latency

        return cost


class FirstFitPlacement:
    """First-fit placement: place each service at first feasible location."""

    def __init__(self, problem: PlacementProblem):
        self.problem = problem

    def solve(self) -> PlacementSolution:
        """
        First-fit placement in topological order.

        Returns:
            PlacementSolution
        """
        start_time = time.time()

        topo_order = list(nx.topological_sort(self.problem.service_dag))

        placement = {}
        cpu_usage = [0.0] * self.problem.num_locations
        mem_usage = [0.0] * self.problem.num_locations

        for service_id in topo_order:
            service = self.problem.get_service_by_id(service_id)

            # Find first feasible location
            placed = False
            for loc_id, location in enumerate(self.problem.locations):
                # Check resource feasibility
                if cpu_usage[loc_id] + service.cpu_demand > location.cpu_capacity + 1e-6:
                    continue
                if mem_usage[loc_id] + service.memory_demand > location.memory_capacity + 1e-6:
                    continue

                # Check constraints (same as greedy)
                feasible = True
                for (s1, s2) in self.problem.anti_affinity:
                    if s1 == service_id and s2 in placement and placement[s2] == loc_id:
                        feasible = False
                        break
                    if s2 == service_id and s1 in placement and placement[s1] == loc_id:
                        feasible = False
                        break
                if not feasible:
                    continue

                for (s1, s2) in self.problem.colocation:
                    if s1 == service_id and s2 in placement and placement[s2] != loc_id:
                        feasible = False
                        break
                    if s2 == service_id and s1 in placement and placement[s1] != loc_id:
                        feasible = False
                        break
                if not feasible:
                    continue

                # Place here
                placement[service_id] = loc_id
                cpu_usage[loc_id] += service.cpu_demand
                mem_usage[loc_id] += service.memory_demand
                placed = True
                break

            if not placed:
                raise RuntimeError(f"No feasible location found for service {service_id}")

        solve_time = time.time() - start_time

        # Compute objective
        obj_value = 0.0
        for (src_id, tgt_id), freq in self.problem.edge_frequencies.items():
            src_loc = placement[src_id]
            tgt_loc = placement[tgt_id]
            latency = self.problem.get_latency(src_loc, tgt_loc)
            obj_value += freq * latency

        return PlacementSolution(
            problem=self.problem,
            placement=placement,
            objective_value=obj_value,
            solve_time=solve_time,
            solver_status="feasible"
        )


class CloudOnlyPlacement:
    """Baseline: place everything at cloud (highest-tier location)."""

    def __init__(self, problem: PlacementProblem):
        self.problem = problem

    def solve(self) -> PlacementSolution:
        """
        Place all services at the cloud location (highest tier).

        Returns:
            PlacementSolution
        """
        start_time = time.time()

        # Find cloud location (highest tier)
        cloud_loc = max(self.problem.locations, key=lambda loc: loc.tier)

        # Check if all services fit
        total_cpu = sum(s.cpu_demand for s in self.problem.services)
        total_mem = sum(s.memory_demand for s in self.problem.services)

        if total_cpu > cloud_loc.cpu_capacity or total_mem > cloud_loc.memory_capacity:
            raise RuntimeError("All services don't fit in cloud")

        # Place everything at cloud
        placement = {s.id: cloud_loc.id for s in self.problem.services}

        solve_time = time.time() - start_time

        # Compute objective (all internal, latency = 0)
        obj_value = 0.0

        return PlacementSolution(
            problem=self.problem,
            placement=placement,
            objective_value=obj_value,
            solve_time=solve_time,
            solver_status="feasible"
        )
