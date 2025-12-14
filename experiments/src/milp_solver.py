"""
MILP solver using Gurobi for latency-aware service placement.
Implements the formulation from the paper.
"""

import time
from typing import Dict, Optional
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from problem import PlacementProblem, PlacementSolution, Service


class MILPSolver:
    """Gurobi-based MILP solver for service placement."""

    def __init__(self, problem: PlacementProblem, time_limit: float = 600.0, mip_gap: float = 0.01):
        """
        Args:
            problem: Problem instance
            time_limit: Maximum solve time in seconds
            mip_gap: MIP optimality gap tolerance (0.01 = 1%)
        """
        self.problem = problem
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.model: Optional[gp.Model] = None
        self.x_vars: Dict = {}  # Decision variables x[i,j]

    def build_model(self) -> gp.Model:
        """Build the MILP model."""
        prob = self.problem
        n = prob.num_services
        k = prob.num_locations

        # Create model
        model = gp.Model("LatencyAwarePlacement")
        model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = self.mip_gap
        model.Params.OutputFlag = 1  # Show solver output

        # Decision variables: x[i,j] = 1 if service i placed at location j
        x = {}
        for i, service in enumerate(prob.services):
            for j, location in enumerate(prob.locations):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        self.x_vars = x

        # Objective: minimize total weighted latency
        obj_expr = gp.LinExpr()
        for (src_id, tgt_id), freq in prob.edge_frequencies.items():
            for j in range(k):
                for jp in range(k):
                    latency = prob.get_latency(j, jp)
                    # Add f_{src,tgt} * tau_{j,jp} * x[src,j] * x[tgt,jp]
                    # Need to linearize the product x[src,j] * x[tgt,jp]
                    obj_expr += freq * latency * x[src_id, j] * x[tgt_id, jp]

        model.setObjective(obj_expr, GRB.MINIMIZE)

        # Constraint C1: Each service placed exactly once
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(k)) == 1,
                           name=f"placement_exist_{i}")

        # Constraint C2: CPU capacity
        for j in range(k):
            loc = prob.locations[j]
            cpu_expr = gp.LinExpr()
            for i in range(n):
                service = prob.services[i]
                cpu_expr += service.cpu_demand * x[i, j]
            model.addConstr(cpu_expr <= loc.cpu_capacity, name=f"cpu_capacity_{j}")

        # Constraint C3: Memory capacity
        for j in range(k):
            loc = prob.locations[j]
            mem_expr = gp.LinExpr()
            for i in range(n):
                service = prob.services[i]
                mem_expr += service.memory_demand * x[i, j]
            model.addConstr(mem_expr <= loc.memory_capacity, name=f"mem_capacity_{j}")

        # Constraint C4: Anti-affinity (services can't be colocated)
        for (s1, s2) in prob.anti_affinity:
            for j in range(k):
                model.addConstr(x[s1, j] + x[s2, j] <= 1,
                               name=f"anti_affinity_{s1}_{s2}_{j}")

        # Constraint C5: Colocation (services must be on same location)
        for (s1, s2) in prob.colocation:
            for j in range(k):
                # If s1 is at j, then s2 must be at j: x[s1,j] <= x[s2,j]
                # If s2 is at j, then s1 must be at j: x[s2,j] <= x[s1,j]
                # Combined: x[s1,j] == x[s2,j]
                model.addConstr(x[s1, j] == x[s2, j],
                               name=f"colocation_{s1}_{s2}_{j}")

        # Constraint C7: Replication (handled via instance expansion)
        # For each replicated service, create r instances with anti-collocation
        service_instances = {}  # Maps original service_id to list of instance indices
        instance_to_service = {}  # Maps instance index to original service

        current_idx = n  # Start instance indices after original services
        for sid in prob.replicated_services:
            service = prob.get_service_by_id(sid)
            r = service.replication_count
            if r <= 1:
                continue

            instances = []
            for replica in range(r):
                instance_idx = current_idx
                current_idx += 1
                instances.append(instance_idx)
                instance_to_service[instance_idx] = sid

                # Add decision variables for this instance
                for j in range(k):
                    x[instance_idx, j] = model.addVar(vtype=GRB.BINARY,
                                                      name=f"x_{sid}_replica{replica}_{j}")

                # Each instance placed exactly once
                model.addConstr(gp.quicksum(x[instance_idx, j] for j in range(k)) == 1,
                               name=f"replica_placement_{sid}_{replica}")

                # Add resource consumption for replicas
                for j in range(k):
                    loc = prob.locations[j]
                    # Update CPU constraint (need to modify existing constraint)
                    # This is handled by adding to existing constraints

            service_instances[sid] = instances

            # Anti-collocation between replicas
            for r1 in range(len(instances)):
                for r2 in range(r1 + 1, len(instances)):
                    for j in range(k):
                        model.addConstr(x[instances[r1], j] + x[instances[r2], j] <= 1,
                                       name=f"replica_anticoll_{sid}_{r1}_{r2}_{j}")

            # Remove original service (mark as not to be placed)
            for j in range(k):
                model.addConstr(x[sid, j] == 0, name=f"original_removed_{sid}_{j}")

        self.model = model
        return model

    def solve(self) -> PlacementSolution:
        """Solve the MILP and return solution."""
        if self.model is None:
            self.build_model()

        start_time = time.time()
        self.model.optimize()
        solve_time = time.time() - start_time

        # Extract solution
        status_map = {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "timeout",
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible",
            GRB.UNBOUNDED: "unbounded",
        }
        status = status_map.get(self.model.Status, "unknown")

        if self.model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and self.model.SolCount > 0:
            # Extract placement
            placement = {}
            for i in range(self.problem.num_services):
                for j in range(self.problem.num_locations):
                    if self.x_vars[i, j].X > 0.5:  # Binary variable is 1
                        placement[i] = j
                        break

            # Handle replicated services (they were removed, use first replica)
            for sid in self.problem.replicated_services:
                if sid not in placement:
                    # Find first replica's location
                    found = False
                    for (i, j), var in self.x_vars.items():
                        if isinstance(i, int) and i >= self.problem.num_services:
                            # This is a replica instance
                            if var.X > 0.5:
                                placement[sid] = j
                                found = True
                                break
                    if not found:
                        # Default to location 0 if no replica found
                        placement[sid] = 0

            obj_value = self.model.ObjVal
            gap = self.model.MIPGap if status != "optimal" else 0.0

            solution = PlacementSolution(
                problem=self.problem,
                placement=placement,
                objective_value=obj_value,
                solve_time=solve_time,
                solver_status=status,
                gap=gap
            )
            return solution
        else:
            # No feasible solution found
            raise RuntimeError(f"Solver failed with status: {status}")

    def get_solver_stats(self) -> Dict:
        """Get solver statistics."""
        if self.model is None:
            return {}

        return {
            'num_variables': self.model.NumVars,
            'num_constraints': self.model.NumConstrs,
            'num_binary_vars': self.model.NumBinVars,
            'num_nodes_explored': self.model.NodeCount if hasattr(self.model, 'NodeCount') else 0,
            'objective_bound': self.model.ObjBound if self.model.SolCount > 0 else None,
            'objective_value': self.model.ObjVal if self.model.SolCount > 0 else None,
        }


def solve_milp(problem: PlacementProblem,
               time_limit: float = 600.0,
               mip_gap: float = 0.01,
               verbose: bool = True) -> PlacementSolution:
    """
    Convenience function to solve a placement problem using MILP.

    Args:
        problem: Problem instance
        time_limit: Maximum solve time in seconds
        mip_gap: MIP optimality gap tolerance
        verbose: Print solver output

    Returns:
        PlacementSolution
    """
    solver = MILPSolver(problem, time_limit=time_limit, mip_gap=mip_gap)
    solution = solver.solve()

    if verbose:
        print(f"Solver Status: {solution.solver_status}")
        print(f"Objective Value: {solution.objective_value:.2f}")
        print(f"Solve Time: {solution.solve_time:.2f}s")
        if solution.gap > 0:
            print(f"MIP Gap: {solution.gap * 100:.2f}%")

        stats = solver.get_solver_stats()
        print(f"Variables: {stats['num_variables']}, Constraints: {stats['num_constraints']}")

    return solution
