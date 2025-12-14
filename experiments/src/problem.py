"""
Data structures for the Latency-Aware Service Placement problem.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
import networkx as nx
import numpy as np


@dataclass
class Service:
    """Represents a microservice in the DAG."""
    id: int
    name: str
    cpu_demand: float  # CPU cores
    memory_demand: float  # MB
    replication_count: int = 1  # Number of replicas required

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Service) and self.id == other.id


@dataclass
class Location:
    """Represents a location in the fog-edge-cloud continuum."""
    id: int
    name: str
    cpu_capacity: float  # CPU cores
    memory_capacity: float  # MB
    tier: int  # 1=edge, 2=fog, 3=regional, 4=metro, 5=cloud

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Location) and self.id == other.id


@dataclass
class ServiceEdge:
    """Represents a dependency edge between services."""
    source: Service
    target: Service
    frequency: float  # requests per second
    data_size: float = 0.0  # MB per request (optional)

    def __hash__(self):
        return hash((self.source.id, self.target.id))


@dataclass
class PlacementProblem:
    """Complete problem instance."""
    services: List[Service]
    locations: List[Location]
    service_dag: nx.DiGraph  # DAG of service dependencies
    edge_frequencies: Dict[Tuple[int, int], float]  # (src_id, tgt_id) -> frequency
    latency_matrix: np.ndarray  # latency[i][j] = latency from location i to j (ms)

    # Optional DevOps constraints
    anti_affinity: Set[Tuple[int, int]] = field(default_factory=set)  # Service pairs that can't colocate
    colocation: Set[Tuple[int, int]] = field(default_factory=set)  # Service pairs that must colocate
    replicated_services: Set[int] = field(default_factory=set)  # Service IDs requiring replication
    dependency_order: Dict[int, Set[int]] = field(default_factory=dict)  # Deployment dependencies

    def __post_init__(self):
        """Validate problem instance."""
        assert len(self.services) > 0, "Must have at least one service"
        assert len(self.locations) > 0, "Must have at least one location"
        assert self.latency_matrix.shape == (len(self.locations), len(self.locations))
        assert nx.is_directed_acyclic_graph(self.service_dag), "Service graph must be a DAG"

    @property
    def num_services(self) -> int:
        return len(self.services)

    @property
    def num_locations(self) -> int:
        return len(self.locations)

    def get_service_by_id(self, sid: int) -> Service:
        """Get service by ID."""
        for s in self.services:
            if s.id == sid:
                return s
        raise ValueError(f"Service {sid} not found")

    def get_location_by_id(self, lid: int) -> Location:
        """Get location by ID."""
        for loc in self.locations:
            if loc.id == lid:
                return loc
        raise ValueError(f"Location {lid} not found")

    def get_edge_frequency(self, src_id: int, tgt_id: int) -> float:
        """Get frequency for edge between services."""
        return self.edge_frequencies.get((src_id, tgt_id), 0.0)

    def get_latency(self, loc_i: int, loc_j: int) -> float:
        """Get network latency between two locations."""
        return self.latency_matrix[loc_i, loc_j]


@dataclass
class PlacementSolution:
    """Solution to the placement problem."""
    problem: PlacementProblem
    placement: Dict[int, int]  # service_id -> location_id
    objective_value: float  # Total weighted latency
    solve_time: float  # Seconds
    solver_status: str  # "optimal", "feasible", "infeasible", "timeout"
    gap: float = 0.0  # MIP gap (for non-optimal solutions)

    def __post_init__(self):
        """Validate solution."""
        assert len(self.placement) == self.problem.num_services, \
            f"Placement must assign all {self.problem.num_services} services"

    def get_service_location(self, service_id: int) -> int:
        """Get location assignment for a service."""
        return self.placement[service_id]

    def compute_total_latency(self) -> float:
        """Compute total weighted latency from placement."""
        total = 0.0
        for (src_id, tgt_id), freq in self.problem.edge_frequencies.items():
            src_loc = self.placement[src_id]
            tgt_loc = self.placement[tgt_id]
            latency = self.problem.get_latency(src_loc, tgt_loc)
            total += freq * latency
        return total

    def compute_critical_path_latency(self) -> float:
        """Compute critical path latency (max over all root-to-leaf paths)."""
        dag = self.problem.service_dag

        # Find all root-to-leaf paths
        roots = [n for n in dag.nodes() if dag.in_degree(n) == 0]
        leaves = [n for n in dag.nodes() if dag.out_degree(n) == 0]

        max_latency = 0.0
        for root in roots:
            for leaf in leaves:
                try:
                    paths = list(nx.all_simple_paths(dag, root, leaf))
                    for path in paths:
                        path_latency = 0.0
                        for i in range(len(path) - 1):
                            src_id = path[i]
                            tgt_id = path[i + 1]
                            src_loc = self.placement[src_id]
                            tgt_loc = self.placement[tgt_id]
                            path_latency += self.problem.get_latency(src_loc, tgt_loc)
                        max_latency = max(max_latency, path_latency)
                except nx.NetworkXNoPath:
                    continue

        return max_latency

    def get_resource_utilization(self) -> Dict[int, Dict[str, float]]:
        """Compute resource utilization at each location."""
        utilization = {}
        for loc in self.problem.locations:
            utilization[loc.id] = {
                'cpu_used': 0.0,
                'memory_used': 0.0,
                'cpu_capacity': loc.cpu_capacity,
                'memory_capacity': loc.memory_capacity
            }

        for service in self.problem.services:
            loc_id = self.placement[service.id]
            utilization[loc_id]['cpu_used'] += service.cpu_demand
            utilization[loc_id]['memory_used'] += service.memory_demand

        # Add utilization percentages
        for loc_id in utilization:
            u = utilization[loc_id]
            u['cpu_util_pct'] = 100 * u['cpu_used'] / u['cpu_capacity'] if u['cpu_capacity'] > 0 else 0
            u['memory_util_pct'] = 100 * u['memory_used'] / u['memory_capacity'] if u['memory_capacity'] > 0 else 0

        return utilization

    def validate_constraints(self) -> Dict[str, bool]:
        """Check if solution satisfies all constraints."""
        violations = {
            'resource_cpu': [],
            'resource_memory': [],
            'anti_affinity': [],
            'colocation': [],
        }

        # Check resource constraints
        util = self.get_resource_utilization()
        for loc_id, u in util.items():
            if u['cpu_used'] > u['cpu_capacity'] + 1e-6:
                violations['resource_cpu'].append(
                    f"Location {loc_id}: {u['cpu_used']:.2f} > {u['cpu_capacity']:.2f}"
                )
            if u['memory_used'] > u['memory_capacity'] + 1e-6:
                violations['resource_memory'].append(
                    f"Location {loc_id}: {u['memory_used']:.2f} > {u['memory_capacity']:.2f}"
                )

        # Check anti-affinity
        for (s1, s2) in self.problem.anti_affinity:
            if self.placement.get(s1) == self.placement.get(s2):
                violations['anti_affinity'].append(f"Services {s1} and {s2} on same location")

        # Check colocation
        for (s1, s2) in self.problem.colocation:
            if self.placement.get(s1) != self.placement.get(s2):
                violations['colocation'].append(f"Services {s1} and {s2} on different locations")

        return {
            'valid': all(len(v) == 0 for v in violations.values()),
            'violations': violations
        }
