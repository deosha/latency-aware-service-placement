"""
Benchmark instance generators for service placement experiments.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Set
import random

from problem import PlacementProblem, Service, Location


class BenchmarkGenerator:
    """Generates synthetic problem instances."""

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_dag(self, num_services: int, edge_prob: float = 0.3,
                     dag_type: str = "random") -> Tuple[nx.DiGraph, dict]:
        """
        Generate service DAG.

        Args:
            num_services: Number of services
            edge_prob: Probability of edge (for random DAG)
            dag_type: "chain", "tree", "random", "diamond"

        Returns:
            (DAG, edge_frequencies)
        """
        if dag_type == "chain":
            # Linear chain: s0 -> s1 -> s2 -> ... -> sn
            G = nx.DiGraph()
            G.add_nodes_from(range(num_services))
            for i in range(num_services - 1):
                G.add_edge(i, i + 1)

        elif dag_type == "tree":
            # Binary tree
            G = nx.DiGraph()
            G.add_nodes_from(range(num_services))
            for i in range(num_services):
                left_child = 2 * i + 1
                right_child = 2 * i + 2
                if left_child < num_services:
                    G.add_edge(i, left_child)
                if right_child < num_services:
                    G.add_edge(i, right_child)

        elif dag_type == "diamond":
            # Diamond pattern: source -> multiple middle layers -> sink
            G = nx.DiGraph()
            G.add_nodes_from(range(num_services))

            if num_services >= 3:
                # Source node
                G.add_edge(0, 1)
                # Middle nodes
                for i in range(1, num_services - 1):
                    G.add_edge(i, num_services - 1)
                # Additional cross edges
                for i in range(1, min(num_services - 1, 3)):
                    for j in range(i + 1, min(num_services - 1, 5)):
                        if np.random.rand() < 0.3:
                            G.add_edge(i, j)

        else:  # random
            # Random DAG using erdos_renyi + topological sort
            while True:
                G_undirected = nx.erdos_renyi_graph(num_services, edge_prob, seed=self.seed)
                G = nx.DiGraph()
                G.add_nodes_from(range(num_services))

                # Orient edges according to topological order
                topo_order = list(range(num_services))
                np.random.shuffle(topo_order)
                order_map = {node: i for i, node in enumerate(topo_order)}

                for u, v in G_undirected.edges():
                    if order_map[u] < order_map[v]:
                        G.add_edge(u, v)
                    else:
                        G.add_edge(v, u)

                # Ensure DAG is connected and non-trivial
                if nx.is_directed_acyclic_graph(G) and G.number_of_edges() > 0:
                    break

        # Generate edge frequencies (requests/sec)
        edge_frequencies = {}
        for (u, v) in G.edges():
            # Higher frequency for edges near the source
            base_freq = np.random.uniform(5, 50)
            edge_frequencies[(u, v)] = base_freq

        return G, edge_frequencies

    def generate_continuum_topology(self, num_locations: int,
                                    topology_type: str = "hierarchical") -> np.ndarray:
        """
        Generate latency matrix for fog-edge-cloud continuum.

        Args:
            num_locations: Number of locations
            topology_type: "hierarchical", "mesh", "hybrid"

        Returns:
            Latency matrix (symmetric)
        """
        latency = np.zeros((num_locations, num_locations))

        if topology_type == "hierarchical":
            # 5-tier hierarchy: edge -> fog -> regional -> metro -> cloud
            # Latencies increase with distance in hierarchy
            tier_latencies = {
                0: 0,    # self
                1: 2,    # adjacent tier
                2: 20,   # 2 tiers apart
                3: 60,   # 3 tiers apart
                4: 180,  # 4 tiers apart
            }

            for i in range(num_locations):
                for j in range(i + 1, num_locations):
                    tier_dist = abs(i - j)
                    base_latency = tier_latencies.get(min(tier_dist, 4), 200)
                    # Add some noise
                    noise = np.random.uniform(0.9, 1.1)
                    latency[i, j] = latency[j, i] = base_latency * noise

        elif topology_type == "mesh":
            # Full mesh with triangle inequality
            # Use metric embedding in 2D space
            coords = np.random.rand(num_locations, 2) * 100  # Random positions
            for i in range(num_locations):
                for j in range(i + 1, num_locations):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    # Map distance to latency (1-10 ms per unit distance)
                    latency[i, j] = latency[j, i] = dist * np.random.uniform(0.5, 2.0)

        else:  # hybrid
            # Combination: local clusters with high inter-cluster latency
            cluster_size = max(2, num_locations // 3)
            for i in range(num_locations):
                for j in range(i + 1, num_locations):
                    if i // cluster_size == j // cluster_size:
                        # Same cluster: low latency
                        latency[i, j] = latency[j, i] = np.random.uniform(1, 10)
                    else:
                        # Different clusters: high latency
                        latency[i, j] = latency[j, i] = np.random.uniform(50, 150)

        return latency

    def generate_services(self, num_services: int, resource_dist: str = "uniform") -> List[Service]:
        """
        Generate service instances with resource demands.

        Args:
            num_services: Number of services
            resource_dist: "uniform", "skewed", "realistic"

        Returns:
            List of services
        """
        services = []

        for i in range(num_services):
            if resource_dist == "uniform":
                cpu = np.random.uniform(0.5, 4.0)
                memory = np.random.uniform(256, 2048)

            elif resource_dist == "skewed":
                # Heavy-tailed: few services need lots of resources
                if np.random.rand() < 0.2:  # 20% are resource-intensive
                    cpu = np.random.uniform(4.0, 8.0)
                    memory = np.random.uniform(2048, 8192)
                else:
                    cpu = np.random.uniform(0.25, 1.0)
                    memory = np.random.uniform(128, 512)

            else:  # realistic (based on microservice patterns)
                service_types = ["frontend", "api", "compute", "database", "cache"]
                stype = random.choice(service_types)

                if stype == "frontend":
                    cpu, memory = np.random.uniform(0.5, 1.0), np.random.uniform(256, 512)
                elif stype == "api":
                    cpu, memory = np.random.uniform(1.0, 2.0), np.random.uniform(512, 1024)
                elif stype == "compute":
                    cpu, memory = np.random.uniform(2.0, 8.0), np.random.uniform(1024, 4096)
                elif stype == "database":
                    cpu, memory = np.random.uniform(2.0, 4.0), np.random.uniform(2048, 8192)
                else:  # cache
                    cpu, memory = np.random.uniform(0.5, 1.5), np.random.uniform(512, 2048)

            service = Service(
                id=i,
                name=f"service_{i}",
                cpu_demand=cpu,
                memory_demand=memory,
                replication_count=1
            )
            services.append(service)

        return services

    def generate_locations(self, num_locations: int, capacity_mult: float = 2.0,
                          total_cpu_demand: float = None, total_mem_demand: float = None) -> List[Location]:
        """
        Generate locations with heterogeneous capacities.

        Args:
            num_locations: Number of locations
            capacity_mult: Multiplier for total capacity vs. total demand
            total_cpu_demand: Total CPU demand from all services
            total_mem_demand: Total memory demand from all services

        Returns:
            List of locations
        """
        locations = []

        # 5-tier model: edge (smallest) to cloud (largest)
        for i in range(num_locations):
            tier = min(4, i % 5)  # Assign tier cyclically

            if tier == 0:  # Edge
                cpu_cap = np.random.uniform(4, 12)
                mem_cap = np.random.uniform(2048, 8192)
                name = f"edge_{i}"
            elif tier == 1:  # Fog
                cpu_cap = np.random.uniform(12, 24)
                mem_cap = np.random.uniform(8192, 24576)
                name = f"fog_{i}"
            elif tier == 2:  # Regional
                cpu_cap = np.random.uniform(24, 48)
                mem_cap = np.random.uniform(24576, 49152)
                name = f"regional_{i}"
            elif tier == 3:  # Metro
                cpu_cap = np.random.uniform(48, 96)
                mem_cap = np.random.uniform(49152, 98304)
                name = f"metro_{i}"
            else:  # Cloud
                cpu_cap = np.random.uniform(96, 192)
                mem_cap = np.random.uniform(98304, 196608)
                name = f"cloud_{i}"

            location = Location(
                id=i,
                name=name,
                cpu_capacity=cpu_cap,
                memory_capacity=mem_cap,
                tier=tier
            )
            locations.append(location)

        # Scale capacities to ensure feasibility
        if total_cpu_demand is not None and total_mem_demand is not None:
            total_cpu_cap = sum(loc.cpu_capacity for loc in locations)
            total_mem_cap = sum(loc.memory_capacity for loc in locations)

            cpu_scale = (total_cpu_demand * capacity_mult) / total_cpu_cap
            mem_scale = (total_mem_demand * capacity_mult) / total_mem_cap

            # Scale up if needed
            if cpu_scale > 1.0 or mem_scale > 1.0:
                scale = max(cpu_scale, mem_scale)
                for loc in locations:
                    loc.cpu_capacity *= scale
                    loc.memory_capacity *= scale

        return locations

    def generate_problem(self,
                        num_services: int,
                        num_locations: int,
                        dag_type: str = "random",
                        topology_type: str = "hierarchical",
                        resource_dist: str = "uniform",
                        anti_affinity_prob: float = 0.1,
                        colocation_prob: float = 0.05,
                        replication_prob: float = 0.1) -> PlacementProblem:
        """
        Generate a complete problem instance.

        Args:
            num_services: Number of services
            num_locations: Number of locations
            dag_type: Service DAG structure
            topology_type: Network topology
            resource_dist: Resource demand distribution
            anti_affinity_prob: Probability of anti-affinity constraint
            colocation_prob: Probability of colocation constraint
            replication_prob: Probability a service needs replication

        Returns:
            PlacementProblem instance
        """
        services = self.generate_services(num_services, resource_dist)

        # Calculate total demands
        total_cpu = sum(s.cpu_demand for s in services)
        total_mem = sum(s.memory_demand for s in services)

        locations = self.generate_locations(num_locations, capacity_mult=2.5,
                                            total_cpu_demand=total_cpu,
                                            total_mem_demand=total_mem)
        dag, edge_freq = self.generate_dag(num_services, dag_type=dag_type)
        latency_matrix = self.generate_continuum_topology(num_locations, topology_type)

        # Generate DevOps constraints
        anti_affinity = set()
        colocation = set()
        replicated_services = set()

        # Anti-affinity: some service pairs can't be colocated
        for i in range(num_services):
            for j in range(i + 1, num_services):
                if np.random.rand() < anti_affinity_prob:
                    anti_affinity.add((i, j))

        # Colocation: some service pairs must be colocated (rare)
        for i in range(num_services):
            for j in range(i + 1, num_services):
                if (i, j) not in anti_affinity and np.random.rand() < colocation_prob:
                    colocation.add((i, j))

        # Replication: some services need multiple replicas
        for i in range(num_services):
            if np.random.rand() < replication_prob:
                replicated_services.add(i)
                services[i].replication_count = np.random.randint(2, 4)

        problem = PlacementProblem(
            services=services,
            locations=locations,
            service_dag=dag,
            edge_frequencies=edge_freq,
            latency_matrix=latency_matrix,
            anti_affinity=anti_affinity,
            colocation=colocation,
            replicated_services=replicated_services
        )

        return problem


def create_smart_factory_benchmark() -> PlacementProblem:
    """Create the smart factory benchmark from the paper (Section 4.1)."""

    # 5 services
    services = [
        Service(id=0, name="ImageIngestion", cpu_demand=2.0, memory_demand=512),
        Service(id=1, name="ObjectDetection", cpu_demand=4.0, memory_demand=2048),
        Service(id=2, name="DefectClassification", cpu_demand=3.0, memory_demand=1536),
        Service(id=3, name="AlertManager", cpu_demand=0.5, memory_demand=256),
        Service(id=4, name="DataArchival", cpu_demand=1.0, memory_demand=512),
    ]

    # 5 locations (Table 2 from paper)
    locations = [
        Location(id=0, name="EdgeGateway", cpu_capacity=4.0, memory_capacity=2048, tier=0),
        Location(id=1, name="FactoryFog", cpu_capacity=8.0, memory_capacity=8192, tier=1),
        Location(id=2, name="RegionalFogDC", cpu_capacity=16.0, memory_capacity=16384, tier=2),
        Location(id=3, name="MetroCloud", cpu_capacity=32.0, memory_capacity=32768, tier=3),
        Location(id=4, name="ContinentalCloud", cpu_capacity=64.0, memory_capacity=65536, tier=4),
    ]

    # DAG: Linear chain
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1, 2, 3, 4])
    dag.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

    # Edge frequencies (all 10 req/s from paper)
    edge_frequencies = {
        (0, 1): 10.0,
        (1, 2): 10.0,
        (2, 3): 10.0,
        (3, 4): 10.0,
    }

    # Latency matrix (Table 2 from paper, in milliseconds)
    latency_matrix = np.array([
        [0,   2,   22,  82,  202],
        [2,   0,   20,  80,  200],
        [22,  20,  0,   60,  180],
        [82,  80,  60,  0,   120],
        [202, 200, 180, 120, 0],
    ], dtype=float)

    # DevOps constraints from paper
    anti_affinity = {(1, 2)}  # ObjectDetection and DefectClassification can't be colocated
    colocation = set()  # None specified
    replicated_services = set()  # None in base scenario
    dependency_order = {}

    problem = PlacementProblem(
        services=services,
        locations=locations,
        service_dag=dag,
        edge_frequencies=edge_frequencies,
        latency_matrix=latency_matrix,
        anti_affinity=anti_affinity,
        colocation=colocation,
        replicated_services=replicated_services,
        dependency_order=dependency_order
    )

    return problem
