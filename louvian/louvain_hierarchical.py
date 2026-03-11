#!/usr/bin/env python3
"""
Custom Louvain Algorithm Implementation for Hierarchical Graph Clustering

This module implements the Louvain community detection algorithm from scratch,
with support for:
- Configurable number of iterations/passes
- Hierarchical coarsening for LLM-friendly graph reduction
- Expansion back to original nodes

The Louvain algorithm works in two phases:
1. Local moving: Each node is moved to the community that maximizes modularity gain
2. Aggregation: Communities become super-nodes, creating a coarser graph

These phases repeat until no improvement is possible.

Reference: Blondel et al. "Fast unfolding of communities in large networks" (2008)
"""

import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import json


class LouvainGraph:
    """
    A graph representation optimized for the Louvain algorithm.
    Supports both directed and undirected graphs.
    """
    
    def __init__(self):
        # Adjacency list with weights: node -> {neighbor: weight}
        self.adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Total weight of edges incident to each node (degree for unweighted)
        self.node_weights: Dict[str, float] = defaultdict(float)
        # Self-loop weights
        self.self_loops: Dict[str, float] = defaultdict(float)
        # Total weight of all edges in the graph (sum of all edge weights)
        self.total_weight: float = 0.0
        # Set of all nodes
        self.nodes: Set[str] = set()
    
    def add_edge(self, u: str, v: str, weight: float = 1.0):
        """Add an edge (or increase weight if exists)"""
        self.nodes.add(u)
        self.nodes.add(v)
        
        if u == v:
            # Self-loop
            self.self_loops[u] += weight
            self.node_weights[u] += weight  # Self-loop contributes once to degree
            self.total_weight += weight
        else:
            # Regular edge (store both directions for undirected)
            self.adj[u][v] += weight
            self.adj[v][u] += weight
            self.node_weights[u] += weight
            self.node_weights[v] += weight
            self.total_weight += weight  # Count once (undirected)
    
    def get_neighbors(self, node: str) -> Dict[str, float]:
        """Get neighbors and edge weights for a node"""
        return dict(self.adj[node])
    
    def get_edge_weight(self, u: str, v: str) -> float:
        """Get weight of edge between u and v"""
        if u == v:
            return self.self_loops.get(u, 0.0)
        return self.adj[u].get(v, 0.0)
    
    def number_of_nodes(self) -> int:
        return len(self.nodes)
    
    def number_of_edges(self) -> int:
        # Count unique edges (adj stores both directions)
        count = sum(len(neighbors) for neighbors in self.adj.values()) // 2
        count += len(self.self_loops)
        return count
    
    @classmethod
    def from_networkx(cls, nx_graph, directed: bool = True) -> 'LouvainGraph':
        """Create LouvainGraph from a NetworkX graph"""
        lg = cls()
        
        # For directed graphs, we treat it as undirected for community detection
        # but preserve the original edges
        for u, v in nx_graph.edges():
            weight = nx_graph[u][v].get('weight', 1.0)
            lg.add_edge(str(u), str(v), weight)
        
        # Make sure isolated nodes are included
        for node in nx_graph.nodes():
            lg.nodes.add(str(node))
            if str(node) not in lg.node_weights:
                lg.node_weights[str(node)] = 0.0
        
        return lg


class Community:
    """Represents a community (cluster) of nodes"""
    
    def __init__(self):
        # Nodes in this community
        self.nodes: Set[str] = set()
        # Sum of weights of nodes in this community
        self.total_weight: float = 0.0
        # Sum of internal edge weights (edges within community)
        self.internal_weight: float = 0.0
    
    def add_node(self, node: str, node_weight: float, internal_edges: float = 0.0):
        """Add a node to the community"""
        self.nodes.add(node)
        self.total_weight += node_weight
        self.internal_weight += internal_edges
    
    def remove_node(self, node: str, node_weight: float, internal_edges: float = 0.0):
        """Remove a node from the community"""
        self.nodes.discard(node)
        self.total_weight -= node_weight
        self.internal_weight -= internal_edges


class LouvainAlgorithm:
    """
    Implementation of the Louvain community detection algorithm.
    
    Parameters:
    - resolution: Resolution parameter (gamma). Higher values lead to more communities.
    - min_modularity_gain: Minimum improvement to continue iterating.
    - max_passes: Maximum number of complete passes through all nodes.
    - random_seed: Seed for reproducibility.
    """
    
    def __init__(self, resolution: float = 1.0, min_modularity_gain: float = 1e-7,
                 max_passes: int = 10, random_seed: Optional[int] = None):
        self.resolution = resolution
        self.min_modularity_gain = min_modularity_gain
        self.max_passes = max_passes
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def _modularity_gain(self, graph: LouvainGraph, node: str, 
                         community: Community, node_to_comm_weight: float) -> float:
        """
        Calculate the modularity gain from moving a node into a community.
        
        Modularity gain formula:
        ΔQ = [k_i,in / m] - [Σ_tot * k_i / (2m²)] * resolution
        
        Where:
        - k_i,in = sum of weights from node i to nodes in community
        - m = total weight of graph
        - Σ_tot = sum of weights of nodes in community
        - k_i = weight (degree) of node i
        """
        m = graph.total_weight
        if m == 0:
            return 0.0
        
        k_i = graph.node_weights[node]
        sigma_tot = community.total_weight
        
        # Weight from node to community
        k_i_in = node_to_comm_weight
        
        # Modularity gain
        gain = k_i_in / m - (sigma_tot * k_i / (2 * m * m)) * self.resolution
        
        return gain
    
    def _get_node_to_community_weight(self, graph: LouvainGraph, node: str,
                                       community: Community) -> float:
        """Calculate sum of edge weights from node to all nodes in community"""
        total = 0.0
        neighbors = graph.get_neighbors(node)
        
        for neighbor, weight in neighbors.items():
            if neighbor in community.nodes and neighbor != node:
                total += weight
        
        return total
    
    def _one_level(self, graph: LouvainGraph) -> Tuple[Dict[str, int], bool]:
        """
        Perform one level of the Louvain algorithm (local moving phase).
        
        Returns:
        - node_to_community: mapping of nodes to community IDs
        - improved: whether any improvement was made
        """
        # Initialize: each node in its own community
        node_to_community: Dict[str, int] = {}
        communities: Dict[int, Community] = {}
        
        for i, node in enumerate(graph.nodes):
            node_to_community[node] = i
            comm = Community()
            comm.add_node(node, graph.node_weights[node], graph.self_loops.get(node, 0.0))
            communities[i] = comm
        
        improved = True
        total_improvement = False
        pass_count = 0
        
        while improved and pass_count < self.max_passes:
            improved = False
            pass_count += 1
            
            # Randomize node order for better results
            nodes = list(graph.nodes)
            random.shuffle(nodes)
            
            for node in nodes:
                current_comm_id = node_to_community[node]
                current_comm = communities[current_comm_id]
                
                # Weight from node to its current community (excluding self)
                current_weight = self._get_node_to_community_weight(graph, node, current_comm)
                
                # Remove node from current community temporarily
                node_weight = graph.node_weights[node]
                self_loop = graph.self_loops.get(node, 0.0)
                current_comm.remove_node(node, node_weight, self_loop + current_weight)
                
                # Find the best community to move to
                best_comm_id = current_comm_id
                best_gain = 0.0
                
                # Check neighboring communities
                neighbor_comms: Set[int] = set()
                for neighbor in graph.get_neighbors(node):
                    neighbor_comms.add(node_to_community[neighbor])
                
                # Always consider staying in current community
                neighbor_comms.add(current_comm_id)
                
                for comm_id in neighbor_comms:
                    comm = communities[comm_id]
                    weight_to_comm = self._get_node_to_community_weight(graph, node, comm)
                    
                    gain = self._modularity_gain(graph, node, comm, weight_to_comm)
                    
                    if gain > best_gain + self.min_modularity_gain:
                        best_gain = gain
                        best_comm_id = comm_id
                
                # Move node to best community
                best_comm = communities[best_comm_id]
                best_weight = self._get_node_to_community_weight(graph, node, best_comm)
                best_comm.add_node(node, node_weight, self_loop + best_weight)
                node_to_community[node] = best_comm_id
                
                if best_comm_id != current_comm_id:
                    improved = True
                    total_improvement = True
        
        # Renumber communities to be contiguous
        unique_comms = set(node_to_community.values())
        comm_mapping = {old: new for new, old in enumerate(sorted(unique_comms))}
        node_to_community = {node: comm_mapping[comm] for node, comm in node_to_community.items()}
        
        return node_to_community, total_improvement
    
    def _aggregate(self, graph: LouvainGraph, 
                   node_to_community: Dict[str, int]) -> LouvainGraph:
        """
        Create a new coarser graph where each community becomes a super-node.
        """
        new_graph = LouvainGraph()
        
        # Map community IDs to string names for the new graph
        comm_names = {comm: f"comm_{comm}" for comm in set(node_to_community.values())}
        
        # Add edges between communities
        for node in graph.nodes:
            node_comm = node_to_community[node]
            node_comm_name = comm_names[node_comm]
            new_graph.nodes.add(node_comm_name)
            
            # Self-loops
            if node in graph.self_loops:
                new_graph.add_edge(node_comm_name, node_comm_name, graph.self_loops[node])
            
            # Regular edges
            for neighbor, weight in graph.get_neighbors(node).items():
                neighbor_comm = node_to_community[neighbor]
                neighbor_comm_name = comm_names[neighbor_comm]
                
                # Only add each edge once (check node < neighbor to avoid duplicates)
                if node <= neighbor:
                    new_graph.add_edge(node_comm_name, neighbor_comm_name, weight)
        
        return new_graph
    
    def run(self, graph: LouvainGraph, num_levels: int = None) -> List[Dict[str, int]]:
        """
        Run the full Louvain algorithm.
        
        Parameters:
        - graph: The input graph
        - num_levels: Number of hierarchical levels (None = run until convergence)
        
        Returns:
        - List of partitions, one per level. Each partition is a dict: node -> community_id
        """
        partitions = []
        current_graph = graph
        
        # Track mapping from original nodes through aggregation
        # Initially, each node maps to itself
        node_mapping: Dict[str, Set[str]] = {node: {node} for node in graph.nodes}
        
        level = 0
        while True:
            if num_levels is not None and level >= num_levels:
                break
            
            # Run one level
            partition, improved = self._one_level(current_graph)
            
            if not improved and level > 0:
                break
            
            # Store the partition (mapped back to original nodes)
            original_partition: Dict[str, int] = {}
            for current_node, comm_id in partition.items():
                # current_node might be a super-node, so expand it
                for original_node in node_mapping.get(current_node, {current_node}):
                    original_partition[original_node] = comm_id
            
            partitions.append(original_partition)
            
            # Check if we should stop
            num_communities = len(set(partition.values()))
            if num_communities == current_graph.number_of_nodes():
                # No improvement possible
                break
            
            if num_communities == 1:
                # Everything in one community
                break
            
            # Aggregate to create coarser graph
            new_graph = self._aggregate(current_graph, partition)
            
            # Update node mapping for next level
            new_node_mapping: Dict[str, Set[str]] = defaultdict(set)
            for current_node, comm_id in partition.items():
                new_comm_name = f"comm_{comm_id}"
                for original_node in node_mapping.get(current_node, {current_node}):
                    new_node_mapping[new_comm_name].add(original_node)
            
            node_mapping = dict(new_node_mapping)
            current_graph = new_graph
            level += 1
            
            print(f"   Louvain level {level}: {num_communities} communities")
        
        return partitions
    
    def get_communities(self, partition: Dict[str, int]) -> Dict[str, List[str]]:
        """Convert partition dict to community dict: community_id -> [nodes]"""
        communities: Dict[str, List[str]] = defaultdict(list)
        
        for node, comm_id in partition.items():
            communities[f"cluster_{comm_id}"].append(node)
        
        return dict(communities)


class HierarchicalLouvain:
    """
    Hierarchical Louvain for LLM-friendly graph coarsening.
    
    This class provides:
    1. Multi-level coarsening to reduce graph size
    2. Super-node graph for LLM clustering
    3. Expansion of LLM clustering back to original nodes
    """
    
    def __init__(self, graph, num_coarsening_levels: int = 2, 
                 resolution: float = 1.0, random_seed: int = 42):
        """
        Initialize hierarchical Louvain.
        
        Parameters:
        - graph: NetworkX graph (directed or undirected)
        - num_coarsening_levels: How many times to coarsen (more = smaller graph)
        - resolution: Louvain resolution parameter (higher = more clusters)
        - random_seed: For reproducibility
        """
        self.original_graph = graph
        self.num_levels = num_coarsening_levels
        self.resolution = resolution
        self.random_seed = random_seed
        
        # Convert to LouvainGraph
        self.louvain_graph = LouvainGraph.from_networkx(graph)
        
        # Will be populated after coarsening
        self.partitions: List[Dict[str, int]] = []
        self.super_node_to_original: Dict[str, List[str]] = {}
        self.coarse_graph: Optional[LouvainGraph] = None
        self.coarse_edges: List[Tuple[str, str, float]] = []
    
    def coarsen(self) -> Tuple[Dict[str, List[str]], List[Dict]]:
        """
        Run Louvain coarsening to create super-nodes.
        
        Returns:
        - super_node_mapping: super_node_name -> [original_nodes]
        - super_node_edges: list of {source, target, weight} for the coarse graph
        """
        print(f"\n📊 Running Louvain coarsening ({self.num_levels} levels)...")
        print(f"   Original graph: {self.louvain_graph.number_of_nodes()} nodes, "
              f"{self.louvain_graph.number_of_edges()} edges")
        
        # Run Louvain
        louvain = LouvainAlgorithm(
            resolution=self.resolution,
            max_passes=10,
            random_seed=self.random_seed
        )
        
        self.partitions = louvain.run(self.louvain_graph, num_levels=self.num_levels)
        
        if not self.partitions:
            print("   ⚠ Louvain produced no partitions, using original graph")
            # Each node is its own super-node
            self.super_node_to_original = {node: [node] for node in self.louvain_graph.nodes}
            return self.super_node_to_original, []
        
        # Use the last (most coarse) partition
        final_partition = self.partitions[-1]
        
        # Build super-node mapping
        self.super_node_to_original = defaultdict(list)
        for node, comm_id in final_partition.items():
            super_node = f"super_{comm_id}"
            self.super_node_to_original[super_node].append(node)
        
        self.super_node_to_original = dict(self.super_node_to_original)
        
        # Build coarse graph edges
        self._build_coarse_graph(final_partition)
        
        print(f"   Coarsened graph: {len(self.super_node_to_original)} super-nodes")
        
        # Print super-node sizes
        sizes = [len(nodes) for nodes in self.super_node_to_original.values()]
        print(f"   Super-node sizes: min={min(sizes)}, max={max(sizes)}, "
              f"avg={sum(sizes)/len(sizes):.1f}")
        
        return self.super_node_to_original, self.coarse_edges
    
    def _build_coarse_graph(self, partition: Dict[str, int]):
        """Build the coarse graph with edges between super-nodes"""
        # Map original nodes to super-nodes
        node_to_super = {}
        for super_node, original_nodes in self.super_node_to_original.items():
            for node in original_nodes:
                node_to_super[node] = super_node
        
        # Count edges between super-nodes
        edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        
        for u, v in self.original_graph.edges():
            u_str, v_str = str(u), str(v)
            super_u = node_to_super.get(u_str, u_str)
            super_v = node_to_super.get(v_str, v_str)
            
            # Create canonical edge key (sorted to avoid duplicates)
            edge_key = tuple(sorted([super_u, super_v]))
            edge_weights[edge_key] += 1.0
        
        # Convert to edge list
        self.coarse_edges = [
            {"source": u, "target": v, "weight": w}
            for (u, v), w in edge_weights.items()
        ]
    
    def get_coarse_graph_info(self) -> Dict:
        """Get information about the coarse graph for LLM"""
        # Count internal vs external edges for each super-node
        super_node_stats = {}
        
        node_to_super = {}
        for super_node, original_nodes in self.super_node_to_original.items():
            for node in original_nodes:
                node_to_super[node] = super_node
        
        for super_node, original_nodes in self.super_node_to_original.items():
            internal_edges = 0
            external_edges = 0
            
            for node in original_nodes:
                for neighbor in self.original_graph.successors(str(node)):
                    if node_to_super.get(str(neighbor)) == super_node:
                        internal_edges += 1
                    else:
                        external_edges += 1
            
            super_node_stats[super_node] = {
                "size": len(original_nodes),
                "internal_edges": internal_edges,
                "external_edges": external_edges,
                "sample_nodes": original_nodes[:5]  # Show first 5 nodes as sample
            }
        
        return {
            "num_super_nodes": len(self.super_node_to_original),
            "num_coarse_edges": len(self.coarse_edges),
            "super_nodes": super_node_stats
        }
    
    def expand_clustering(self, super_node_clustering: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Expand LLM's clustering of super-nodes back to original nodes.
        
        Parameters:
        - super_node_clustering: LLM's output: cluster_name -> [super_node_names]
        
        Returns:
        - Original node clustering: cluster_name -> [original_node_names]
        """
        expanded_clustering: Dict[str, List[str]] = {}
        
        for cluster_name, super_nodes in super_node_clustering.items():
            expanded_nodes = []
            for super_node in super_nodes:
                # Get original nodes for this super-node
                original_nodes = self.super_node_to_original.get(super_node, [super_node])
                expanded_nodes.extend(original_nodes)
            
            expanded_clustering[cluster_name] = expanded_nodes
        
        return expanded_clustering
    
    def get_super_node_for_original(self, original_node: str) -> Optional[str]:
        """Find which super-node contains an original node"""
        for super_node, original_nodes in self.super_node_to_original.items():
            if original_node in original_nodes:
                return super_node
        return None


def compute_modularity(graph: LouvainGraph, partition: Dict[str, int]) -> float:
    """
    Compute the modularity of a partition.
    
    Q = (1/2m) * Σ [A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
    
    Where:
    - A_ij = edge weight between i and j
    - k_i = degree of node i
    - m = total edge weight
    - δ(c_i, c_j) = 1 if i and j in same community, 0 otherwise
    """
    m = graph.total_weight
    if m == 0:
        return 0.0
    
    Q = 0.0
    
    for node_i in graph.nodes:
        comm_i = partition[node_i]
        k_i = graph.node_weights[node_i]
        
        for node_j in graph.nodes:
            comm_j = partition[node_j]
            
            if comm_i != comm_j:
                continue
            
            k_j = graph.node_weights[node_j]
            A_ij = graph.get_edge_weight(node_i, node_j)
            
            Q += A_ij - (k_i * k_j) / (2 * m)
    
    return Q / (2 * m)


# ============================================================================
# Testing / Demo
# ============================================================================

def demo():
    """Demo the Louvain algorithm on a small test graph"""
    import networkx as nx
    
    # Create a test graph with clear community structure
    G = nx.Graph()
    
    # Community 1: nodes 0-4 (densely connected)
    for i in range(5):
        for j in range(i+1, 5):
            G.add_edge(i, j)
    
    # Community 2: nodes 5-9 (densely connected)
    for i in range(5, 10):
        for j in range(i+1, 10):
            G.add_edge(i, j)
    
    # Community 3: nodes 10-14 (densely connected)
    for i in range(10, 15):
        for j in range(i+1, 15):
            G.add_edge(i, j)
    
    # Sparse connections between communities
    G.add_edge(2, 7)
    G.add_edge(4, 10)
    G.add_edge(8, 12)
    
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Run Louvain
    lg = LouvainGraph.from_networkx(G)
    louvain = LouvainAlgorithm(random_seed=42)
    partitions = louvain.run(lg)
    
    print(f"\nFound {len(partitions)} partition level(s)")
    
    for level, partition in enumerate(partitions):
        communities = louvain.get_communities(partition)
        print(f"\nLevel {level}: {len(communities)} communities")
        for comm_name, nodes in communities.items():
            print(f"  {comm_name}: {nodes}")
        
        modularity = compute_modularity(lg, partition)
        print(f"  Modularity: {modularity:.4f}")
    
    # Test hierarchical coarsening
    print("\n" + "="*50)
    print("Testing Hierarchical Coarsening")
    print("="*50)
    
    hier = HierarchicalLouvain(G, num_coarsening_levels=1, random_seed=42)
    super_nodes, edges = hier.coarsen()
    
    print(f"\nSuper-nodes: {list(super_nodes.keys())}")
    for sn, nodes in super_nodes.items():
        print(f"  {sn}: {nodes}")
    
    print(f"\nCoarse edges: {edges}")
    
    # Simulate LLM clustering of super-nodes
    llm_clustering = {
        "cluster_A": ["super_0", "super_1"],
        "cluster_B": ["super_2"]
    }
    
    expanded = hier.expand_clustering(llm_clustering)
    print(f"\nExpanded clustering:")
    for cluster, nodes in expanded.items():
        print(f"  {cluster}: {nodes}")


if __name__ == "__main__":
    demo()
