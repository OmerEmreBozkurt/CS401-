import networkx as nx
from typing import Dict, List

class GraphDataProvider:
    """Holds the graph data and provides query functions"""
    
    def __init__(self, graph):
        self.graph = graph
        # print(f"📊 Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    def get_graph_info(self) -> Dict:
        """Get basic graph statistics"""
        G = self.graph
        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "avg_in_degree": sum(d for _, d in G.in_degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            "avg_out_degree": sum(d for _, d in G.out_degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
    
    def get_all_nodes(self) -> List[str]:
        """Get list of all nodes"""
        return list(self.graph.nodes())
    
    def get_all_edges(self) -> List[Dict]:
        """Get all dependency edges"""
        edges = []
        for u, v in self.graph.edges():
            edges.append({"source": u, "target": v})
        # Print summary instead of full list
        # print(f"[Returning {len(edges)} edges]")
        return edges
    
    def get_node_dependencies(self, node: str) -> List[str]:
        """Get what a node depends on"""
        if node not in self.graph:
            return []
        return list(self.graph.successors(node))
    
    def get_node_dependents(self, node: str) -> List[str]:
        """Get what depends on this node"""
        if node not in self.graph:
            return []
        return list(self.graph.predecessors(node))
    
    def get_hub_nodes(self, top_n: int = 10) -> List[Dict]:
        """Get most depended-on nodes"""
        in_degrees = dict(self.graph.in_degree())
        sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        return [{"node": node, "dependents": degree} for node, degree in sorted_nodes[:top_n]]
