#!/usr/bin/env python3
"""
Interactive AutoGen Graph Analyzer with Function Calling
The graph is NOT embedded in prompts - agent queries it via functions
Agent can request graph data whenever it needs it
"""

import sys
import os
import pickle
import json
import time
from typing import Dict, List, Annotated

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, register_function
except ImportError:
    print("❌ Error: AutoGen not installed")
    print("Install with: pip install pyautogen")
    sys.exit(1)

import networkx as nx


# ============================================================================
# PROMPTS - EDIT THESE!
# ============================================================================

SYSTEM_PROMPT = """You are an expert in microservice architecture and dependency analysis.

You have access to a dependency graph through these functions:
- get_graph_info() - Get basic graph statistics
- get_all_nodes() - Get list of all nodes
- get_all_edges() - Get all dependency edges
- get_node_dependencies(node) - Get dependencies of a specific node
- get_node_dependents(node) - Get what depends on a specific node

Your task: Analyze the graph and cluster nodes into microservices.

Strategy:
1. First, call get_graph_info() to understand the graph
2. Call get_all_edges() to see all dependencies
3. Analyze the dependency patterns
4. Create clusters that minimize inter-cluster dependencies
5. Output your clustering as JSON

Important: Call the functions to get the data you need!"""


# ============================================================================
# Graph Data Holder (Agent can query this)
# ============================================================================

class GraphDataProvider:
    """
    Holds the graph data and provides query functions
    Agent can call these functions to get graph information
    """
    
    def __init__(self, graph):
        self.graph = graph
        print(f"📊 Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"   Agent can now query this graph via functions")
    
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
        return edges
    
    def get_node_dependencies(self, node: str) -> List[str]:
        """Get what a node depends on (outgoing edges)"""
        if node not in self.graph:
            return []
        return list(self.graph.successors(node))
    
    def get_node_dependents(self, node: str) -> List[str]:
        """Get what depends on this node (incoming edges)"""
        if node not in self.graph:
            return []
        return list(self.graph.predecessors(node))
    
    def get_hub_nodes(self, top_n: int = 10) -> List[Dict]:
        """Get most depended-on nodes"""
        in_degrees = dict(self.graph.in_degree())
        sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        return [{"node": node, "dependents": degree} for node, degree in sorted_nodes[:top_n]]


# ============================================================================
# Interactive Analyzer
# ============================================================================

class InteractiveAutoGenAnalyzer:
    """
    Agent queries graph via functions instead of receiving it in prompts
    """
    
    def __init__(self, ollama_host="http://localhost:11434", model="gpt-oss:120b-cloud"):
        self.ollama_host = ollama_host
        self.model = model
        self.graph = None
        self.graph_provider = None
        
    def load_graph(self, graph_file):
        """Load the graph"""
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")

        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)

        print(f"✓ Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Create graph provider (agent will query this)
        self.graph_provider = GraphDataProvider(self.graph)

    def setup_agents(self, num_clusters, system_prompt=None):
        """Setup AutoGen agents with function calling"""
        
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        
        # Add task description to system prompt
        system_prompt += f"\n\nYour goal: Create {num_clusters} clusters from the graph."
        
        # Configuration for Ollama
        config_list = [{
            "model": self.model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": "ollama",
            "api_type": "openai",
            "temperature": 0,
        }]
        
        # Assistant agent with function calling
        self.assistant = AssistantAgent(
            name="ClusteringExpert",
            system_message=system_prompt,
            llm_config={
                "config_list": config_list,
                "timeout": 600,
                "cache_seed": None,
            }
        )
        
        # User proxy (executes functions)
        self.user_proxy = UserProxyAgent(
            name="GraphProvider",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=20,  # Allow multiple back-and-forth
            code_execution_config=False,
        )
        
        # Register graph query functions
        self._register_functions()

    def _register_functions(self):
        """Register functions that agent can call to query graph"""
        
        provider = self.graph_provider
        
        # Create wrapper functions (AutoGen needs standalone functions, not bound methods)
        def get_graph_info():
            return provider.get_graph_info()
        
        def get_all_nodes():
            return provider.get_all_nodes()
        
        def get_all_edges():
            return provider.get_all_edges()
        
        def get_node_dependencies(node: str):
            return provider.get_node_dependencies(node)
        
        def get_node_dependents(node: str):
            return provider.get_node_dependents(node)
        
        def get_hub_nodes(top_n: int = 10):
            return provider.get_hub_nodes(top_n)
        
        # Register each function
        register_function(
            get_graph_info,
            caller=self.assistant,
            executor=self.user_proxy,
            name="get_graph_info",
            description="Get basic statistics about the graph (nodes, edges, density, etc.)"
        )
        
        register_function(
            get_all_nodes,
            caller=self.assistant,
            executor=self.user_proxy,
            name="get_all_nodes",
            description="Get a list of all nodes in the graph"
        )
        
        register_function(
            get_all_edges,
            caller=self.assistant,
            executor=self.user_proxy,
            name="get_all_edges",
            description="Get all dependency edges in the graph"
        )
        
        register_function(
            get_node_dependencies,
            caller=self.assistant,
            executor=self.user_proxy,
            name="get_node_dependencies",
            description="Get what a specific node depends on"
        )
        
        register_function(
            get_node_dependents,
            caller=self.assistant,
            executor=self.user_proxy,
            name="get_node_dependents",
            description="Get what depends on a specific node"
        )
        
        register_function(
            get_hub_nodes,
            caller=self.assistant,
            executor=self.user_proxy,
            name="get_hub_nodes",
            description="Get the most depended-on nodes (hubs)"
        )
        
        print("✓ Registered 6 graph query functions for agent")

    def analyze(self, num_clusters=10):
        """
        Interactive analysis - agent queries graph via functions
        """
        print(f"\n{'='*70}")
        print(f"INTERACTIVE AUTOGEN ANALYSIS")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Target clusters: {num_clusters}")
        print(f"\nAgent will query the graph via functions (not embedded in prompt)")
        
        # Setup agents with function calling
        self.setup_agents(num_clusters)
        
        # Initial message to agent (NO GRAPH DATA)
        initial_message = f"""Please analyze the dependency graph and create {num_clusters} microservice clusters.

Steps:
1. Use get_graph_info() to understand the graph structure
2. Use get_all_edges() to see all dependencies
3. Optionally use get_hub_nodes() to identify shared services
4. Analyze the dependency patterns
5. Create {num_clusters} clusters that minimize inter-cluster dependencies

Output your final clustering as JSON:
{{
  "clusters": {{
    "cluster_1": ["node1", "node2", ...],
    "cluster_2": ["node3", "node4", ...],
    ...
  }}
}}

Start by calling the functions to explore the graph!"""

        print(f"\n🤖 Starting interactive analysis...")
        print(f"   Agent can call functions to query graph data")
        start_time = time.time()
        
        try:
            # Start the conversation
            self.user_proxy.initiate_chat(
                self.assistant,
                message=initial_message
            )
            
            elapsed = time.time() - start_time
            print(f"\n✓ Analysis completed in {elapsed:.1f} seconds")
            
            # Extract clusters from conversation
            clusters = self._extract_clusters()
            
            if not clusters:
                print("⚠ Could not extract clusters, using fallback")
                clusters = self._fallback_clustering()
            
            # Compute stats
            stats = self._compute_stats(clusters)
            
            result = {
                "model": self.model,
                "approach": "interactive_function_calling",
                "clusters": clusters,
                "statistics": stats,
                "metadata": {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "processing_time": f"{elapsed:.1f}s",
                    "graph_queried_via_functions": True
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_clusters(self):
        """Extract clusters from conversation"""
        try:
            # Get messages from assistant
            for conv_id, messages in self.assistant.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        
                        # Try to parse JSON
                        import re
                        json_match = re.search(r'\{.*"clusters".*?\}', content, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                            if 'clusters' in data:
                                return data['clusters']
        except Exception as e:
            print(f"⚠ Error extracting clusters: {e}")
            pass
        
        return None

    def _fallback_clustering(self):
        """Fallback using NetworkX"""
        try:
            import networkx.algorithms.community as nx_comm
            G_undirected = self.graph.to_undirected()
            communities = nx_comm.louvain_communities(G_undirected, seed=42)
            
            clusters = {}
            for i, community in enumerate(communities):
                clusters[f"cluster_{i + 1}"] = list(community)
            
            print(f"✓ Fallback created {len(clusters)} clusters")
            return clusters
        except:
            return {"cluster_1": list(self.graph.nodes())}

    def _compute_stats(self, clusters):
        """Compute statistics"""
        stats = {
            "num_clusters": len(clusters),
            "cluster_sizes": {},
            "intra_cluster_edges": 0,
            "inter_cluster_edges": 0
        }

        node_to_cluster = {}
        for cname, members in clusters.items():
            stats["cluster_sizes"][cname] = len(members)
            for node in members:
                node_to_cluster[node] = cname

        for u, v in self.graph.edges():
            if u in node_to_cluster and v in node_to_cluster:
                if node_to_cluster[u] == node_to_cluster[v]:
                    stats["intra_cluster_edges"] += 1
                else:
                    stats["inter_cluster_edges"] += 1

        total = stats["intra_cluster_edges"] + stats["inter_cluster_edges"]
        if total > 0:
            stats["cohesion_score"] = stats["intra_cluster_edges"] / total
            stats["coupling_score"] = stats["inter_cluster_edges"] / total

        return stats

    def save_results(self, result, output_file):
        """Save results"""
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive analyzer - agent queries graph via functions"
    )
    parser.add_argument("graph_file", help="Path to pickled graph file")
    parser.add_argument("--model", default="gpt-oss:120b-cloud", 
                       help="Ollama model (default: gpt-oss:120b-cloud)")
    parser.add_argument("--clusters", type=int, default=10,
                       help="Number of clusters (default: 10)")
    parser.add_argument("--output", default="interactive_clusters.json",
                       help="Output file (default: interactive_clusters.json)")
    parser.add_argument("--host", default="http://localhost:11434",
                       help="Ollama host (default: http://localhost:11434)")

    args = parser.parse_args()

    # Create analyzer
    analyzer = InteractiveAutoGenAnalyzer(
        ollama_host=args.host,
        model=args.model
    )

    # Load graph
    try:
        analyzer.load_graph(args.graph_file)
    except FileNotFoundError:
        print(f"❌ Graph file not found: {args.graph_file}")
        sys.exit(1)

    # Analyze
    result = analyzer.analyze(num_clusters=args.clusters)

    if result:
        # Save
        analyzer.save_results(result, args.output)

        # Display summary
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Clusters: {result['statistics']['num_clusters']}")
        print(f"Cohesion Score: {result['statistics'].get('cohesion_score', 0):.2%}")
        print(f"Coupling Score: {result['statistics'].get('coupling_score', 0):.2%}")
        
        print(f"\nCluster Sizes:")
        for name, size in result['statistics']['cluster_sizes'].items():
            print(f"  {name}: {size} nodes")
    else:
        print("❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

