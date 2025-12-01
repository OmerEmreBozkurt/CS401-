#!/usr/bin/env python3
"""
Interactive AutoGen Graph Analyzer with Metrics Agent
Pipeline: Cluster → Evaluate (TurboMQ + MoJo-FM) → Analyze → Improve → Done
"""

import sys
import os
import pickle
import json
import time
import subprocess
import tempfile
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
# PROMPTS
# ============================================================================

CLUSTERING_SYSTEM_PROMPT = """You are an expert in microservice architecture and dependency analysis.

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


METRICS_SYSTEM_PROMPT = """You are an expert in software architecture metrics evaluation.

You evaluate clustering quality using:
- TurboMQ: Measures cohesion (higher is better, range 0-1)
- MoJo-FM: Measures distance from reference (lower is better)

Your task:
1. Calculate metrics for the current clustering
2. Analyze the scores
3. Identify issues (low cohesion, high coupling, unbalanced clusters)
4. Suggest specific improvements
5. Provide actionable feedback for the clustering agent

Output your analysis as JSON with scores, issues, and recommendations."""


# ============================================================================
# Graph Data Provider
# ============================================================================

class GraphDataProvider:
    """Holds the graph data and provides query functions"""
    
    def __init__(self, graph):
        self.graph = graph
        print(f"📊 Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
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
        print(f"[Returning {len(edges)} edges]")
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


# ============================================================================
# Metrics Calculator
# ============================================================================

class MetricsCalculator:
    """Calculates TurboMQ and MoJo-FM metrics"""
    
    def __init__(self, dependency_rsf_path, experiments_dir=None):
        self.dependency_rsf_path = os.path.abspath(dependency_rsf_path) if dependency_rsf_path else None
        self.experiments_dir = experiments_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "experiments"
        )
        
        # Validate dependency RSF exists
        if self.dependency_rsf_path and not os.path.exists(self.dependency_rsf_path):
            print(f"⚠ Warning: Dependency RSF not found: {self.dependency_rsf_path}")
            print(f"   Current working directory: {os.getcwd()}")
            # Try to suggest correct path
            dirname = os.path.dirname(self.dependency_rsf_path)
            if os.path.exists(dirname):
                print(f"   Available files in {dirname}:")
                for f in os.listdir(dirname):
                    if f.endswith('.rsf'):
                        print(f"     - {f}")
    
    def clusters_to_rsf(self, clusters, output_path):
        """Convert clusters dict to RSF format"""
        with open(output_path, 'w') as f:
            for cluster_name, nodes in clusters.items():
                for node in nodes:
                    f.write(f"contain {cluster_name} {node}\n")
        print(f"✓ Converted clusters to RSF: {output_path}")
    
    def calculate_turbomq(self, clustering_rsf_path):
        """Calculate TurboMQ score using Java jar"""
        turbomq_jar = os.path.join(self.experiments_dir, "turbomq.jar")
        
        if not os.path.exists(turbomq_jar):
            print(f"⚠ Warning: {turbomq_jar} not found, skipping TurboMQ")
            return None
        
        # Validate input files exist
        if not os.path.exists(self.dependency_rsf_path):
            print(f"⚠ TurboMQ error: Dependency RSF not found: {self.dependency_rsf_path}")
            return None
        
        if not os.path.exists(clustering_rsf_path):
            print(f"⚠ TurboMQ error: Clustering RSF not found: {clustering_rsf_path}")
            return None
        
        try:
            import shutil
            
            # Copy files to experiments directory (jar expects relative paths from there)
            dep_rsf_temp = os.path.join(self.experiments_dir, "temp_dependency.rsf")
            clust_rsf_temp = os.path.join(self.experiments_dir, "temp_clustering.rsf")
            
            shutil.copy2(self.dependency_rsf_path, dep_rsf_temp)
            shutil.copy2(clustering_rsf_path, clust_rsf_temp)
            
            # Run from experiments directory with relative paths
            result = subprocess.run(
                ["java", "-jar", "turbomq.jar", "temp_dependency.rsf", "temp_clustering.rsf"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir
            )
            
            # Clean up temp files
            try:
                os.remove(dep_rsf_temp)
                os.remove(clust_rsf_temp)
            except:
                pass
            
            # Debug output
            print(f"   TurboMQ stdout: {result.stdout[:200]}")
            if result.stderr:
                print(f"   TurboMQ stderr: {result.stderr[:200]}")
            
            if result.returncode == 0:
                try:
                    score = float(result.stdout.strip())
                    return score
                except ValueError as e:
                    print(f"⚠ TurboMQ error: Could not parse output as float: {result.stdout.strip()}")
                    return None
            else:
                print(f"⚠ TurboMQ error (exit code {result.returncode}): {result.stderr}")
                return None
        except Exception as e:
            print(f"⚠ TurboMQ calculation failed: {e}")
            return None
    
    def calculate_mojo_fm(self, clustering_rsf_path, reference_rsf_path=None):
        """Calculate MoJo-FM distance using Java jar"""
        mojo_jar = os.path.join(self.experiments_dir, "mojo.jar")
        
        if not os.path.exists(mojo_jar):
            print(f"⚠ Warning: {mojo_jar} not found, skipping MoJo-FM")
            return None
        
        # If no reference, skip MoJo-FM (it needs a reference clustering)
        if not reference_rsf_path or not os.path.exists(reference_rsf_path):
            print(f"⚠ No reference RSF provided, skipping MoJo-FM")
            return None
        
        try:
            result = subprocess.run(
                ["java", "-jar", mojo_jar, clustering_rsf_path, reference_rsf_path, "-fm"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse MoJo-FM output (format may vary)
                output = result.stdout.strip()
                # Try to extract number
                import re
                match = re.search(r'[\d.]+', output)
                if match:
                    score = float(match.group())
                    return score
                return None
            else:
                print(f"⚠ MoJo-FM error: {result.stderr}")
                return None
        except Exception as e:
            print(f"⚠ MoJo-FM calculation failed: {e}")
            return None
    
    def calculate_metrics(self, clusters, reference_rsf_path=None):
        """Calculate all metrics for given clusters"""
        # Create temporary RSF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rsf', delete=False) as f:
            temp_rsf = f.name
        
        try:
            # Convert clusters to RSF
            self.clusters_to_rsf(clusters, temp_rsf)
            
            # Calculate metrics
            turbomq = self.calculate_turbomq(temp_rsf)
            mojo_fm = self.calculate_mojo_fm(temp_rsf, reference_rsf_path)
            
            return {
                "turbomq": turbomq,
                "mojo_fm": mojo_fm,
                "clustering_rsf": temp_rsf  # Keep for reference
            }
        except Exception as e:
            print(f"⚠ Metrics calculation error: {e}")
            return {"turbomq": None, "mojo_fm": None, "clustering_rsf": temp_rsf}
        finally:
            # Don't delete temp file yet - might need it
            pass


# ============================================================================
# Interactive Analyzer with Metrics
# ============================================================================

class InteractiveAnalyzerWithMetrics:
    """Analyzer with integrated metrics evaluation and improvement loop"""
    
    def __init__(self, ollama_host="http://localhost:11434", 
                 clustering_model="gpt-oss:120b-cloud",
                 metrics_model=None,
                 dependency_rsf_path=None, reference_rsf_path=None,
                 timeout=600, temperature=0):
        self.ollama_host = ollama_host
        self.clustering_model = clustering_model
        self.metrics_model = metrics_model or clustering_model  # Default to clustering model if not specified
        self.timeout = timeout
        self.temperature = temperature
        self.graph = None
        self.graph_provider = None
        self.metrics_calculator = None
        self.dependency_rsf_path = dependency_rsf_path
        self.reference_rsf_path = reference_rsf_path
        
    def load_graph(self, graph_file):
        """Load the graph"""
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")

        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)

        print(f"✓ Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        self.graph_provider = GraphDataProvider(self.graph)
        
        # Initialize metrics calculator if dependency RSF provided
        if self.dependency_rsf_path:
            self.metrics_calculator = MetricsCalculator(self.dependency_rsf_path)
            print(f"✓ Metrics calculator initialized with dependency RSF: {self.dependency_rsf_path}")

    def setup_clustering_agent(self, num_clusters):
        """Setup clustering agent"""
        system_prompt = CLUSTERING_SYSTEM_PROMPT + f"\n\nYour goal: Create {num_clusters} clusters from the graph."
        
        config_list = [{
            "model": self.clustering_model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": "ollama",
            "api_type": "openai",
            "temperature": self.temperature,
        }]
        
        self.clustering_agent = AssistantAgent(
            name="ClusteringExpert",
            system_message=system_prompt,
            llm_config={
                "config_list": config_list,
                "timeout": self.timeout,
                "cache_seed": None,
            }
        )
        
        self.user_proxy = UserProxyAgent(
            name="GraphProvider",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            default_auto_reply="",
            is_termination_msg=lambda x: "clusters" in x.get("content", "").lower() and "{" in x.get("content", ""),
            silent=True,  # Suppress verbose function output
        )
        
        self._register_graph_functions()
    
    def setup_metrics_agent(self):
        """Setup metrics evaluation agent"""
        config_list = [{
            "model": self.metrics_model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": "ollama",
            "api_type": "openai",
            "temperature": self.temperature,
        }]
        
        self.metrics_agent = AssistantAgent(
            name="MetricsEvaluator",
            system_message=METRICS_SYSTEM_PROMPT,
            llm_config={
                "config_list": config_list,
                "timeout": self.timeout,
                "cache_seed": None,
            }
        )
        
        self.metrics_proxy = UserProxyAgent(
            name="MetricsProvider",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False,
            silent=True,  # Suppress verbose function output
        )
        
        self._register_metrics_functions()
    
    def _register_graph_functions(self):
        """Register graph query functions"""
        provider = self.graph_provider
        
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
        
        for func, name, desc in [
            (get_graph_info, "get_graph_info", "Get basic graph statistics"),
            (get_all_nodes, "get_all_nodes", "Get list of all nodes"),
            (get_all_edges, "get_all_edges", "Get all dependency edges"),
            (get_node_dependencies, "get_node_dependencies", "Get what a node depends on"),
            (get_node_dependents, "get_node_dependents", "Get what depends on a node"),
            (get_hub_nodes, "get_hub_nodes", "Get most depended-on nodes"),
        ]:
            register_function(
                func,
                caller=self.clustering_agent,
                executor=self.user_proxy,
                name=name,
                description=desc
            )
        
        print("✓ Registered 6 graph query functions")
    
    def _register_metrics_functions(self):
        """Register metrics calculation functions"""
        calculator = self.metrics_calculator
        
        if not calculator:
            return
        
        def calculate_turbomq(clusters_json: str):
            """Calculate TurboMQ score for clusters"""
            clusters = json.loads(clusters_json)
            result = calculator.calculate_metrics(clusters, self.reference_rsf_path)
            return {
                "turbomq": result.get("turbomq"),
                "mojo_fm": result.get("mojo_fm"),
                "note": "TurboMQ: higher is better (0-1). MoJo-FM: lower is better."
            }
        
        register_function(
            calculate_turbomq,
            caller=self.metrics_agent,
            executor=self.metrics_proxy,
            name="calculate_turbomq",
            description="Calculate TurboMQ and MoJo-FM metrics for clustering"
        )
        
        print("✓ Registered metrics calculation function")
    
    def extract_clusters(self, agent):
        """Extract clusters from agent conversation"""
        try:
            for conv_id, messages in agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        import re
                        json_match = re.search(r'\{.*"clusters".*?\}', content, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                            if 'clusters' in data:
                                return data['clusters']
        except:
            pass
        return None
    
    def analyze_with_metrics(self, num_clusters=10, max_iterations=1):
        """
        Complete pipeline: Cluster → Evaluate → Analyze → Improve → Done
        """
        print(f"\n{'='*70}")
        print(f"INTERACTIVE ANALYSIS WITH METRICS")
        print(f"{'='*70}")
        print(f"Clustering Model: {self.clustering_model}")
        print(f"Metrics Model: {self.metrics_model}")
        print(f"Target clusters: {num_clusters}")
        print(f"Max iterations: {max_iterations}")
        
        # Setup agents
        self.setup_clustering_agent(num_clusters)
        if self.metrics_calculator:
            self.setup_metrics_agent()
        
        best_clusters = None
        best_scores = None
        iteration_history = []  # Track all iterations
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            
            # Step 1: Cluster
            print(f"\n📍 Step 1: Clustering...")
            clusters = self._run_clustering(num_clusters, iteration > 0, best_scores)
            
            if not clusters:
                print("⚠ Clustering failed, using fallback")
                clusters = self._fallback_clustering()
            
            # Step 2: Evaluate
            print(f"\n📍 Step 2: Evaluating metrics...")
            scores = self._evaluate_metrics(clusters)
            
            # Track this iteration
            iteration_history.append({
                "iteration": iteration + 1,
                "num_clusters": len(clusters),
                "turbomq": scores.get('turbomq') if scores else None,
                "mojo_fm": scores.get('mojo_fm') if scores else None
            })
            
            # Step 3: Analyze (if metrics agent available)
            if self.metrics_agent and scores:
                print(f"\n📍 Step 3: Analyzing scores...")
                analysis = self._analyze_scores(scores, clusters)
                
                # Check if we should improve
                if iteration < max_iterations - 1 and analysis:
                    print(f"\n📍 Step 4: Improving clustering...")
                    # Use analysis to guide next iteration
                    best_clusters = clusters
                    best_scores = scores
                    continue
                else:
                    # Done - use current clusters
                    best_clusters = clusters
                    best_scores = scores
                    break
            else:
                # No metrics agent - just use first clustering
                best_clusters = clusters
                best_scores = scores
                break
        
        # Final results
        stats = self._compute_stats(best_clusters)
        
        result = {
            "clustering_model": self.clustering_model,
            "metrics_model": self.metrics_model,
            "approach": "interactive_with_metrics",
            "clusters": best_clusters,
            "statistics": stats,
            "metrics": best_scores,
            "iteration_history": iteration_history,  # Add history
            "metadata": {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "iterations": iteration + 1,
            }
        }
        
        return result
    
    def _run_clustering(self, num_clusters, is_improvement, previous_scores):
        """Run clustering step"""
        if is_improvement and previous_scores:
            message = f"""Please improve the clustering based on these metrics:

Previous Scores:
- TurboMQ: {previous_scores.get('turbomq', 'N/A')} (higher is better, aim for >0.7)
- MoJo-FM: {previous_scores.get('mojo_fm', 'N/A')} (lower is better)

Issues to address:
- Low TurboMQ suggests poor cohesion
- High MoJo-FM suggests clustering doesn't match expected structure

Create {num_clusters} improved clusters that address these issues.
Output JSON with clusters."""
        else:
            message = f"""Please analyze the dependency graph and create {num_clusters} microservice clusters.

Steps:
1. Use get_graph_info() to understand the graph structure
2. Use get_all_edges() to see all dependencies
3. Analyze the dependency patterns
4. Create {num_clusters} clusters that minimize inter-cluster dependencies

Output your clustering as JSON:
{{
  "clusters": {{
    "cluster_1": ["node1", "node2", ...],
    "cluster_2": ["node3", "node4", ...],
    ...
  }}
}}

Start by calling the functions to explore the graph!"""
        
        try:
            self.user_proxy.initiate_chat(
                self.clustering_agent,
                message=message,
                max_turns=15,
            )
            
            return self.extract_clusters(self.clustering_agent)
        except Exception as e:
            print(f"⚠ Clustering error: {e}")
            return None
    
    def _evaluate_metrics(self, clusters):
        """Evaluate metrics for clusters"""
        if not self.metrics_calculator:
            print("⚠ No metrics calculator available")
            return None
        
        try:
            result = self.metrics_calculator.calculate_metrics(clusters, self.reference_rsf_path)
            
            print(f"✓ Metrics calculated:")
            if result.get('turbomq') is not None:
                print(f"  TurboMQ: {result['turbomq']:.4f} (higher is better, range 0-1)")
            if result.get('mojo_fm') is not None:
                print(f"  MoJo-FM: {result['mojo_fm']:.4f} (lower is better)")
            
            return result
        except Exception as e:
            print(f"⚠ Metrics evaluation error: {e}")
            return None
    
    def _analyze_scores(self, scores, clusters):
        """Have metrics agent analyze the scores"""
        if not self.metrics_agent:
            return None
        
        message = f"""Analyze these clustering metrics:

Scores:
- TurboMQ: {scores.get('turbomq', 'N/A')} (target: >0.7)
- MoJo-FM: {scores.get('mojo_fm', 'N/A')} (target: lower is better)

Clusters: {len(clusters)} clusters

Provide analysis:
1. Are the scores good or need improvement?
2. What specific issues exist?
3. What improvements should be made?

Output JSON with analysis and recommendations."""
        
        try:
            self.metrics_proxy.initiate_chat(
                self.metrics_agent,
                message=message,
                max_turns=5,
            )
            
            # Extract analysis from conversation
            for conv_id, messages in self.metrics_agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        return msg.get('content', '')
        except Exception as e:
            print(f"⚠ Analysis error: {e}")
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
        """Compute clustering statistics"""
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
        description="Interactive analyzer with metrics evaluation and improvement"
    )
    parser.add_argument("--config", default="config.json",
                       help="Path to config file (default: config.json)")
    parser.add_argument("graph_file", nargs='?', default=None,
                       help="Path to pickled graph file (overrides config)")
    parser.add_argument("--dependency-rsf", 
                       help="Path to dependency RSF file (for metrics, overrides config)")
    parser.add_argument("--reference-rsf", 
                       help="Path to reference clustering RSF (for MoJo-FM, overrides config)")
    parser.add_argument("--clustering-model",
                       help="Ollama model for clustering agent (overrides config)")
    parser.add_argument("--metrics-model",
                       help="Ollama model for metrics agent (overrides config)")
    parser.add_argument("--clusters", type=int,
                       help="Number of clusters (overrides config)")
    parser.add_argument("--iterations", type=int,
                       help="Max improvement iterations (overrides config)")
    parser.add_argument("--output",
                       help="Output file (overrides config)")
    parser.add_argument("--host",
                       help="Ollama host (overrides config)")
    parser.add_argument("--timeout", type=int,
                       help="Request timeout in seconds (overrides config)")
    parser.add_argument("--temperature", type=float,
                       help="Temperature for LLM (overrides config)")

    args = parser.parse_args()

    # Load config file
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded config from {config_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load config file: {e}")
            print("   Using defaults and command line arguments")
    else:
        print(f"⚠ Warning: Config file not found: {config_path}")
        print("   Using defaults and command line arguments")

    # Get run_config or default_settings
    run_config = config.get("run_config", config.get("default_settings", {}))
    
    # Merge config with command line arguments (CLI overrides config)
    graph_file = args.graph_file or run_config.get("graph_file")
    if not graph_file:
        print("❌ Error: graph_file must be provided either in config or as argument")
        sys.exit(1)
    
    dependency_rsf = args.dependency_rsf if args.dependency_rsf is not None else run_config.get("dependency_rsf")
    reference_rsf = args.reference_rsf if args.reference_rsf is not None else run_config.get("reference_rsf")
    clustering_model = args.clustering_model or run_config.get("clustering_model") or run_config.get("model", "gpt-oss:120b-cloud")
    metrics_model = args.metrics_model or run_config.get("metrics_model") or clustering_model
    clusters = args.clusters if args.clusters is not None else run_config.get("clusters", 10)
    iterations = args.iterations if args.iterations is not None else run_config.get("iterations", 1)
    output = args.output or run_config.get("output", "clusters_with_metrics.json")
    host = args.host or run_config.get("ollama_host", "http://localhost:11434")
    timeout = args.timeout if args.timeout is not None else run_config.get("timeout", 600)
    temperature = args.temperature if args.temperature is not None else run_config.get("temperature", 0)

    # Create analyzer
    analyzer = InteractiveAnalyzerWithMetrics(
        ollama_host=host,
        clustering_model=clustering_model,
        metrics_model=metrics_model,
        dependency_rsf_path=dependency_rsf,
        reference_rsf_path=reference_rsf,
        timeout=timeout,
        temperature=temperature
    )

    # Load graph
    try:
        analyzer.load_graph(graph_file)
    except FileNotFoundError:
        print(f"❌ Graph file not found: {graph_file}")
        sys.exit(1)

    # Run analysis with metrics
    result = analyzer.analyze_with_metrics(
        num_clusters=clusters,
        max_iterations=iterations
    )

    if result:
        # Save
        analyzer.save_results(result, output)

        # Display summary
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Clusters: {result['statistics']['num_clusters']}")
        print(f"Cohesion Score: {result['statistics'].get('cohesion_score', 0):.2%}")
        print(f"Coupling Score: {result['statistics'].get('coupling_score', 0):.2%}")
        
        if result.get('metrics'):
            metrics = result['metrics']
            if metrics.get('turbomq') is not None:
                print(f"TurboMQ: {metrics['turbomq']:.4f}")
            if metrics.get('mojo_fm') is not None:
                print(f"MoJo-FM: {metrics['mojo_fm']:.4f}")
        
        # Show iteration history if multiple iterations
        if len(result.get('iteration_history', [])) > 1:
            print(f"\n{'='*70}")
            print(f"ITERATION HISTORY")
            print(f"{'='*70}")
            for hist in result['iteration_history']:
                turbo_str = f"{hist['turbomq']:.4f}" if hist['turbomq'] is not None else "N/A"
                mojo_str = f"{hist['mojo_fm']:.4f}" if hist['mojo_fm'] is not None else "N/A"
                print(f"Iteration {hist['iteration']}: {hist['num_clusters']} clusters, TurboMQ={turbo_str}, MoJo-FM={mojo_str}")
            
            # Calculate improvement
            if len(result['iteration_history']) > 1:
                first = result['iteration_history'][0]
                last = result['iteration_history'][-1]
                if first['turbomq'] is not None and last['turbomq'] is not None:
                    improvement = last['turbomq'] - first['turbomq']
                    improvement_pct = (improvement / first['turbomq'] * 100) if first['turbomq'] > 0 else 0
                    print(f"\n📈 TurboMQ Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        print(f"\nCluster Sizes:")
        for name, size in result['statistics']['cluster_sizes'].items():
            print(f"  {name}: {size} nodes")
    else:
        print("❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

