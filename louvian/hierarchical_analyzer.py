#!/usr/bin/env python3
"""
Interactive Graph Analyzer with Hierarchical Louvain Pre-processing

This version uses Louvain algorithm to coarsen the graph BEFORE sending to LLM:
1. Louvain coarsening: 234 nodes → ~20-50 super-nodes
2. LLM clustering: Works on the smaller super-node graph
3. Expansion: Map LLM's clustering back to original nodes
4. Metrics: Calculate on original node clustering

Benefits:
- LLM sees a manageable number of nodes
- LLM can focus on high-level structure
- Original granularity preserved through expansion
"""

import sys
import os
import pickle
import json
import time
import argparse
import contextlib
import io
import re
import subprocess
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, register_function
except ImportError:
    print("❌ Error: AutoGen not installed")
    print("Install with: pip install pyautogen")
    sys.exit(1)

import networkx as nx

# Import our custom Louvain implementation
from louvain_hierarchical import LouvainGraph, LouvainAlgorithm, HierarchicalLouvain, compute_modularity


# ============================================================================
# Prompts
# ============================================================================

CLUSTERING_SYSTEM_PROMPT = """You are an expert in microservice architecture and dependency analysis.

You are working with a PRE-PROCESSED graph where nodes have been grouped into "super-nodes" 
using the Louvain algorithm. Each super-node represents a group of related original nodes.

You have access to these functions:
- get_graph_info() - Get statistics about the super-node graph
- get_all_super_nodes() - Get list of all super-nodes with their sizes and sample members
- get_super_node_edges() - Get edges between super-nodes (with weights = # of original edges)
- get_super_node_details(super_node) - Get details about a specific super-node

Your task: Cluster the SUPER-NODES into microservice clusters.

Strategy:
1. First, call get_graph_info() to understand the graph
2. Call get_all_super_nodes() to see what super-nodes exist and their sizes
3. Call get_super_node_edges() to see connections between super-nodes
4. Group related super-nodes together based on edge weights
5. Output your clustering in SIMPLE TEXT FORMAT

CRITICAL: You are clustering SUPER-NODES (like super_0, super_1, etc), NOT original nodes!

Output format:
```
CLUSTER cluster_1:
super_0
super_3
super_5

CLUSTER cluster_2:
super_1
super_2
```

Each cluster should contain super-node names (super_X format).
The super-nodes will later be expanded back to original nodes."""


INSPECTOR_SYSTEM_PROMPT = """You are the Lead Software Architect and Quality Inspector.

You are reviewing a clustering of SUPER-NODES (pre-grouped communities of nodes).
Each super-node represents multiple original nodes that were grouped by the Louvain algorithm.

Your goal: Identify problems in the super-node clustering.

You have access to:
- get_super_node_details(super_node) - See what's inside a super-node
- get_super_node_edges() - See connections between super-nodes

Process:
1. Review the proposed clustering and metrics
2. Check if highly-connected super-nodes are split across clusters
3. Look for super-nodes that should be moved

Output your feedback as specific orders:
{
  "analysis": "Your analysis...",
  "specific_orders": ["Move super_X to cluster_Y", "Merge cluster_A and cluster_B"]
}
"""


# ============================================================================
# Graph Data Provider for Coarsened Graph
# ============================================================================

class CoarseGraphProvider:
    """Provides query functions for the coarsened (super-node) graph"""
    
    def __init__(self, hierarchical_louvain: HierarchicalLouvain):
        self.hier = hierarchical_louvain
    
    def get_graph_info(self) -> str:
        """Get basic statistics about the super-node graph"""
        info = self.hier.get_coarse_graph_info()
        
        # Simplify for LLM
        summary = {
            "num_super_nodes": info["num_super_nodes"],
            "num_edges_between_super_nodes": info["num_coarse_edges"],
            "original_nodes_total": sum(sn["size"] for sn in info["super_nodes"].values()),
            "average_super_node_size": round(
                sum(sn["size"] for sn in info["super_nodes"].values()) / info["num_super_nodes"], 1
            ) if info["num_super_nodes"] > 0 else 0
        }
        return json.dumps(summary, indent=2)
    
    def get_all_super_nodes(self) -> str:
        """Get all super-nodes with their sizes and sample members"""
        super_nodes = []
        
        for super_node, original_nodes in self.hier.super_node_to_original.items():
            super_nodes.append({
                "name": super_node,
                "size": len(original_nodes),
                "sample_members": original_nodes[:5],  # First 5 as sample
                "all_members_if_small": original_nodes if len(original_nodes) <= 10 else None
            })
        
        # Sort by size descending
        super_nodes.sort(key=lambda x: x["size"], reverse=True)
        
        return json.dumps(super_nodes, indent=2)
    
    def get_super_node_edges(self) -> str:
        """Get edges between super-nodes with weights"""
        # Filter out self-loops and sort by weight
        edges = [e for e in self.hier.coarse_edges if e["source"] != e["target"]]
        edges.sort(key=lambda x: x["weight"], reverse=True)
        
        return json.dumps(edges, indent=2)
    
    def get_super_node_details(self, super_node: str) -> str:
        """Get detailed information about a specific super-node"""
        if super_node not in self.hier.super_node_to_original:
            return json.dumps({"error": f"Super-node '{super_node}' not found"})
        
        original_nodes = self.hier.super_node_to_original[super_node]
        
        # Find edges to other super-nodes
        connected_to = defaultdict(float)
        for edge in self.hier.coarse_edges:
            if edge["source"] == super_node and edge["target"] != super_node:
                connected_to[edge["target"]] += edge["weight"]
            elif edge["target"] == super_node and edge["source"] != super_node:
                connected_to[edge["source"]] += edge["weight"]
        
        return json.dumps({
            "name": super_node,
            "size": len(original_nodes),
            "members": original_nodes,
            "connected_to": dict(connected_to)
        }, indent=2)


# ============================================================================
# Metrics Calculator (User's Original Working Version)
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
            pass
    
    def clusters_to_rsf(self, clusters, output_path):
        """Convert clusters dict to RSF format"""
        with open(output_path, 'w') as f:
            for cluster_name, nodes in clusters.items():
                for node in nodes:
                    f.write(f"contain {cluster_name} {node}\n")
    
    def calculate_turbomq(self, clustering_rsf_path):
        """Calculate TurboMQ score using Java jar"""
        import shutil
        
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
            # Copy files to experiments directory to avoid path issues with the jar
            dep_rsf_temp = os.path.join(self.experiments_dir, "temp_dependency.rsf")
            clust_rsf_temp = os.path.join(self.experiments_dir, "temp_clustering.rsf")
            
            shutil.copy2(self.dependency_rsf_path, dep_rsf_temp)
            shutil.copy2(clustering_rsf_path, clust_rsf_temp)
            
            # Debug output
            cmd = ["java", "-jar", "turbomq.jar", "temp_dependency.rsf", "temp_clustering.rsf"]
            print(f"   Executing: {' '.join(cmd)}")
            print(f"   CWD: {self.experiments_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir
            )
            
            # Clean up temp files
            try:
                if os.path.exists(dep_rsf_temp):
                    os.remove(dep_rsf_temp)
                if os.path.exists(clust_rsf_temp):
                    os.remove(clust_rsf_temp)
            except:
                pass
            
            # Debug output
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
            # Use absolute paths
            clust_rsf_abs = os.path.abspath(clustering_rsf_path)
            ref_rsf_abs = os.path.abspath(reference_rsf_path)
            
            cmd = ["java", "-jar", "mojo.jar", clust_rsf_abs, ref_rsf_abs, "-fm"]
            print(f"   Executing: {' '.join(cmd)}")
            print(f"   CWD: {self.experiments_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir
            )
            
            if result.returncode == 0:
                # Parse MoJo-FM output (format may vary)
                output = result.stdout.strip()
                # Try to extract number
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
        import tempfile
        
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
# Main Analyzer
# ============================================================================

class HierarchicalAnalyzer:
    """
    Graph analyzer with Louvain pre-processing for LLM-friendly clustering.
    """
    
    def __init__(self, ollama_host="http://localhost:11434", model="llama3:8b",
                 dependency_rsf_path=None, reference_rsf_path=None, 
                 verbose=False, louvain_levels=2, louvain_resolution=1.0):
        self.ollama_host = ollama_host
        self.model = model
        self.dependency_rsf_path = dependency_rsf_path
        self.reference_rsf_path = reference_rsf_path
        self.verbose = verbose
        self.louvain_levels = louvain_levels
        self.louvain_resolution = louvain_resolution
        
        # Components
        self.graph = None
        self.hierarchical_louvain: Optional[HierarchicalLouvain] = None
        self.coarse_provider: Optional[CoarseGraphProvider] = None
        self.metrics_calculator: Optional[MetricsCalculator] = None
        
        # AutoGen agents
        self.clustering_agent = None
        self.user_proxy = None
        self.inspector_agent = None
        self.inspector_proxy = None
    
    def load_graph(self, graph_file):
        """Load graph and run Louvain coarsening"""
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        
        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)
        
        print(f"✓ Loaded graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        # Run Louvain coarsening
        self.hierarchical_louvain = HierarchicalLouvain(
            self.graph,
            num_coarsening_levels=self.louvain_levels,
            resolution=self.louvain_resolution,
            random_seed=42
        )
        self.hierarchical_louvain.coarsen()
        
        # Create provider for coarse graph
        self.coarse_provider = CoarseGraphProvider(self.hierarchical_louvain)
        
        # Initialize metrics calculator
        if self.dependency_rsf_path:
            self.metrics_calculator = MetricsCalculator(self.dependency_rsf_path)
    
    def setup_clustering_agent(self, num_clusters):
        """Setup the clustering agent for super-node clustering"""
        if num_clusters is None or num_clusters <= 0:
            cluster_goal = "Decide the optimal number of clusters based on the super-node structure."
        else:
            cluster_goal = f"Create exactly {num_clusters} clusters from the super-nodes."
        
        system_prompt = CLUSTERING_SYSTEM_PROMPT + f"\n\nYour goal: {cluster_goal}"
        
        config_list = [{
            "model": self.model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": "ollama",
            "api_type": "openai",
            "temperature": 0.1,
        }]
        
        self.clustering_agent = AssistantAgent(
            name="ClusteringExpert",
            system_message=system_prompt,
            llm_config={
                "config_list": config_list,
                "timeout": 600,
                "cache_seed": None,
            }
        )
        
        def termination_check(msg):
            content = msg.get("content", "")
            if "CLUSTER" in content and "super_" in content:
                return True
            return False
        
        self.user_proxy = UserProxyAgent(
            name="GraphProvider",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=15,
            code_execution_config=False,
            default_auto_reply="Please continue and output the final clustering.",
            is_termination_msg=termination_check,
            silent=not self.verbose,
        )
        
        self._register_coarse_functions(self.clustering_agent, self.user_proxy)
    
    def setup_inspector_agent(self):
        """Setup the inspector agent"""
        config_list = [{
            "model": self.model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": "ollama",
            "api_type": "openai",
            "temperature": 0,
        }]
        
        self.inspector_agent = AssistantAgent(
            name="Inspector",
            system_message=INSPECTOR_SYSTEM_PROMPT,
            llm_config={
                "config_list": config_list,
                "timeout": 300,
                "cache_seed": None,
            }
        )
        
        self.inspector_proxy = UserProxyAgent(
            name="InspectorExecutor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            code_execution_config=False,
            is_termination_msg=lambda msg: "specific_orders" in msg.get("content", ""),
            silent=not self.verbose,
        )
        
        self._register_coarse_functions(self.inspector_agent, self.inspector_proxy)
    
    def _register_coarse_functions(self, agent, executor):
        """Register functions for querying the coarse graph"""
        provider = self.coarse_provider
        
        def get_graph_info() -> str:
            return provider.get_graph_info()
        
        def get_all_super_nodes() -> str:
            return provider.get_all_super_nodes()
        
        def get_super_node_edges() -> str:
            return provider.get_super_node_edges()
        
        def get_super_node_details(super_node: str) -> str:
            return provider.get_super_node_details(super_node)
        
        functions = [
            (get_graph_info, "get_graph_info", "Get statistics about the super-node graph"),
            (get_all_super_nodes, "get_all_super_nodes", "Get all super-nodes with sizes and members"),
            (get_super_node_edges, "get_super_node_edges", "Get edges between super-nodes"),
            (get_super_node_details, "get_super_node_details", "Get details about a specific super-node"),
        ]
        
        for func, name, desc in functions:
            register_function(func, caller=agent, executor=executor, name=name, description=desc)
    
    def extract_super_node_clusters(self, agent) -> Optional[Dict[str, List[str]]]:
        """Extract clustering of super-nodes from agent conversation"""
        try:
            for conv_id, messages in agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        
                        # Parse simple text format
                        clusters = self._parse_text_clusters(content)
                        if clusters:
                            return clusters
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Extraction error: {e}")
        return None
    
    def _parse_text_clusters(self, content: str) -> Optional[Dict[str, List[str]]]:
        """Parse simple text format clustering"""
        clusters = {}
        current_cluster = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.upper().startswith('CLUSTER'):
                parts = line.split()
                if len(parts) >= 2:
                    cluster_name = parts[1].rstrip(':')
                    current_cluster = cluster_name
                    clusters[current_cluster] = []
            
            elif current_cluster and line.startswith('super_'):
                # Extract super-node name
                super_node = line.split()[0] if line else None
                if super_node:
                    clusters[current_cluster].append(super_node)
        
        # Validate we got something
        if clusters and any(len(nodes) > 0 for nodes in clusters.values()):
            return clusters
        return None
    
    def analyze(self, num_clusters=None, max_iterations=1):
        """
        Main analysis pipeline:
        1. Louvain coarsening (done in load_graph)
        2. LLM clustering of super-nodes
        3. Expand to original nodes
        4. Calculate metrics
        5. Optional: iterate with inspector
        """
        num_super_nodes = len(self.hierarchical_louvain.super_node_to_original)
        
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL CLUSTERING ANALYSIS")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Original nodes: {self.graph.number_of_nodes()}")
        print(f"Super-nodes (after Louvain): {num_super_nodes}")
        print(f"Louvain levels: {self.louvain_levels}")
        print(f"Target clusters: {num_clusters if num_clusters else 'LLM decides'}")
        print(f"Max iterations: {max_iterations}")
        
        # Setup agents
        self.setup_clustering_agent(num_clusters)
        if self.metrics_calculator:
            self.setup_inspector_agent()
        
        best_clusters = None
        best_scores = None
        iteration_history = []
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            
            # Step 1: Cluster super-nodes
            print(f"\n📍 Step 1: Clustering {num_super_nodes} super-nodes...")
            super_node_clusters = self._run_clustering(num_clusters)
            
            if not super_node_clusters:
                print("⚠ LLM clustering failed, using Louvain directly")
                super_node_clusters = self._fallback_clustering(num_clusters)
            else:
                print(f"✓ LLM created {len(super_node_clusters)} clusters of super-nodes")
            
            # Step 2: Expand to original nodes
            print(f"\n📍 Step 2: Expanding super-nodes to original nodes...")
            original_clusters = self.hierarchical_louvain.expand_clustering(super_node_clusters)
            
            # Validate all nodes covered
            original_clusters = self._validate_and_fix_clusters(original_clusters)
            
            total_nodes = sum(len(nodes) for nodes in original_clusters.values())
            print(f"✓ Expanded to {total_nodes} original nodes in {len(original_clusters)} clusters")
            
            # Step 3: Calculate metrics
            print(f"\n📍 Step 3: Calculating metrics...")
            scores = self._evaluate_metrics(original_clusters)
            
            iteration_history.append({
                "iteration": iteration + 1,
                "num_clusters": len(original_clusters),
                "turbomq": scores.get('turbomq') if scores else None,
                "mojo_fm": scores.get('mojo_fm') if scores else None,
            })
            
            # Step 4: Inspector (if enabled and more iterations)
            if self.inspector_agent and scores and iteration < max_iterations - 1:
                turbomq = scores.get('turbomq', 0) or 0
                
                if turbomq < 0.80:
                    print(f"\n📍 Step 4: Inspector analyzing (TurboMQ={turbomq:.4f} < 0.80)...")
                    # Could add inspector logic here for future improvement
                    pass
            
            best_clusters = original_clusters
            best_scores = scores
        
        # Compute final statistics
        stats = self._compute_stats(best_clusters)
        
        return {
            "model": self.model,
            "approach": "hierarchical_louvain_llm",
            "louvain_levels": self.louvain_levels,
            "num_super_nodes": num_super_nodes,
            "clusters": best_clusters,
            "statistics": stats,
            "metrics": best_scores,
            "iteration_history": iteration_history,
            "metadata": {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
    
    def _run_clustering(self, num_clusters) -> Optional[Dict[str, List[str]]]:
        """Run LLM clustering on super-nodes"""
        num_super_nodes = len(self.hierarchical_louvain.super_node_to_original)
        
        if num_clusters is None or num_clusters <= 0:
            cluster_goal = f"""Analyze the {num_super_nodes} super-nodes and decide the optimal number of clusters.
Consider the edge weights between super-nodes to group related ones together."""
        else:
            cluster_goal = f"""Create exactly {num_clusters} clusters from the {num_super_nodes} super-nodes.
Group super-nodes with high edge weights together."""
        
        message = f"""{cluster_goal}

Steps:
1. Call get_graph_info() to see overview
2. Call get_all_super_nodes() to see all super-nodes and their sizes
3. Call get_super_node_edges() to see which super-nodes are connected
4. Create clusters of super-nodes that minimize cross-cluster edges

Output format (simple text):
```
CLUSTER cluster_1:
super_0
super_2
super_5

CLUSTER cluster_2:
super_1
super_3
```

Start by calling the functions!"""
        
        try:
            if not self.verbose:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.user_proxy.initiate_chat(
                        self.clustering_agent,
                        message=message,
                        max_turns=15,
                    )
            else:
                self.user_proxy.initiate_chat(
                    self.clustering_agent,
                    message=message,
                    max_turns=15,
                )
            
            return self.extract_super_node_clusters(self.clustering_agent)
            
        except Exception as e:
            print(f"⚠ Clustering error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _fallback_clustering(self, num_clusters) -> Dict[str, List[str]]:
        """Fallback: group super-nodes based on edge weights"""
        super_nodes = list(self.hierarchical_louvain.super_node_to_original.keys())
        
        if num_clusters is None or num_clusters <= 0:
            # Default to roughly sqrt(n) clusters
            num_clusters = max(2, int(len(super_nodes) ** 0.5))
        
        # Simple greedy clustering based on edge weights
        # Start with each super-node in its own cluster
        clusters = {f"cluster_{i}": [sn] for i, sn in enumerate(super_nodes)}
        
        # If we have too many clusters, merge the smallest/least connected
        while len(clusters) > num_clusters:
            # Find smallest cluster
            smallest = min(clusters.keys(), key=lambda c: len(clusters[c]))
            smallest_nodes = clusters.pop(smallest)
            
            # Find best cluster to merge into (based on edge connections)
            best_target = None
            best_weight = -1
            
            for target_name, target_nodes in clusters.items():
                weight = 0
                for sn in smallest_nodes:
                    for edge in self.hierarchical_louvain.coarse_edges:
                        if edge["source"] == sn and edge["target"] in target_nodes:
                            weight += edge["weight"]
                        elif edge["target"] == sn and edge["source"] in target_nodes:
                            weight += edge["weight"]
                
                if weight > best_weight:
                    best_weight = weight
                    best_target = target_name
            
            if best_target:
                clusters[best_target].extend(smallest_nodes)
            else:
                # No connections, just merge with first cluster
                first_cluster = list(clusters.keys())[0]
                clusters[first_cluster].extend(smallest_nodes)
        
        # Renumber clusters
        return {f"cluster_{i}": nodes for i, (_, nodes) in enumerate(clusters.items())}
    
    def _validate_and_fix_clusters(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Ensure all original nodes are in exactly one cluster"""
        all_nodes = set(str(n) for n in self.graph.nodes())
        clustered = set()
        
        for nodes in clusters.values():
            clustered.update(nodes)
        
        # Find missing nodes
        missing = all_nodes - clustered
        if missing:
            print(f"   ⚠ {len(missing)} nodes missing, distributing by connections...")
            
            # Create node-to-cluster mapping
            node_to_cluster = {}
            for cname, nodes in clusters.items():
                for node in nodes:
                    node_to_cluster[node] = cname
            
            # Assign missing nodes based on neighbors
            for node in missing:
                cluster_votes = defaultdict(int)
                
                for neighbor in self.graph.successors(node):
                    if str(neighbor) in node_to_cluster:
                        cluster_votes[node_to_cluster[str(neighbor)]] += 1
                
                for neighbor in self.graph.predecessors(node):
                    if str(neighbor) in node_to_cluster:
                        cluster_votes[node_to_cluster[str(neighbor)]] += 1
                
                if cluster_votes:
                    best_cluster = max(cluster_votes, key=cluster_votes.get)
                else:
                    # No connections - assign to smallest cluster
                    best_cluster = min(clusters.keys(), key=lambda c: len(clusters[c]))
                
                clusters[best_cluster].append(node)
        
        # Remove non-existent nodes
        extra = clustered - all_nodes
        if extra:
            print(f"   ⚠ Removing {len(extra)} non-existent nodes")
            for cname in clusters:
                clusters[cname] = [n for n in clusters[cname] if n in all_nodes]
        
        return clusters
    
    def _evaluate_metrics(self, clusters):
        """Evaluate clustering metrics"""
        if not self.metrics_calculator:
            return None
        
        try:
            result = self.metrics_calculator.calculate_metrics(clusters, self.reference_rsf_path)
            turbomq = result.get('turbomq')
            mojo = result.get('mojo_fm')
            
            turbomq_str = f"{turbomq:.4f}" if turbomq is not None else "N/A"
            mojo_str = f"{mojo:.2f}" if mojo is not None else "N/A"
            print(f"✓ Metrics: TurboMQ={turbomq_str}, MoJo-FM={mojo_str}")
            
            return result
        except Exception as e:
            print(f"⚠ Metrics error: {e}")
            return None
    
    def _compute_stats(self, clusters):
        """Compute clustering statistics"""
        stats = {
            "num_clusters": len(clusters),
            "cluster_sizes": {name: len(nodes) for name, nodes in clusters.items()},
            "intra_cluster_edges": 0,
            "inter_cluster_edges": 0
        }
        
        node_to_cluster = {}
        for cname, nodes in clusters.items():
            for node in nodes:
                node_to_cluster[node] = cname
        
        for u, v in self.graph.edges():
            u_str, v_str = str(u), str(v)
            if u_str in node_to_cluster and v_str in node_to_cluster:
                if node_to_cluster[u_str] == node_to_cluster[v_str]:
                    stats["intra_cluster_edges"] += 1
                else:
                    stats["inter_cluster_edges"] += 1
        
        total = stats["intra_cluster_edges"] + stats["inter_cluster_edges"]
        if total > 0:
            stats["cohesion_score"] = stats["intra_cluster_edges"] / total
            stats["coupling_score"] = stats["inter_cluster_edges"] / total
        
        return stats
    
    def save_results(self, result, output_file):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Graph Clustering with Louvain Pre-processing"
    )
    parser.add_argument("graph_file", help="Path to pickled graph file")
    parser.add_argument("--dependency-rsf", help="Path to dependency RSF file (for TurboMQ)")
    parser.add_argument("--reference-rsf", help="Path to reference clustering RSF (for MoJo-FM)")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model name")
    parser.add_argument("--clusters", type=int, default=None, 
                       help="Number of final clusters (None = LLM decides)")
    parser.add_argument("--louvain-levels", type=int, default=2,
                       help="Number of Louvain coarsening levels (more = smaller graph)")
    parser.add_argument("--louvain-resolution", type=float, default=1.0,
                       help="Louvain resolution parameter (higher = more communities)")
    parser.add_argument("--iterations", type=int, default=1, help="Max improvement iterations")
    parser.add_argument("--output", default="hierarchical_clusters.json", help="Output file")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    analyzer = HierarchicalAnalyzer(
        ollama_host=args.host,
        model=args.model,
        dependency_rsf_path=args.dependency_rsf,
        reference_rsf_path=args.reference_rsf,
        verbose=args.verbose,
        louvain_levels=args.louvain_levels,
        louvain_resolution=args.louvain_resolution,
    )
    
    try:
        analyzer.load_graph(args.graph_file)
    except FileNotFoundError:
        print(f"❌ Graph file not found: {args.graph_file}")
        sys.exit(1)
    
    result = analyzer.analyze(
        num_clusters=args.clusters,
        max_iterations=args.iterations
    )
    
    if result:
        analyzer.save_results(result, args.output)
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Clusters: {result['statistics']['num_clusters']}")
        print(f"Super-nodes used: {result['num_super_nodes']}")
        
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"TurboMQ: {metrics.get('turbomq', 'N/A')}")
            print(f"MoJo-FM: {metrics.get('mojo_fm', 'N/A')}")
        
        # Show cluster size distribution
        sizes = list(result['statistics']['cluster_sizes'].values())
        print(f"\nCluster sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")
    else:
        print("❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
