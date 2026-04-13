
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

from louvain_hierarchical import LouvainGraph, LouvainAlgorithm, HierarchicalLouvain, compute_modularity

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
5. Output your clustering as JSON:
{
  "clusters": {
    "cluster_1": ["node1", "node2", ...],
    "cluster_2": ["node3", "node4", ...],
    ...
  }
}"""

INSPECTOR_SYSTEM_PROMPT = """You are the Lead Software Architect and Quality Inspector.
You are critical, strict, and detail-oriented.

Your goal: Analyze the clustering and identify specific improvements.

You have access to the graph tools:
- get_node_dependencies(node)
- get_node_dependents(node)
- get_all_edges()

Process:
1. Receive the proposed clusters and the TurboMQ/MoJo scores.
2. If the score is low (TurboMQ < 0.7), DO NOT just accept it.
3. INVESTIGATE:
   - Pick clusters that seem confused or too large.
   - Use get_node_dependencies to check edges between them.
   - Find specific nodes that are in the wrong place.
4. PROVIDE SPECIFIC ORDERS:
   - "Move node X to cluster Y"
   - "Merge cluster A and cluster B"
   - "Split cluster C"

Output JSON: {"specific_orders": ["..."], "analysis": "..."}"""

class GraphDataProvider:
    """Provides query functions for the graph"""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
    
    def get_graph_info(self) -> str:
        """Get basic graph statistics"""
        G = self.graph
        info = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "density": round(nx.density(G), 4),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "avg_in_degree": round(sum(d for _, d in G.in_degree()) / G.number_of_nodes(), 2) if G.number_of_nodes() > 0 else 0,
            "avg_out_degree": round(sum(d for _, d in G.out_degree()) / G.number_of_nodes(), 2) if G.number_of_nodes() > 0 else 0
        }
        return json.dumps(info, indent=2)
    
    def get_all_nodes(self) -> str:
        """Get list of all nodes"""
        return json.dumps(list(self.graph.nodes()))
    
    def get_all_edges(self) -> str:
        """Get all dependency edges"""
        edges = [{"source": str(u), "target": str(v)} for u, v in self.graph.edges()]
        return json.dumps(edges)
    
    def get_node_dependencies(self, node: str) -> str:
        """Get what a node depends on (successors)"""
        if node not in self.graph:
            return json.dumps({"error": f"Node '{node}' not found"})
        return json.dumps(list(self.graph.successors(node)))
    
    def get_node_dependents(self, node: str) -> str:
        """Get what depends on this node (predecessors)"""
        if node not in self.graph:
            return json.dumps({"error": f"Node '{node}' not found"})
        return json.dumps(list(self.graph.predecessors(node)))

class CoarseGraphProvider:
    """Provides query functions for the coarsened (super-node) graph"""
    
    def __init__(self, hierarchical_louvain: HierarchicalLouvain, original_graph: nx.DiGraph):
        self.hier = hierarchical_louvain
        self.original_graph = original_graph
        
        self.node_to_super = {}
        for super_node, original_nodes in self.hier.super_node_to_original.items():
            for node in original_nodes:
                self.node_to_super[node] = super_node
    
    def get_graph_info(self) -> str:
        """Get basic statistics about the super-node graph"""
        info = self.hier.get_coarse_graph_info()
        
        summary = {
            "total_nodes": info["num_super_nodes"],
            "total_edges": info["num_coarse_edges"],
            "original_nodes_total": sum(sn["size"] for sn in info["super_nodes"].values()),
            "average_super_node_size": round(
                sum(sn["size"] for sn in info["super_nodes"].values()) / info["num_super_nodes"], 1
            ) if info["num_super_nodes"] > 0 else 0
        }
        return json.dumps(summary, indent=2)
    
    def get_all_nodes(self) -> str:
        """Get list of all super-nodes"""
        super_nodes = []
        for super_node, original_nodes in self.hier.super_node_to_original.items():
            super_nodes.append({
                "name": super_node,
                "size": len(original_nodes),
                "members": original_nodes[:10]  # First 10 as sample
            })
        super_nodes.sort(key=lambda x: x["size"], reverse=True)
        return json.dumps(super_nodes, indent=2)
    
    def get_all_edges(self) -> str:
        """Get edges between super-nodes with weights"""
        edges = [e for e in self.hier.coarse_edges if e["source"] != e["target"]]
        edges.sort(key=lambda x: x["weight"], reverse=True)
        return json.dumps(edges, indent=2)
    
    def get_node_dependencies(self, node: str) -> str:
        """Get what a super-node depends on"""
        if node not in self.hier.super_node_to_original:
            return json.dumps({"error": f"Super-node '{node}' not found"})
        
        connected_to = defaultdict(float)
        for edge in self.hier.coarse_edges:
            if edge["source"] == node and edge["target"] != node:
                connected_to[edge["target"]] += edge["weight"]
        
        return json.dumps(dict(connected_to), indent=2)
    
    def get_node_dependents(self, node: str) -> str:
        """Get what depends on this super-node"""
        if node not in self.hier.super_node_to_original:
            return json.dumps({"error": f"Super-node '{node}' not found"})
        
        connected_from = defaultdict(float)
        for edge in self.hier.coarse_edges:
            if edge["target"] == node and edge["source"] != node:
                connected_from[edge["source"]] += edge["weight"]
        
        return json.dumps(dict(connected_from), indent=2)

class MetricsCalculator:
    """Calculates TurboMQ and MoJo-FM metrics"""
    
    def __init__(self, dependency_rsf_path, experiments_dir=None):
        self.dependency_rsf_path = os.path.abspath(dependency_rsf_path) if dependency_rsf_path else None
        self.experiments_dir = experiments_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "experiments"
        )
    
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
            return None
        
        if not os.path.exists(self.dependency_rsf_path):
            return None
        
        if not os.path.exists(clustering_rsf_path):
            return None
        
        try:
            dep_rsf_temp = os.path.join(self.experiments_dir, "temp_dependency.rsf")
            clust_rsf_temp = os.path.join(self.experiments_dir, "temp_clustering.rsf")
            
            shutil.copy2(self.dependency_rsf_path, dep_rsf_temp)
            shutil.copy2(clustering_rsf_path, clust_rsf_temp)
            
            cmd = ["java", "-jar", "turbomq.jar", "temp_dependency.rsf", "temp_clustering.rsf"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir
            )
            
            for f in [dep_rsf_temp, clust_rsf_temp]:
                if os.path.exists(f):
                    os.remove(f)
            
            if result.returncode == 0:
                try:
                    return float(result.stdout.strip())
                except ValueError:
                    return None
            return None
        except Exception as e:
            return None
    
    def calculate_mojo_fm(self, clustering_rsf_path, reference_rsf_path=None):
        """Calculate MoJo-FM distance using Java jar"""
        mojo_jar = os.path.join(self.experiments_dir, "mojo.jar")
        
        if not os.path.exists(mojo_jar):
            return None
        
        if not reference_rsf_path or not os.path.exists(reference_rsf_path):
            return None
        
        try:
            clust_rsf_abs = os.path.abspath(clustering_rsf_path)
            ref_rsf_abs = os.path.abspath(reference_rsf_path)
            
            cmd = ["java", "-jar", "mojo.jar", clust_rsf_abs, ref_rsf_abs, "-fm"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir
            )
            
            if result.returncode == 0:
                match = re.search(r'[\d.]+', result.stdout.strip())
                if match:
                    return float(match.group())
            return None
        except Exception:
            return None
    
    def calculate_metrics(self, clusters, reference_rsf_path=None):
        """Calculate all metrics for given clusters"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rsf', delete=False) as f:
            temp_rsf = f.name
        
        try:
            self.clusters_to_rsf(clusters, temp_rsf)
            turbomq = self.calculate_turbomq(temp_rsf)
            mojo_fm = self.calculate_mojo_fm(temp_rsf, reference_rsf_path)
            
            return {
                "turbomq": turbomq,
                "mojo_fm": mojo_fm
            }
        except Exception as e:
            return {"turbomq": None, "mojo_fm": None}

class HierarchicalAnalyzer:
    """
    Graph analyzer with Louvain pre-processing for LLM-friendly clustering.
    Uses prompts matching the paper listings.
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
        self.graph_provider: Optional[CoarseGraphProvider] = None
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
        
        self.hierarchical_louvain = HierarchicalLouvain(
            self.graph,
            num_coarsening_levels=self.louvain_levels,
            resolution=self.louvain_resolution,
            random_seed=42
        )
        self.hierarchical_louvain.coarsen()
        
        self.graph_provider = CoarseGraphProvider(self.hierarchical_louvain, self.graph)
        
        if self.dependency_rsf_path:
            self.metrics_calculator = MetricsCalculator(self.dependency_rsf_path)
    
    def setup_clustering_agent(self, num_clusters):
        """Setup the clustering agent"""
        if num_clusters is None or num_clusters <= 0:
            cluster_goal = "Decide the optimal number of clusters based on the graph structure."
        else:
            cluster_goal = f"Create exactly {num_clusters} clusters from the graph."
        
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
            if "clusters" in content.lower() and "{" in content:
                return True
            return False
        
        self.user_proxy = UserProxyAgent(
            name="GraphProvider",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=15,
            code_execution_config=False,
            default_auto_reply="Please continue and output the final clustering as JSON.",
            is_termination_msg=termination_check,
            silent=not self.verbose,
        )
        
        self._register_graph_functions(self.clustering_agent, self.user_proxy)
    
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
            max_consecutive_auto_reply=5,
            code_execution_config=False,
            is_termination_msg=lambda msg: "specific_orders" in msg.get("content", "").lower(),
            silent=not self.verbose,
        )
        
        self._register_graph_functions(self.inspector_agent, self.inspector_proxy)
    
    def _register_graph_functions(self, agent, executor):
        """Register functions for querying the graph"""
        provider = self.graph_provider
        
        def get_graph_info() -> str:
            return provider.get_graph_info()
        
        def get_all_nodes() -> str:
            return provider.get_all_nodes()
        
        def get_all_edges() -> str:
            return provider.get_all_edges()
        
        def get_node_dependencies(node: str) -> str:
            return provider.get_node_dependencies(node)
        
        def get_node_dependents(node: str) -> str:
            return provider.get_node_dependents(node)
        
        functions = [
            (get_graph_info, "get_graph_info", "Get basic graph statistics"),
            (get_all_nodes, "get_all_nodes", "Get list of all nodes"),
            (get_all_edges, "get_all_edges", "Get all dependency edges"),
            (get_node_dependencies, "get_node_dependencies", "Get what a node depends on"),
            (get_node_dependents, "get_node_dependents", "Get what depends on a node"),
        ]
        
        for func, name, desc in functions:
            register_function(func, caller=agent, executor=executor, name=name, description=desc)
    
    def extract_clusters(self, agent) -> Optional[Dict[str, List[str]]]:
        """Extract clustering from agent conversation"""
        try:
            for conv_id, messages in agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if '"clusters"' not in content:
                            continue
                        
                        # Find balanced JSON by tracking braces
                        start = content.find('{')
                        while start != -1:
                            depth = 0
                            for i in range(start, len(content)):
                                if content[i] == '{':
                                    depth += 1
                                elif content[i] == '}':
                                    depth -= 1
                                    if depth == 0:
                                        candidate = content[start:i+1]
                                        try:
                                            data = json.loads(candidate)
                                            if isinstance(data, dict) and 'clusters' in data:
                                                return data['clusters']
                                        except json.JSONDecodeError:
                                            pass
                                        break
                            start = content.find('{', start + 1)
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Extraction error: {e}")
        return None
    
    def analyze(self, num_clusters=None, max_iterations=1):
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
        
        self.setup_clustering_agent(num_clusters)
        if self.metrics_calculator:
            self.setup_inspector_agent()
        
        best_clusters = None
        best_scores = None
        last_analysis = None
        iteration_history = []
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            
            print(f"\n📍 Step 1: Clustering {num_super_nodes} super-nodes...")
            
            if iteration == 0:
                clusters = self._run_initial_clustering(num_clusters)
            else:
                clusters = self._run_clustering_with_feedback(
                    num_clusters, 
                    best_scores, 
                    best_clusters, 
                    last_analysis
                )
            
            if not clusters:
                print("❌ LLM clustering failed. Only LLM results are accepted (no Louvain fallback).")
                return None
            print(f"✓ LLM created {len(clusters)} clusters")
            
            print(f"\n📍 Step 2: Expanding super-nodes to original nodes...")
            original_clusters = self._expand_clusters(clusters)
            
            original_clusters = self._validate_and_fix_clusters(original_clusters)
            
            total_nodes = sum(len(nodes) for nodes in original_clusters.values())
            print(f"✓ Expanded to {total_nodes} original nodes in {len(original_clusters)} clusters")
            
            print(f"\n📍 Step 3: Calculating metrics...")
            scores = self._evaluate_metrics(original_clusters)
            
            iteration_history.append({
                "iteration": iteration + 1,
                "num_clusters": len(original_clusters),
                "turbomq": scores.get('turbomq') if scores else None,
                "mojo_fm": scores.get('mojo_fm') if scores else None,
            })
            
            if self.inspector_agent and scores and iteration < max_iterations - 1:
                turbomq = scores.get('turbomq', 0) or 0
                
                if turbomq < 0.85:
                    print(f"\n📍 Step 4: Inspector analyzing (TurboMQ={turbomq:.4f} < 0.85)...")
                    analysis = self._run_inspector(scores, original_clusters)
                    
                    if analysis:
                        print(f"✓ Inspector provided feedback")
                        last_analysis = analysis
                        best_clusters = original_clusters
                        best_scores = scores
                        
                        self.clustering_agent.reset()
                        continue
                    else:
                        print("⚠ Inspector analysis failed")
                else:
                    print(f"✓ TurboMQ={turbomq:.4f} meets threshold, stopping")
            
            best_clusters = original_clusters
            best_scores = scores
            break
        
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
    
    def _run_initial_clustering(self, num_clusters) -> Optional[Dict[str, List[str]]]:
        """
        Run initial clustering (Listing 1 prompt style)
        """
        total_nodes = len(self.hierarchical_louvain.super_node_to_original)
        
        if num_clusters is None or num_clusters <= 0:
            cluster_instruction = f"Analyze the graph and decide the optimal number of clusters."
        else:
            cluster_instruction = f"Create exactly {num_clusters} clusters."
        
        message = f"""Please analyze the dependency graph and create microservice clusters.

{cluster_instruction}

Steps:
1. Use get_graph_info() to understand the graph structure
2. Use get_all_nodes() to see all {total_nodes} nodes
3. Use get_all_edges() to see all dependencies
4. Analyze the dependency patterns
5. Create clusters that minimize inter-cluster dependencies

Output your clustering as JSON:
{{
  "clusters": {{
    "cluster_1": ["node1", "node2", ...],
    "cluster_2": ["node3", "node4", ...],
    ...
  }}
}}

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
            
            return self.extract_clusters(self.clustering_agent)
            
        except Exception as e:
            print(f"⚠ Clustering error: {e}")
            return None
    
    def _run_inspector(self, scores, clusters) -> Optional[str]:
        """
        Run inspector analysis (Listing 2 prompt style)
        """
        clusters_str = json.dumps(clusters, indent=2)
        turbomq = scores.get('turbomq', 'N/A')
        mojo_fm = scores.get('mojo_fm', 'N/A')
        message = f"""Perform a quality inspection on this clustering.

Scores:
- TurboMQ: {turbomq} (Target > 0.7)
- MoJo-FM: {mojo_fm}

Proposed Clusters:
{clusters_str}

INSTRUCTIONS:
1. If TurboMQ is low, use `get_node_dependencies` to check edges between clusters.
2. Identify nodes that are "misplaced" (heavily coupled to a different cluster).
3. Provide a list of specific ORDERS to fix the problems.

Output JSON: {{"specific_orders": ["..."], "analysis": "..."}}"""
        
        try:
            if not self.verbose:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.inspector_proxy.initiate_chat(
                        self.inspector_agent,
                        message=message,
                        max_turns=5,
                    )
            else:
                self.inspector_proxy.initiate_chat(
                    self.inspector_agent,
                    message=message,
                    max_turns=5,
                )
            
            for conv_id, messages in self.inspector_agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if "specific_orders" in content.lower() or "move" in content.lower():
                            return content
            return None
            
        except Exception as e:
            print(f"⚠ Inspector error: {e}")
            return None
    
    def _run_clustering_with_feedback(self, num_clusters, previous_scores, 
                                       current_clusters, analysis_feedback) -> Optional[Dict[str, List[str]]]:
        """
        Run clustering with inspector feedback (Listing 3 prompt style)
        """
        clusters_str = json.dumps(current_clusters, indent=2)
        turbomq = previous_scores.get('turbomq', 'N/A') if previous_scores else 'N/A'
        mojo_fm = previous_scores.get('mojo_fm', 'N/A') if previous_scores else 'N/A'
        
        feedback_text = ""
        if analysis_feedback:
            feedback_text = f"\n🚨 INSPECTOR ORDERS & FEEDBACK:\n{analysis_feedback}\n"
        
        message = f"""You must improve the clustering based on the Inspector's feedback.

Previous Metrics:
- TurboMQ: {turbomq} (Target > 0.7)
- MoJo-FM: {mojo_fm} (Target -> Lower is better)

{feedback_text}

Here is your previous attempt:
{clusters_str}

CRITICAL INSTRUCTION: 
1. Read the Inspector's orders carefully.
2. Move nodes as requested to fix dependencies.
3. Do NOT return the exact same JSON.
4. Output the complete, improved JSON.

{{
  "clusters": {{
    "cluster_1": ["node1", "node2", ...],
    ...
  }}
}}"""
        
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
            
            return self.extract_clusters(self.clustering_agent)
            
        except Exception as e:
            print(f"⚠ Clustering with feedback error: {e}")
            return None
    
    def _expand_clusters(self, super_node_clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Expand super-node clustering back to original nodes"""
        return self.hierarchical_louvain.expand_clustering(super_node_clusters)
    
    def _validate_and_fix_clusters(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Ensure all original nodes are in exactly one cluster"""
        all_nodes = set(str(n) for n in self.graph.nodes())
        clustered = set()
        
        for nodes in clusters.values():
            clustered.update(nodes)
        
        missing = all_nodes - clustered
        if missing:
            print(f"   ⚠ {len(missing)} nodes missing, distributing by connections...")
            
            node_to_cluster = {}
            for cname, nodes in clusters.items():
                for node in nodes:
                    node_to_cluster[node] = cname
            
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
                    best_cluster = min(clusters.keys(), key=lambda c: len(clusters[c]))
                
                clusters[best_cluster].append(node)
        
        extra = clustered - all_nodes
        if extra:
            print(f"Removing {len(extra)} non-existent nodes")
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
        
        sizes = list(result['statistics']['cluster_sizes'].values())
        print(f"\nCluster sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")
    else:
        print("❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()