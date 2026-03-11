#!/usr/bin/env python3
"""
Interactive AutoGen Graph Analyzer with Metrics Agent
Pipeline: Cluster → Evaluate (TurboMQ + MoJo-FM) → Analyze (Inspector) → Improve → Done
"""

import sys
import os
import pickle
import json
import time
import argparse
import contextlib
import io

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, register_function
except ImportError:
    sys.exit(1)


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.graph_provider import GraphDataProvider
from utils.metrics_calculator import MetricsCalculator
from prompts.prompts import CLUSTERING_SYSTEM_PROMPT, INSPECTOR_SYSTEM_PROMPT



class InteractiveAnalyzerWithMetrics:
    """Analyzer with integrated metrics evaluation and improvement loop"""
    
    def __init__(self, ollama_host="http://localhost:11434", model="gpt-oss:120b-cloud",
                 dependency_rsf_path=None, reference_rsf_path=None):
        self.ollama_host = ollama_host
        self.model = model
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
        
        
        if self.dependency_rsf_path:
            self.metrics_calculator = MetricsCalculator(self.dependency_rsf_path)

    def setup_clustering_agent(self, num_clusters):
        """Setup clustering agent"""
        system_prompt = CLUSTERING_SYSTEM_PROMPT + f"\n\nYour goal: Create {num_clusters} clusters from the graph."
        
        config_list = [{
            "model": self.model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": "ollama",
            "api_type": "openai",
            "temperature": 0,
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
        
        self.user_proxy = UserProxyAgent(
            name="GraphProvider",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            default_auto_reply="",
            is_termination_msg=lambda x: "clusters" in x.get("content", "").lower() and "{" in x.get("content", ""),
            silent=True,
        )
        
       
        self._register_graph_functions(self.clustering_agent, self.user_proxy)
    
    def setup_metrics_agent(self):
        """Setup metrics evaluation agent (The Inspector)"""
        config_list = [{
            "model": self.model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": "ollama",
            "api_type": "openai",
            "temperature": 0,
        }]
        
        
        self.metrics_agent = AssistantAgent(
            name="InspectorArchitecture",
            system_message=INSPECTOR_SYSTEM_PROMPT,
            llm_config={
                "config_list": config_list,
                "timeout": 300,
                "cache_seed": None,
            }
        )
        
        self.metrics_proxy = UserProxyAgent(
            name="InspectorExecutor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10, 
            code_execution_config=False,
            silent=True,
        )
        
        
        self._register_metrics_functions()
        self._register_graph_functions(self.metrics_agent, self.metrics_proxy)
    
    def _register_graph_functions(self, agent, executor):
        """Register graph query functions for a specific agent"""
        provider = self.graph_provider
        
       
        def get_graph_info(): return provider.get_graph_info()
        def get_all_nodes(): return provider.get_all_nodes()
        def get_all_edges(): return provider.get_all_edges()
        def get_node_dependencies(node: str): return provider.get_node_dependencies(node)
        def get_node_dependents(node: str): return provider.get_node_dependents(node)
        def get_hub_nodes(top_n: int = 10): return provider.get_hub_nodes(top_n)
        
        functions = [
            (get_graph_info, "get_graph_info", "Get basic graph statistics"),
            (get_all_nodes, "get_all_nodes", "Get list of all nodes"),
            (get_all_edges, "get_all_edges", "Get all dependency edges"),
            (get_node_dependencies, "get_node_dependencies", "Get what a node depends on"),
            (get_node_dependents, "get_node_dependents", "Get what depends on a node"),
            (get_hub_nodes, "get_hub_nodes", "Get most depended-on nodes"),
        ]

        for func, name, desc in functions:
            register_function(
                func,
                caller=agent,
                executor=executor,
                name=name,
                description=desc
            )
    
    def _register_metrics_functions(self):
        """Register metrics calculation functions"""
        calculator = self.metrics_calculator
        if not calculator: return
        
        def calculate_turbomq(clusters_json: str):
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
        print(f"INTERACTIVE ANALYSIS WITH METRICS (INSPECTOR MODE)")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Target clusters: {num_clusters}")
        print(f"Max iterations: {max_iterations}")
        
        # Setup agents
        self.setup_clustering_agent(num_clusters)
        if self.metrics_calculator:
            self.setup_metrics_agent()
        
        best_clusters = None
        best_scores = None
        last_analysis = None  
        iteration_history = []
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            
            # Step 1: Cluster (Passing feedback if we have it)
            print(f"\n Step 1: Clustering...")
            clusters = self._run_clustering(
                num_clusters, 
                iteration > 0, 
                best_scores, 
                best_clusters, 
                analysis_feedback=last_analysis
            )
            
            if not clusters:
                print(" Clustering failed, using fallback")
                clusters = self._fallback_clustering()
            
            # Step 2: Evaluate
            print(f"\n Step 2: Evaluating metrics...")
            scores = self._evaluate_metrics(clusters)
            
            iteration_history.append({
                "iteration": iteration + 1,
                "num_clusters": len(clusters),
                "turbomq": scores.get('turbomq') if scores else None,
                "mojo_fm": scores.get('mojo_fm') if scores else None
            })
            
            # Step 3: Inspector Analysis
            if self.metrics_agent and scores:
                print(f"\n Step 3: Inspector is investigating...")
                analysis = self._analyze_scores(scores, clusters)
                
                if iteration < max_iterations - 1 and analysis:
                    print(f"\n Step 4: Preparing for improvement...")
                    best_clusters = clusters
                    best_scores = scores
                    last_analysis = analysis
                    self.clustering_agent.reset()
                    continue
                else:
                    best_clusters = clusters
                    best_scores = scores
                    break
            else:
                best_clusters = clusters
                best_scores = scores
                break
        
        # Final results
        stats = self._compute_stats(best_clusters)
        
        result = {
            "model": self.model,
            "approach": "interactive_with_metrics",
            "clusters": best_clusters,
            "statistics": stats,
            "metrics": best_scores,
            "iteration_history": iteration_history,
            "metadata": {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "iterations": iteration + 1,
            }
        }
        
        return result
    

    def _run_clustering(self, num_clusters, is_improvement, previous_scores, current_clusters=None, analysis_feedback=None):
        """Run clustering step"""
        if is_improvement and previous_scores and current_clusters:
            clusters_str = json.dumps(current_clusters, indent=2)
            
            # Format the Inspector's feedback
            feedback_text = ""
            if analysis_feedback:
                feedback_text = f"\n🚨 INSPECTOR ORDERS & FEEDBACK:\n{analysis_feedback}\n"
            
            message = f"""You must improve the clustering based on the Inspector's feedback.

Previous Metrics:
- TurboMQ: {previous_scores.get('turbomq', 'N/A')} (Target > 0.7)
- MoJo-FM: {previous_scores.get('mojo_fm', 'N/A')} (Target -> Lower is better)

{feedback_text}

Here is your previous attempt:
{clusters_str}

CRITICAL INSTRUCTION: 
1. Read the Inspector's orders carefully.
2. Move nodes as requested to fix dependencies.
3. Do NOT return the exact same JSON.
4. Output the complete, improved JSON."""
        else:
            message = f"""Please analyze the dependency graph and create {num_clusters} microservice clusters.

Steps:
1. Use get_graph_info() to understand the graph structure
2. Use get_all_edges() to see all dependencies
3. Analyze the dependency patterns
4. Create clusters that minimize inter-cluster dependencies

Output your clustering as JSON:
{{
  "clusters": {{
    "cluster_1": ["node1", "node2", ...],
    "cluster_2": ["node3", "node4", ...],
    ...
  }}
}}"""
        
        try:
            # Suppress verbose output
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.user_proxy.initiate_chat(
                    self.clustering_agent,
                    message=message,
                    max_turns=15,
                )
            return self.extract_clusters(self.clustering_agent)
        except Exception as e:
            print(f" Clustering error: {e}")
            return None
    
    def _evaluate_metrics(self, clusters):
        if not self.metrics_calculator: return None
        try:
            result = self.metrics_calculator.calculate_metrics(clusters, self.reference_rsf_path)
            print(f"✓ Metrics: TurboMQ={result.get('turbomq'):.4f}, MoJo={result.get('mojo_fm')}")
            return result
        except Exception as e:
            print(f" Metrics evaluation error: {e}")
            return None
    
    def _analyze_scores(self, scores, clusters):
        """Have Inspector agent analyze the scores using graph tools"""
        if not self.metrics_agent: return None
        
        # We must give the Inspector the CLUSTERS so it knows what to investigate
        clusters_str = json.dumps(clusters, indent=2)
        
        message = f"""Perform a quality inspection on this clustering.

Scores:
- TurboMQ: {scores.get('turbomq', 'N/A')} (Target > 0.7)
- MoJo-FM: {scores.get('mojo_fm', 'N/A')}

Proposed Clusters:
{clusters_str}

INSTRUCTIONS:
1. If TurboMQ is low, use `get_node_dependencies` to check edges between clusters.
2. Identify nodes that are "misplaced" (heavily coupled to a different cluster).
3. Provide a list of specific ORDERS to fix the problems.

Output JSON: {{"specific_orders": ["..."], "analysis": "..."}}"""
        
        try:
            # Allow more turns so Inspector can call tools multiple times
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.metrics_proxy.initiate_chat(
                    self.metrics_agent,
                    message=message,
                    max_turns=10, 
                )
            
            for conv_id, messages in self.metrics_agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        # Try to parse the specific orders if JSON
                        content = msg.get('content', '')
                        if "specific_orders" in content:
                            return content
                        # If simple text, return it all
                        return content
        except Exception as e:
            print(f" Analysis error: {e}")
            return None
    
    def _fallback_clustering(self):
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
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive analyzer with Inspector Agent")
    parser.add_argument("graph_file", help="Path to pickled graph file")
    parser.add_argument("--dependency-rsf", help="Path to dependency RSF file (for metrics)")
    parser.add_argument("--reference-rsf", help="Path to reference clustering RSF (for MoJo-FM)")
    parser.add_argument("--model", default="gpt-oss:120b-cloud", help="Ollama model")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--iterations", type=int, default=1, help="Max improvement iterations")
    parser.add_argument("--output", default="clusters_with_metrics.json", help="Output file")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")

    args = parser.parse_args()

    analyzer = InteractiveAnalyzerWithMetrics(
        ollama_host=args.host,
        model=args.model,
        dependency_rsf_path=args.dependency_rsf,
        reference_rsf_path=args.reference_rsf
    )

    try:
        analyzer.load_graph(args.graph_file)
    except FileNotFoundError:
        print(f"❌ Graph file not found: {args.graph_file}")
        sys.exit(1)

    result = analyzer.analyze_with_metrics(
        num_clusters=args.clusters,
        max_iterations=args.iterations
    )

    if result:
        analyzer.save_results(result, args.output)
        
        # Display Final Summary
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Clusters: {result['statistics']['num_clusters']}")
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"TurboMQ: {metrics.get('turbomq', 'N/A')}")
            print(f"MoJo-FM: {metrics.get('mojo_fm', 'N/A')}")
            
        if len(result.get('iteration_history', [])) > 1:
            print(f"\nImprovements:")
            first = result['iteration_history'][0]
            last = result['iteration_history'][-1]
            if first['turbomq'] and last['turbomq']:
                imp = last['turbomq'] - first['turbomq']
                print(f"TurboMQ Change: {imp:+.4f}")
    else:
        print("❌ Analysis failed")
        sys.exit(1)

if __name__ == "__main__":
    main()