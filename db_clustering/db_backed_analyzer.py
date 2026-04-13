#!/usr/bin/env python3
"""
DB-Backed LLM Clustering Agent

Uses SQLite for state management instead of loading entire graphs into LLM context.
Calculates TurboMQ and MoJo-FM metrics using Java jars (like hierarchical_analyzer).
No fallback methods — only LLM clustering results are accepted.
"""

import sys
import os
import json
import argparse
import contextlib
import io
import re
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Any

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, register_function
except ImportError:
    print("❌ Error: AutoGen not installed")
    print("Install with: pip install pyautogen")
    sys.exit(1)

try:
    from .clustering_database import ClusteringDatabase
except ImportError:
    from clustering_database import ClusteringDatabase


CLUSTERING_SYSTEM_PROMPT = """You are an expert in microservice architecture and dependency analysis.

You have access to a graph stored in a DATABASE. Instead of seeing all nodes/edges at once,
you query the database for specific information. This is much more efficient!

AVAILABLE FUNCTIONS (READ - get information):
- get_graph_summary() - Get high-level stats (node count, edge count, cohesion)
- get_cluster_list() - Get all clusters with their sizes
- get_cluster_info(cluster_id) - Get details about a specific cluster
- get_node_info(node_id) - Get details about a specific node
- get_cross_cluster_edges(cluster_a, cluster_b) - Get edge count between two clusters
- get_problematic_nodes() - Find nodes that might be misplaced
- get_cluster_connections() - See which clusters are connected
- get_suggested_moves() - Get AI-suggested improvements

AVAILABLE FUNCTIONS (WRITE - make changes):
- move_node(node_id, to_cluster) - Move a node to a different cluster
- merge_clusters(cluster_a, cluster_b) - Merge two clusters
- create_cluster(cluster_id, node_ids) - Create a new cluster with nodes

WORKFLOW:
1. Call get_graph_summary() to understand the current state
2. Call get_cluster_list() to see existing clusters
3. Call get_problematic_nodes() to find issues
4. Call get_suggested_moves() for improvement ideas
5. Use move_node() to make improvements
6. Repeat until satisfied

When you're done improving, say "CLUSTERING_COMPLETE" and provide a summary."""


INSPECTOR_SYSTEM_PROMPT = """You are the Lead Software Architect and Quality Inspector.

You review clustering decisions and suggest improvements.

AVAILABLE FUNCTIONS:
- get_graph_summary() - See overall metrics
- get_cluster_info(cluster_id) - Inspect a cluster
- get_cross_cluster_edges(cluster_a, cluster_b) - Check coupling between clusters
- get_problematic_nodes() - Find misplaced nodes
- get_cluster_connections() - See inter-cluster dependencies

Your job:
1. Analyze current clustering quality
2. Identify specific problems
3. Provide actionable orders

Output format:
{
  "analysis": "Your analysis...",
  "problems": ["problem 1", "problem 2"],
  "specific_orders": ["Move node X to cluster Y", "Merge cluster A and B"]
}"""


class MetricsCalculator:
    """Calculates TurboMQ and MoJo-FM metrics using Java jars"""

    def __init__(self, dependency_rsf_path, experiments_dir=None):
        self.dependency_rsf_path = os.path.abspath(dependency_rsf_path) if dependency_rsf_path else None
        self.experiments_dir = experiments_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"
        )

    def clusters_to_rsf(self, clusters, output_path):
        with open(output_path, 'w') as f:
            for cluster_name, nodes in clusters.items():
                for node in nodes:
                    f.write(f"contain {cluster_name} {node}\n")

    def calculate_turbomq(self, clustering_rsf_path):
        turbomq_jar = os.path.join(self.experiments_dir, "turbomq.jar")
        if not os.path.exists(turbomq_jar) or not self.dependency_rsf_path:
            return None

        try:
            dep_tmp = os.path.join(self.experiments_dir, "temp_dependency.rsf")
            clust_tmp = os.path.join(self.experiments_dir, "temp_clustering.rsf")
            shutil.copy2(self.dependency_rsf_path, dep_tmp)
            shutil.copy2(clustering_rsf_path, clust_tmp)

            result = subprocess.run(
                ["java", "-jar", "turbomq.jar", "temp_dependency.rsf", "temp_clustering.rsf"],
                capture_output=True, text=True, timeout=60, cwd=self.experiments_dir
            )

            for f in [dep_tmp, clust_tmp]:
                if os.path.exists(f):
                    os.remove(f)

            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            print(f"⚠ TurboMQ error: {e}")
        return None

    def calculate_mojo_fm(self, clustering_rsf_path, reference_rsf_path):
        mojo_jar = os.path.join(self.experiments_dir, "mojo.jar")
        if not os.path.exists(mojo_jar) or not reference_rsf_path:
            return None

        try:
            result = subprocess.run(
                ["java", "-jar", "mojo.jar",
                 os.path.abspath(clustering_rsf_path),
                 os.path.abspath(reference_rsf_path), "-fm"],
                capture_output=True, text=True, timeout=60, cwd=self.experiments_dir
            )
            if result.returncode == 0:
                match = re.search(r'[\d.]+', result.stdout.strip())
                if match:
                    return float(match.group())
        except Exception as e:
            print(f"⚠ MoJo-FM error: {e}")
        return None

    def calculate_all(self, clusters, reference_rsf_path=None):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rsf', delete=False) as f:
            temp_rsf = f.name

        try:
            self.clusters_to_rsf(clusters, temp_rsf)
            turbomq = self.calculate_turbomq(temp_rsf)
            mojo_fm = self.calculate_mojo_fm(temp_rsf, reference_rsf_path) if reference_rsf_path else None

            print(f"   TurboMQ: {turbomq}")
            print(f"   MoJo-FM: {mojo_fm}")

            return {"turbomq": turbomq, "mojo_fm": mojo_fm, "clustering_rsf": temp_rsf}
        except Exception as e:
            print(f"⚠ Metrics error: {e}")
            return {"turbomq": None, "mojo_fm": None}


class DBBackedAnalyzer:
    """
    Clustering analyzer that uses database for state management.
    Calculates TurboMQ and MoJo-FM scores. No fallback — LLM only.
    """

    def __init__(self,
                 db_path: str = ":memory:",
                 ollama_host: str = "http://localhost:11434",
                 model: str = "llama3:8b",
                 dependency_rsf_path: str = None,
                 reference_rsf_path: str = None,
                 verbose: bool = False):
        self.db_path = db_path
        self.ollama_host = ollama_host
        self.model = model
        self.dependency_rsf_path = dependency_rsf_path
        self.reference_rsf_path = reference_rsf_path
        self.verbose = verbose

        self.db: Optional[ClusteringDatabase] = None
        self.metrics_calculator: Optional[MetricsCalculator] = None

        self.clustering_agent = None
        self.user_proxy = None
        self.inspector_agent = None
        self.inspector_proxy = None

    def initialize_database(self, graph_pickle_path: str,
                           initial_clustering: Optional[Dict[str, List[str]]] = None):
        self.db = ClusteringDatabase(self.db_path)
        graph = self.db.load_from_pickle(graph_pickle_path, initial_clustering)

        if self.dependency_rsf_path:
            self.metrics_calculator = MetricsCalculator(self.dependency_rsf_path)

        return graph

    def initialize_from_louvain(self, graph_pickle_path: str,
                                resolution: float = 1.0):
        import pickle
        import networkx as nx

        with open(graph_pickle_path, 'rb') as f:
            graph = pickle.load(f)

        try:
            from networkx.algorithms.community import louvain_communities
            G_undirected = graph.to_undirected()
            communities = louvain_communities(G_undirected, resolution=resolution, seed=42)

            initial_clustering = {}
            for i, community in enumerate(communities):
                initial_clustering[f"cluster_{i}"] = [str(n) for n in community]

            print(f"✓ Louvain created {len(initial_clustering)} initial clusters")
        except ImportError:
            print("⚠ Louvain not available, starting with no clustering")
            initial_clustering = None

        self.db = ClusteringDatabase(self.db_path)
        self.db.load_from_networkx(graph, initial_clustering)

        if self.dependency_rsf_path:
            self.metrics_calculator = MetricsCalculator(self.dependency_rsf_path)

        return graph

    def setup_agents(self):
        config_list = [{
            "model": self.model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": "ollama",
            "api_type": "openai",
            "temperature": 0,
        }]

        self.clustering_agent = AssistantAgent(
            name="ClusteringExpert",
            system_message=CLUSTERING_SYSTEM_PROMPT,
            llm_config={"config_list": config_list, "timeout": 600, "cache_seed": None}
        )

        self.user_proxy = UserProxyAgent(
            name="DBExecutor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=30,
            code_execution_config=False,
            default_auto_reply="Continue improving the clustering.",
            is_termination_msg=lambda msg: "CLUSTERING_COMPLETE" in msg.get("content", ""),
            silent=not self.verbose,
        )

        self._register_db_functions(self.clustering_agent, self.user_proxy)

        self.inspector_agent = AssistantAgent(
            name="Inspector",
            system_message=INSPECTOR_SYSTEM_PROMPT,
            llm_config={"config_list": config_list, "timeout": 300, "cache_seed": None}
        )

        self.inspector_proxy = UserProxyAgent(
            name="InspectorExecutor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            is_termination_msg=lambda msg: "specific_orders" in msg.get("content", "").lower(),
            silent=not self.verbose,
        )

        self._register_db_functions(self.inspector_agent, self.inspector_proxy, read_only=True)

    def _register_db_functions(self, agent, executor, read_only: bool = False):
        db = self.db

        def get_graph_summary() -> str:
            """Get high-level graph statistics"""
            return db.get_graph_summary()

        def get_cluster_list() -> str:
            """Get list of all clusters with sizes"""
            return db.get_cluster_list()

        def get_cluster_info(cluster_id: str) -> str:
            """Get detailed info about a cluster"""
            return db.get_cluster_info(cluster_id)

        def get_node_info(node_id: str) -> str:
            """Get info about a specific node"""
            return db.get_node_info(node_id)

        def get_cross_cluster_edges(cluster_a: str, cluster_b: str) -> str:
            """Get edge count between two clusters"""
            return db.get_cross_cluster_edges(cluster_a, cluster_b)

        def get_problematic_nodes(limit: int = 10) -> str:
            """Find nodes that might be misplaced"""
            return db.get_problematic_nodes(limit)

        def get_cluster_connections() -> str:
            """Get connections between clusters"""
            return db.get_cluster_connections()

        def get_suggested_moves(limit: int = 5) -> str:
            """Get suggested node moves for improvement"""
            return db.get_suggested_moves(limit)

        read_functions = [
            (get_graph_summary, "get_graph_summary", "Get high-level graph statistics"),
            (get_cluster_list, "get_cluster_list", "Get list of all clusters"),
            (get_cluster_info, "get_cluster_info", "Get details about a specific cluster"),
            (get_node_info, "get_node_info", "Get details about a specific node"),
            (get_cross_cluster_edges, "get_cross_cluster_edges", "Get edge count between two clusters"),
            (get_problematic_nodes, "get_problematic_nodes", "Find potentially misplaced nodes"),
            (get_cluster_connections, "get_cluster_connections", "Get inter-cluster edge counts"),
            (get_suggested_moves, "get_suggested_moves", "Get AI-suggested improvements"),
        ]

        for func, name, desc in read_functions:
            register_function(func, caller=agent, executor=executor, name=name, description=desc)

        if not read_only:
            def move_node(node_id: str, to_cluster: str) -> str:
                """Move a node to a different cluster"""
                success = db.move_node(node_id, to_cluster)
                if success:
                    return json.dumps({"success": True, "moved": node_id, "to": to_cluster})
                return json.dumps({"success": False, "error": f"Failed to move {node_id}"})

            def merge_clusters(cluster_a: str, cluster_b: str) -> str:
                """Merge two clusters into one"""
                result = db.merge_clusters(cluster_a, cluster_b)
                return json.dumps({"success": True, "merged_into": result})

            def create_cluster(cluster_id: str, node_ids: str) -> str:
                """Create a new cluster with specified nodes (comma-separated)"""
                nodes = [n.strip() for n in node_ids.split(",")]
                success = db.create_cluster(cluster_id, nodes)
                return json.dumps({"success": success, "cluster": cluster_id, "nodes": len(nodes)})

            write_functions = [
                (move_node, "move_node", "Move a node to a different cluster"),
                (merge_clusters, "merge_clusters", "Merge two clusters into one"),
                (create_cluster, "create_cluster", "Create a new cluster with nodes (comma-separated IDs)"),
            ]

            for func, name, desc in write_functions:
                register_function(func, caller=agent, executor=executor, name=name, description=desc)

    def _compute_external_metrics(self, clusters):
        """Compute TurboMQ and MoJo-FM using Java jars"""
        if not self.metrics_calculator:
            return None

        print(f"\n📍 Computing TurboMQ / MoJo-FM...")
        return self.metrics_calculator.calculate_all(clusters, self.reference_rsf_path)

    def run_clustering(self, max_iterations: int = 3) -> Optional[Dict]:
        """
        Run the clustering improvement loop.
        No fallback — returns None if LLM fails.
        """
        print(f"\n{'='*70}")
        print("DB-BACKED CLUSTERING ANALYSIS")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Max iterations: {max_iterations}")
        print(f"Dependency RSF: {self.dependency_rsf_path or 'N/A'}")
        print(f"Reference RSF: {self.reference_rsf_path or 'N/A'}")

        self.setup_agents()

        iteration_history = []
        best_clusters = None
        best_scores = None

        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")

            self.db.start_iteration()

            metrics_before = self.db.calculate_metrics()
            print(f"📊 Cohesion: {metrics_before['cohesion']:.4f}")

            print(f"\n📍 Step 1: Clustering agent analyzing and improving...")

            message = f"""Analyze the current clustering and make improvements.

Current metrics:
- Cohesion: {metrics_before['cohesion']:.4f}
- Internal edges: {metrics_before['internal_edges']}
- External edges: {metrics_before['external_edges']}

Start by calling get_graph_summary() and get_problematic_nodes() to understand the situation.
Then use move_node() to fix any issues.

Say "CLUSTERING_COMPLETE" when you're satisfied with the clustering."""

            try:
                if not self.verbose:
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        self.user_proxy.initiate_chat(
                            self.clustering_agent, message=message, max_turns=30)
                else:
                    self.user_proxy.initiate_chat(
                        self.clustering_agent, message=message, max_turns=30)
            except Exception as e:
                print(f"❌ LLM clustering failed: {e}")
                return None

            current_clustering = self.db.get_current_clustering()
            if not current_clustering:
                print("❌ No clustering produced by LLM.")
                return None

            metrics_after = self.db.calculate_metrics()
            print(f"📊 Cohesion after: {metrics_after['cohesion']:.4f}")
            improvement = metrics_after['cohesion'] - metrics_before['cohesion']
            print(f"📈 Improvement: {improvement:+.4f}")

            ext_metrics = self._compute_external_metrics(current_clustering)
            turbomq = ext_metrics.get('turbomq') if ext_metrics else None
            mojo_fm = ext_metrics.get('mojo_fm') if ext_metrics else None

            if turbomq is not None:
                print(f"✓ TurboMQ: {turbomq:.4f}")
            if mojo_fm is not None:
                print(f"✓ MoJo-FM: {mojo_fm:.2f}")

            iteration_history.append({
                "iteration": iteration + 1,
                "num_clusters": len(current_clustering),
                "cohesion": metrics_after['cohesion'],
                "turbomq": turbomq,
                "mojo_fm": mojo_fm,
            })

            best_clusters = current_clustering
            best_scores = {"turbomq": turbomq, "mojo_fm": mojo_fm, **metrics_after}

            if iteration < max_iterations - 1:
                print(f"\n📍 Step 2: Inspector analyzing...")
                inspector_feedback = self._run_inspector(metrics_after, turbomq, mojo_fm)
                if inspector_feedback:
                    print(f"✓ Inspector provided feedback")
                    self.clustering_agent.reset()
                else:
                    print("⚠ Inspector returned no feedback, stopping.")
                    break

        if not best_clusters:
            print("❌ No valid clustering produced.")
            return None

        stats = {
            "num_clusters": len(best_clusters),
            "cluster_sizes": {name: len(nodes) for name, nodes in best_clusters.items()},
        }

        return {
            "model": self.model,
            "approach": "db_backed_clustering",
            "clusters": best_clusters,
            "statistics": stats,
            "metrics": best_scores,
            "iteration_history": iteration_history,
            "metadata": {"timestamp": time.strftime('%Y-%m-%d %H:%M:%S')},
        }

    def _run_inspector(self, current_metrics: Dict, turbomq=None, mojo_fm=None) -> Optional[str]:
        tmq_str = f"{turbomq:.4f}" if turbomq is not None else "N/A"
        mojo_str = f"{mojo_fm:.2f}" if mojo_fm is not None else "N/A"

        message = f"""Review the current clustering quality.

Current metrics:
- Cohesion: {current_metrics['cohesion']:.4f}
- TurboMQ: {tmq_str}
- MoJo-FM: {mojo_str}
- Internal edges: {current_metrics['internal_edges']}
- External edges: {current_metrics['external_edges']}

Use get_problematic_nodes() and get_cluster_connections() to identify issues.
Provide specific orders for improvement."""

        try:
            if not self.verbose:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.inspector_proxy.initiate_chat(
                        self.inspector_agent, message=message, max_turns=10)
            else:
                self.inspector_proxy.initiate_chat(
                    self.inspector_agent, message=message, max_turns=10)

            for conv_id, messages in self.inspector_agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        return msg.get('content', '')
        except Exception as e:
            print(f"⚠ Inspector error: {e}")
        return None

    def save_results(self, result: Dict, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Results saved to {output_path}")

    def export_clustering(self, rsf_path: str):
        self.db.export_clustering_to_rsf(rsf_path)

    def close(self):
        if self.db:
            self.db.close()


def inspect_database(db_path: str):
    """Manual inspection of a DB-backed clustering database"""
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    db = ClusteringDatabase(db_path)

    print(f"\n{'='*60}")
    print(f"DATABASE INSPECTION: {db_path}")
    print(f"{'='*60}")

    print("\n📊 Graph Summary:")
    print(db.get_graph_summary())

    print("\n📋 Cluster List:")
    print(db.get_cluster_list())

    print("\n🔗 Cluster Connections (top cross-cluster edges):")
    print(db.get_cluster_connections())

    print("\n⚠ Problematic Nodes (top 10):")
    print(db.get_problematic_nodes(10))

    print("\n💡 Suggested Moves:")
    print(db.get_suggested_moves(5))

    clustering = db.get_current_clustering()
    print(f"\n📦 Clusters ({len(clustering)} total):")
    for cid, nodes in sorted(clustering.items(), key=lambda x: -len(x[1])):
        sample = nodes[:5]
        extra = f"... +{len(nodes)-5} more" if len(nodes) > 5 else ""
        print(f"   {cid} ({len(nodes)} nodes): {', '.join(sample)} {extra}")

    metrics = db.calculate_metrics()
    print(f"\n📈 Internal Metrics:")
    print(f"   Cohesion: {metrics['cohesion']:.4f}")
    print(f"   Coupling: {metrics['coupling']:.4f}")
    print(f"   Internal edges: {metrics['internal_edges']}")
    print(f"   External edges: {metrics['external_edges']}")

    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="DB-Backed Graph Clustering with LLM (TurboMQ + MoJo-FM)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run clustering analysis")
    run_parser.add_argument("graph_file", help="Path to pickled graph file")
    run_parser.add_argument("--dependency-rsf", help="Path to dependency RSF file (for TurboMQ)")
    run_parser.add_argument("--reference-rsf", help="Path to reference clustering RSF (for MoJo-FM)")
    run_parser.add_argument("--model", default="llama3:8b", help="Ollama model name")
    run_parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    run_parser.add_argument("--iterations", type=int, default=3, help="Max iterations")
    run_parser.add_argument("--db-path", default=":memory:", help="Database path (or :memory:)")
    run_parser.add_argument("--output", default="db_clustering_result.json", help="Output file")
    run_parser.add_argument("--export-rsf", default=None, help="Export clustering to RSF file")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    run_parser.add_argument("--use-louvain", action="store_true", help="Use Louvain for initial clustering")
    run_parser.add_argument("--louvain-resolution", type=float, default=1.0, help="Louvain resolution")

    # --- inspect command ---
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a clustering database")
    inspect_parser.add_argument("db_path", help="Path to SQLite database file")

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_database(args.db_path)
        return

    if args.command == "run" or args.command is None:
        if not hasattr(args, 'graph_file'):
            parser.print_help()
            return

        analyzer = DBBackedAnalyzer(
            db_path=args.db_path,
            ollama_host=args.host,
            model=args.model,
            dependency_rsf_path=args.dependency_rsf,
            reference_rsf_path=args.reference_rsf,
            verbose=args.verbose,
        )

        if args.use_louvain:
            analyzer.initialize_from_louvain(args.graph_file, resolution=args.louvain_resolution)
        else:
            analyzer.initialize_database(args.graph_file)

        result = analyzer.run_clustering(max_iterations=args.iterations)

        if result:
            analyzer.save_results(result, args.output)

            if args.export_rsf:
                analyzer.export_clustering(args.export_rsf)

            print(f"\n{'='*70}")
            print("FINAL RESULTS")
            print(f"{'='*70}")
            print(f"Clusters: {result['statistics']['num_clusters']}")
            if result.get('metrics'):
                m = result['metrics']
                print(f"TurboMQ: {m.get('turbomq', 'N/A')}")
                print(f"MoJo-FM: {m.get('mojo_fm', 'N/A')}")
                print(f"Cohesion: {m.get('cohesion', 'N/A')}")

            sizes = list(result['statistics']['cluster_sizes'].values())
            if sizes:
                print(f"\nCluster sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")
        else:
            print("❌ Analysis failed — LLM produced no valid clustering.")
            sys.exit(1)

        analyzer.close()


if __name__ == "__main__":
    main()
