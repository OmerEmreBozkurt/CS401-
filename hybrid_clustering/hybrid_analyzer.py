#!/usr/bin/env python3
"""
Hybrid Baseline-Guided Multi-Agent Clustering System

Architecture:
1. ANALYST AGENT: Compares baselines, identifies controversial nodes, picks best starting point
2. CLUSTERING AGENT: Starts from best baseline, focuses on controversial nodes
3. INSPECTOR AGENT: Reviews changes, compares with baselines, gives improvement orders

This is a token-efficient, deterministic-friendly approach to graph clustering.
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
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, register_function
except ImportError:
    print("❌ Error: AutoGen not installed")
    print("Install with: pip install pyautogen")
    sys.exit(1)

from hybrid_database import HybridClusteringDatabase


# ============================================================================
# Prompts
# ============================================================================

ANALYST_SYSTEM_PROMPT = """You are a Data Analyst specializing in clustering algorithm comparison.

You have access to multiple baseline clustering results (ACDC, MGMC, Louvain, etc.) stored in a database.

YOUR TASK:
1. Compare all baseline algorithms
2. Identify the best baseline to start from (prefer highest TurboMQ, then cohesion)
3. Find "controversial" nodes where algorithms disagree
4. Provide a strategic recommendation

AVAILABLE FUNCTIONS:
- get_baseline_comparison() - Compare all baselines (TurboMQ, cohesion, etc.)
- get_best_baseline() - Get the best performing baseline
- get_controversial_nodes() - Find nodes where baselines disagree
- get_baseline_placement(node_id) - See where each baseline placed a node

OUTPUT FORMAT:
After your analysis, provide a JSON summary:
{
    "best_baseline": "algorithm_id",
    "best_turbomq": 0.XX,
    "num_controversial_nodes": N,
    "key_controversial_nodes": ["node1", "node2", ...],
    "recommendation": "Start from X and focus on nodes Y, Z which are controversial"
}

Say "ANALYSIS_COMPLETE" when done."""


CLUSTERING_SYSTEM_PROMPT = """You are a Clustering Expert for microservice architecture.

You are working with a database-backed system. Instead of seeing all nodes at once,
you query specific information as needed. This is much more efficient!

CONTEXT:
- You are starting from a baseline clustering (identified by the Analyst)
- You should focus on CONTROVERSIAL nodes (where baseline algorithms disagreed)
- Your goal is to IMPROVE the clustering's TurboMQ score based on actual edge connections

AVAILABLE FUNCTIONS (READ):
- get_graph_summary() - Current clustering metrics
- get_cluster_list() - List all clusters
- get_cluster_info(cluster_id) - Details about a cluster
- get_controversial_nodes() - Nodes where baselines disagreed
- get_problematic_nodes() - Nodes with more external than internal edges
- get_suggested_moves() - AI-suggested improvements
- get_baseline_placement(node_id) - See where baselines placed a node
- compare_with_baselines() - Compare current vs all baselines (TurboMQ)

AVAILABLE FUNCTIONS (WRITE):
- move_node(node_id, to_cluster) - Move a node
- merge_clusters(cluster_a, cluster_b) - Merge two clusters

STRATEGY:
1. Start with get_graph_summary() to see current state
2. Check get_controversial_nodes() - these need attention
3. For each controversial node, use get_baseline_placement() to understand disagreement
4. Use get_suggested_moves() for improvement ideas
5. Make targeted moves using move_node()
6. Verify improvement with compare_with_baselines()

The primary metric is TurboMQ. Maximize intra-cluster connectivity and minimize inter-cluster coupling.

When satisfied, say "CLUSTERING_COMPLETE" with a summary of changes made."""


INSPECTOR_SYSTEM_PROMPT = """You are a Quality Inspector for clustering decisions.

Your role is to critically review the clustering and suggest improvements to maximize TurboMQ.

AVAILABLE FUNCTIONS:
- get_graph_summary() - Current metrics
- compare_with_baselines() - How does current compare to baselines (TurboMQ)?
- get_problematic_nodes() - Find misplaced nodes
- get_cluster_info(cluster_id) - Inspect a cluster
- get_controversial_nodes() - Review controversial nodes
- get_cluster_list() - List all clusters

YOUR PROCESS:
1. Check compare_with_baselines() - are we beating all baselines on TurboMQ?
2. Check get_problematic_nodes() - any nodes that should move?
3. Check cluster sizes - any too large or too small?
4. Provide SPECIFIC improvement orders

OUTPUT FORMAT:
{
    "current_turbomq": "0.XX or N/A",
    "vs_best_baseline": "+0.XX" or "-0.XX",
    "status": "improved" or "needs_work",
    "problems_found": ["issue 1", "issue 2"],
    "specific_orders": [
        "Move node_X to cluster_Y because it has 5 edges there vs 1 in current",
        "Merge cluster_A and cluster_B because they have 10 edges between them"
    ]
}

Say "INSPECTION_COMPLETE" when done."""


# ============================================================================
# Metrics Calculator (TurboMQ + MoJo-FM via Java jars)
# ============================================================================

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


# ============================================================================
# Hybrid Analyzer
# ============================================================================

class HybridBaselineAnalyzer:
    """
    Multi-agent system for baseline-guided clustering improvement.
    
    Flow:
    1. Load baselines into database
    2. Analyst Agent analyzes and picks best starting point
    3. Clustering Agent improves from baseline
    4. Inspector Agent reviews and suggests more improvements
    5. Iterate until satisfied
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
        
        self.db: Optional[HybridClusteringDatabase] = None
        self.metrics_calculator: Optional[MetricsCalculator] = None
        
        # Agents
        self.analyst_agent = None
        self.analyst_proxy = None
        self.clustering_agent = None
        self.clustering_proxy = None
        self.inspector_agent = None
        self.inspector_proxy = None
    
    def initialize(self, graph_pickle_path: str):
        """Initialize database with graph"""
        self.db = HybridClusteringDatabase(self.db_path)
        graph = self.db.load_graph_from_pickle(graph_pickle_path)
        
        if self.dependency_rsf_path:
            self.metrics_calculator = MetricsCalculator(self.dependency_rsf_path)
        
        return graph
    
    def _compute_external_metrics(self, clusters):
        """Compute TurboMQ and MoJo-FM using Java jars"""
        if not self.metrics_calculator:
            return None

        print(f"\n📍 Computing TurboMQ / MoJo-FM...")
        return self.metrics_calculator.calculate_all(clusters, self.reference_rsf_path)
    
    def add_baseline_from_rsf(self, rsf_path: str, baseline_id: str, 
                               algorithm: str, turbomq: float = None, mojo_fm: float = None):
        """Add a baseline from RSF file"""
        self.db.load_baseline_from_rsf(rsf_path, baseline_id, algorithm, turbomq, mojo_fm)
    
    def add_baseline(self, baseline_id: str, algorithm: str,
                     clustering: Dict[str, List[str]],
                     turbomq: float = None, mojo_fm: float = None):
        """Add a baseline clustering"""
        self.db.add_baseline(baseline_id, algorithm, clustering, turbomq, mojo_fm)
    
    def _create_llm_config(self):
        """Create LLM config for agents"""
        return {
            "config_list": [{
                "model": self.model,
                "base_url": f"{self.ollama_host}/v1",
                "api_key": "ollama",
                "api_type": "openai",
                "temperature": 0,  # Deterministic
            }],
            "timeout": 600,
            "cache_seed": 42,  # For reproducibility
        }
    
    def setup_analyst_agent(self):
        """Setup the Analyst Agent"""
        self.analyst_agent = AssistantAgent(
            name="Analyst",
            system_message=ANALYST_SYSTEM_PROMPT,
            llm_config=self._create_llm_config()
        )
        
        self.analyst_proxy = UserProxyAgent(
            name="AnalystExecutor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            is_termination_msg=lambda msg: "ANALYSIS_COMPLETE" in msg.get("content", ""),
            silent=not self.verbose,
        )
        
        self._register_analyst_functions()
    
    def setup_clustering_agent(self):
        """Setup the Clustering Agent"""
        self.clustering_agent = AssistantAgent(
            name="ClusteringExpert",
            system_message=CLUSTERING_SYSTEM_PROMPT,
            llm_config=self._create_llm_config()
        )
        
        self.clustering_proxy = UserProxyAgent(
            name="ClusteringExecutor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=30,
            code_execution_config=False,
            default_auto_reply="Continue improving the clustering.",
            is_termination_msg=lambda msg: "CLUSTERING_COMPLETE" in msg.get("content", ""),
            silent=not self.verbose,
        )
        
        self._register_clustering_functions()
    
    def setup_inspector_agent(self):
        """Setup the Inspector Agent"""
        self.inspector_agent = AssistantAgent(
            name="Inspector",
            system_message=INSPECTOR_SYSTEM_PROMPT,
            llm_config=self._create_llm_config()
        )
        
        self.inspector_proxy = UserProxyAgent(
            name="InspectorExecutor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            is_termination_msg=lambda msg: "INSPECTION_COMPLETE" in msg.get("content", ""),
            silent=not self.verbose,
        )
        
        self._register_inspector_functions()
    
    def _register_analyst_functions(self):
        """Register functions for Analyst Agent"""
        db = self.db
        
        def get_baseline_comparison() -> str:
            return db.get_baseline_comparison()
        
        def get_best_baseline() -> str:
            return db.get_best_baseline()
        
        def get_controversial_nodes(limit: int = 20) -> str:
            return db.get_controversial_nodes(limit)
        
        def get_baseline_placement(node_id: str) -> str:
            return db.get_baseline_placement(node_id)
        
        functions = [
            (get_baseline_comparison, "get_baseline_comparison", "Compare all baseline algorithms"),
            (get_best_baseline, "get_best_baseline", "Get the best performing baseline"),
            (get_controversial_nodes, "get_controversial_nodes", "Find nodes where baselines disagree"),
            (get_baseline_placement, "get_baseline_placement", "See where each baseline placed a node"),
        ]
        
        for func, name, desc in functions:
            register_function(func, caller=self.analyst_agent, executor=self.analyst_proxy, 
                            name=name, description=desc)
    
    def _register_clustering_functions(self):
        """Register functions for Clustering Agent"""
        db = self.db
        
        # Read functions
        def get_graph_summary() -> str:
            return db.get_graph_summary()
        
        def get_cluster_list() -> str:
            return db.get_cluster_list()
        
        def get_cluster_info(cluster_id: str) -> str:
            return db.get_cluster_info(cluster_id)
        
        def get_controversial_nodes(limit: int = 15) -> str:
            return db.get_controversial_nodes(limit)
        
        def get_problematic_nodes(limit: int = 10) -> str:
            return db.get_problematic_nodes(limit)
        
        def get_suggested_moves(limit: int = 5) -> str:
            return db.get_suggested_moves(limit)
        
        def get_baseline_placement(node_id: str) -> str:
            return db.get_baseline_placement(node_id)
        
        def compare_with_baselines() -> str:
            return db.compare_with_baselines()
        
        # Write functions
        def move_node(node_id: str, to_cluster: str) -> str:
            success = db.move_node(node_id, to_cluster)
            return json.dumps({"success": success, "moved": node_id, "to": to_cluster})
        
        def merge_clusters(cluster_a: str, cluster_b: str) -> str:
            result = db.merge_clusters(cluster_a, cluster_b)
            return json.dumps({"success": True, "merged_into": result})
        
        read_functions = [
            (get_graph_summary, "get_graph_summary", "Get current clustering metrics"),
            (get_cluster_list, "get_cluster_list", "List all clusters"),
            (get_cluster_info, "get_cluster_info", "Get details about a cluster"),
            (get_controversial_nodes, "get_controversial_nodes", "Nodes where baselines disagreed"),
            (get_problematic_nodes, "get_problematic_nodes", "Nodes with more external edges"),
            (get_suggested_moves, "get_suggested_moves", "AI-suggested node moves"),
            (get_baseline_placement, "get_baseline_placement", "Where baselines placed a node"),
            (compare_with_baselines, "compare_with_baselines", "Compare current vs baselines"),
        ]
        
        write_functions = [
            (move_node, "move_node", "Move a node to different cluster"),
            (merge_clusters, "merge_clusters", "Merge two clusters"),
        ]
        
        for func, name, desc in read_functions + write_functions:
            register_function(func, caller=self.clustering_agent, executor=self.clustering_proxy,
                            name=name, description=desc)
    
    def _register_inspector_functions(self):
        """Register functions for Inspector Agent"""
        db = self.db
        
        def get_graph_summary() -> str:
            return db.get_graph_summary()
        
        def compare_with_baselines() -> str:
            return db.compare_with_baselines()
        
        def get_problematic_nodes(limit: int = 10) -> str:
            return db.get_problematic_nodes(limit)
        
        def get_cluster_info(cluster_id: str) -> str:
            return db.get_cluster_info(cluster_id)
        
        def get_controversial_nodes(limit: int = 10) -> str:
            return db.get_controversial_nodes(limit)
        
        def get_cluster_list() -> str:
            return db.get_cluster_list()
        
        functions = [
            (get_graph_summary, "get_graph_summary", "Get current metrics"),
            (compare_with_baselines, "compare_with_baselines", "Compare with all baselines"),
            (get_problematic_nodes, "get_problematic_nodes", "Find misplaced nodes"),
            (get_cluster_info, "get_cluster_info", "Inspect a cluster"),
            (get_controversial_nodes, "get_controversial_nodes", "Check controversial nodes"),
            (get_cluster_list, "get_cluster_list", "List all clusters"),
        ]
        
        for func, name, desc in functions:
            register_function(func, caller=self.inspector_agent, executor=self.inspector_proxy,
                            name=name, description=desc)
    
    def run_analysis(self, max_iterations: int = 3) -> Dict:
        """
        Run the full hybrid analysis pipeline.
        
        Flow:
        1. Analyst picks best baseline
        2. Clustering Agent improves
        3. Inspector reviews
        4. Repeat if needed
        """
        print(f"\n{'='*70}")
        print("HYBRID BASELINE-GUIDED CLUSTERING")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Max iterations: {max_iterations}")
        print(f"Database: {self.db_path}")
        print(f"Dependency RSF: {self.dependency_rsf_path or 'N/A'}")
        print(f"Reference RSF: {self.reference_rsf_path or 'N/A'}")
        
        # Setup agents
        self.setup_analyst_agent()
        self.setup_clustering_agent()
        self.setup_inspector_agent()
        
        iteration_history = []
        best_clusters = None
        best_scores = None
        
        # =====================================================================
        # PHASE 1: Analyst picks best baseline
        # =====================================================================
        print(f"\n{'='*70}")
        print("PHASE 1: ANALYST ANALYSIS")
        print(f"{'='*70}")
        
        analyst_result = self._run_analyst()
        best_baseline = self._extract_best_baseline(analyst_result)
        
        if best_baseline:
            print(f"✓ Analyst recommends starting from: {best_baseline}")
            self.db.start_from_baseline(best_baseline)
        else:
            baselines = json.loads(self.db.get_baseline_comparison())
            if baselines:
                best_baseline = baselines[0]['id']
                self.db.start_from_baseline(best_baseline)
                print(f"⚠ Using first baseline: {best_baseline}")
        
        initial_metrics = self.db.calculate_current_metrics()
        initial_clustering = self.db.get_current_clustering()
        initial_ext = self._compute_external_metrics(initial_clustering)
        initial_turbomq = initial_ext.get('turbomq') if initial_ext else None
        initial_mojo_fm = initial_ext.get('mojo_fm') if initial_ext else None
        
        print(f"📊 Initial cohesion: {initial_metrics['cohesion']:.4f}")
        if initial_turbomq is not None:
            print(f"📊 Initial TurboMQ: {initial_turbomq:.4f}")
        if initial_mojo_fm is not None:
            print(f"📊 Initial MoJo-FM: {initial_mojo_fm:.2f}")
        
        # =====================================================================
        # PHASE 2-4: Clustering + Inspector Loop
        # =====================================================================
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            
            self.db.start_iteration()
            metrics_before = self.db.calculate_current_metrics()
            
            # PHASE 2: Clustering Agent improves
            print(f"\n📍 PHASE 2: CLUSTERING AGENT")
            print(f"   Starting cohesion: {metrics_before['cohesion']:.4f}")
            
            clustering_result = self._run_clustering_agent(iteration)
            
            current_clustering = self.db.get_current_clustering()
            if not current_clustering:
                print("❌ No clustering produced.")
                break
            
            metrics_after = self.db.calculate_current_metrics()
            print(f"   Ending cohesion: {metrics_after['cohesion']:.4f}")
            print(f"   Change: {metrics_after['cohesion'] - metrics_before['cohesion']:+.4f}")
            
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
                "cohesion_before": metrics_before['cohesion'],
                "cohesion_after": metrics_after['cohesion'],
                "improvement": metrics_after['cohesion'] - metrics_before['cohesion'],
                "turbomq": turbomq,
                "mojo_fm": mojo_fm,
            })
            
            best_clusters = current_clustering
            best_scores = {"turbomq": turbomq, "mojo_fm": mojo_fm, **metrics_after}
            
            # PHASE 3: Inspector reviews
            print(f"\n📍 PHASE 3: INSPECTOR REVIEW")
            
            inspector_result = self._run_inspector(metrics_after, turbomq, mojo_fm)
            inspector_orders = self._extract_inspector_orders(inspector_result)
            
            if not inspector_orders:
                print("✓ Inspector satisfied — no further orders")
                break
            
            # Reset clustering agent for next iteration
            self.clustering_agent.reset()
        
        # =====================================================================
        # FINAL RESULTS
        # =====================================================================
        final_clustering = best_clusters or self.db.get_current_clustering()
        final_metrics = self.db.calculate_current_metrics()
        final_ext = self._compute_external_metrics(final_clustering)
        final_turbomq = final_ext.get('turbomq') if final_ext else None
        final_mojo_fm = final_ext.get('mojo_fm') if final_ext else None
        
        return {
            "model": self.model,
            "approach": "hybrid_baseline_guided",
            "starting_baseline": best_baseline,
            "initial_cohesion": initial_metrics['cohesion'],
            "initial_turbomq": initial_turbomq,
            "final_cohesion": final_metrics['cohesion'],
            "final_turbomq": final_turbomq,
            "final_mojo_fm": final_mojo_fm,
            "improvement_cohesion": final_metrics['cohesion'] - initial_metrics['cohesion'],
            "improvement_turbomq": (final_turbomq - initial_turbomq) if (final_turbomq is not None and initial_turbomq is not None) else None,
            "clusters": final_clustering,
            "metrics": best_scores or final_metrics,
            "iteration_history": iteration_history,
            "action_history": self.db.get_history(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _run_analyst(self) -> Optional[str]:
        """Run analyst agent"""
        message = """Analyze the available baseline algorithms and provide a recommendation.

Steps:
1. Call get_baseline_comparison() to see all baselines
2. Call get_best_baseline() to identify the top performer
3. Call get_controversial_nodes() to find where algorithms disagree
4. Provide your recommendation on which baseline to start from

Focus on TurboMQ score (prefer highest TurboMQ) and identify which nodes need attention."""
        
        try:
            if not self.verbose:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.analyst_proxy.initiate_chat(
                        self.analyst_agent,
                        message=message,
                        max_turns=10
                    )
            else:
                self.analyst_proxy.initiate_chat(
                    self.analyst_agent,
                    message=message,
                    max_turns=10
                )
            
            # Extract result
            for conv_id, messages in self.analyst_agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        return msg.get('content', '')
        except Exception as e:
            print(f"⚠ Analyst error: {e}")
        
        return None
    
    def _run_clustering_agent(self, iteration: int) -> Optional[str]:
        """Run clustering agent"""
        if iteration == 0:
            message = """You are starting from the best baseline. Your task is to IMPROVE it.

Steps:
1. Call get_graph_summary() to see current metrics
2. Call get_controversial_nodes() to find nodes where baselines disagreed
3. For controversial nodes, check get_baseline_placement() to understand the disagreement
4. Use get_suggested_moves() for improvement ideas
5. Make changes with move_node() to improve TurboMQ (maximize intra-cluster edges, minimize inter-cluster edges)
6. Verify with compare_with_baselines()

Focus on the controversial nodes first - they are likely misplaced!

Say "CLUSTERING_COMPLETE" when you've made improvements."""
        else:
            message = """Continue improving the clustering based on the Inspector's feedback.

Check get_problematic_nodes() and get_suggested_moves() for improvement opportunities.
Use move_node() to make changes.

Say "CLUSTERING_COMPLETE" when done."""
        
        try:
            if not self.verbose:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.clustering_proxy.initiate_chat(
                        self.clustering_agent,
                        message=message,
                        max_turns=30
                    )
            else:
                self.clustering_proxy.initiate_chat(
                    self.clustering_agent,
                    message=message,
                    max_turns=30
                )
            
            for conv_id, messages in self.clustering_agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        return msg.get('content', '')
        except Exception as e:
            print(f"⚠ Clustering error: {e}")
        
        return None
    
    def _run_inspector(self, current_metrics: Dict = None,
                       turbomq=None, mojo_fm=None) -> Optional[str]:
        """Run inspector agent with TurboMQ/MoJo-FM context"""
        tmq_str = f"{turbomq:.4f}" if turbomq is not None else "N/A"
        mojo_str = f"{mojo_fm:.2f}" if mojo_fm is not None else "N/A"
        cohesion_str = f"{current_metrics['cohesion']:.4f}" if current_metrics else "N/A"
        
        message = f"""Review the current clustering quality.

Current metrics:
- TurboMQ: {tmq_str}
- MoJo-FM: {mojo_str}
- Cohesion: {cohesion_str}

Steps:
1. Call compare_with_baselines() to see if we're beating the baselines on TurboMQ
2. Call get_problematic_nodes() to find any remaining issues
3. Call get_cluster_list() to check cluster sizes
4. Provide specific improvement orders if needed

Focus on improving TurboMQ score. If TurboMQ is already beating all baselines, say so.
Otherwise, provide specific orders like "Move node_X to cluster_Y".

Say "INSPECTION_COMPLETE" when done."""
        
        try:
            if not self.verbose:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.inspector_proxy.initiate_chat(
                        self.inspector_agent,
                        message=message,
                        max_turns=10
                    )
            else:
                self.inspector_proxy.initiate_chat(
                    self.inspector_agent,
                    message=message,
                    max_turns=10
                )
            
            for conv_id, messages in self.inspector_agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        return msg.get('content', '')
        except Exception as e:
            print(f"⚠ Inspector error: {e}")
        
        return None
    
    def _extract_best_baseline(self, analyst_result: Optional[str]) -> Optional[str]:
        """Extract best baseline from analyst result"""
        if not analyst_result:
            return None
        
        # Try to find JSON
        try:
            match = re.search(r'\{[^{}]*"best_baseline"[^{}]*\}', analyst_result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get('best_baseline')
        except:
            pass
        
        # Try to find baseline ID mentioned
        for baseline_id in ['acdc', 'mgmc', 'louvain']:
            if baseline_id in analyst_result.lower():
                return baseline_id
        
        return None
    
    def _extract_inspector_orders(self, inspector_result: Optional[str]) -> List[str]:
        """Extract specific orders from inspector result"""
        if not inspector_result:
            return []
        
        orders = []
        
        # Try to find JSON
        try:
            match = re.search(r'\{[^{}]*"specific_orders"[^{}]*\}', inspector_result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                orders = data.get('specific_orders', [])
        except:
            pass
        
        # Also look for "Move" statements
        for line in inspector_result.split('\n'):
            if 'move' in line.lower() and 'node' in line.lower():
                orders.append(line.strip())
        
        return orders[:5]  # Limit to 5 orders
    
    def save_results(self, result: Dict, output_path: str):
        """Save results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Results saved to {output_path}")
    
    def export_clustering(self, rsf_path: str):
        """Export current clustering to RSF"""
        self.db.export_to_rsf(rsf_path)
    
    def close(self):
        """Cleanup"""
        if self.db:
            self.db.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Baseline-Guided Clustering System"
    )
    parser.add_argument("graph_file", help="Path to pickled graph file")
    parser.add_argument("--baseline", action="append", nargs=3,
                       metavar=("ID", "RSF_PATH", "ALGORITHM"),
                       help="Add baseline: --baseline acdc path/to/acdc.rsf ACDC")
    parser.add_argument("--dependency-rsf", help="Path to dependency RSF file (for TurboMQ)")
    parser.add_argument("--reference-rsf", help="Path to reference clustering RSF (for MoJo-FM)")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations")
    parser.add_argument("--db-path", default=":memory:", help="Database path")
    parser.add_argument("--output", default="hybrid_result.json", help="Output file")
    parser.add_argument("--export-rsf", help="Export final clustering to RSF")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.exists(args.graph_file):
        print(f"❌ Graph file not found: {args.graph_file}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = HybridBaselineAnalyzer(
        db_path=args.db_path,
        ollama_host=args.host,
        model=args.model,
        dependency_rsf_path=args.dependency_rsf,
        reference_rsf_path=args.reference_rsf,
        verbose=args.verbose
    )
    
    # Initialize
    analyzer.initialize(args.graph_file)
    
    # Add baselines
    if args.baseline:
        for baseline_id, rsf_path, algorithm in args.baseline:
            if os.path.exists(rsf_path):
                analyzer.add_baseline_from_rsf(rsf_path, baseline_id, algorithm)
            else:
                print(f"⚠ Baseline RSF not found: {rsf_path}")
    else:
        print("⚠ No baselines provided. Using Louvain as default...")
        # Run Louvain as default baseline
        import networkx as nx
        with open(args.graph_file, 'rb') as f:
            import pickle
            graph = pickle.load(f)
        
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(graph.to_undirected(), seed=42)
            louvain_clustering = {f"cluster_{i}": [str(n) for n in comm] 
                                 for i, comm in enumerate(communities)}
            analyzer.add_baseline("louvain", "Louvain", louvain_clustering)
        except Exception as e:
            print(f"❌ Could not create Louvain baseline: {e}")
            sys.exit(1)
    
    # Run analysis
    result = analyzer.run_analysis(max_iterations=args.iterations)
    
    # Save results
    analyzer.save_results(result, args.output)
    
    if args.export_rsf:
        analyzer.export_clustering(args.export_rsf)
    
    # Print summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Starting baseline: {result['starting_baseline']}")
    print(f"Initial cohesion: {result['initial_cohesion']:.4f}")
    print(f"Final cohesion: {result['final_cohesion']:.4f}")
    print(f"Cohesion change: {result['improvement_cohesion']:+.4f}")
    if result.get('initial_turbomq') is not None:
        print(f"Initial TurboMQ: {result['initial_turbomq']:.4f}")
    if result.get('final_turbomq') is not None:
        print(f"Final TurboMQ: {result['final_turbomq']:.4f}")
    if result.get('improvement_turbomq') is not None:
        print(f"TurboMQ change: {result['improvement_turbomq']:+.4f}")
    if result.get('final_mojo_fm') is not None:
        print(f"Final MoJo-FM: {result['final_mojo_fm']:.2f}")
    print(f"Final clusters: {len(result['clusters'])}")
    print(f"Total actions: {len(result['action_history'])}")
    
    analyzer.close()


if __name__ == "__main__":
    main()
