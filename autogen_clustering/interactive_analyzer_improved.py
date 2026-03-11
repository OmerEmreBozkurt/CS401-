#!/usr/bin/env python3
import sys
import os
import pickle
import json
import time
import argparse
import contextlib
import io
import re
from typing import Dict, List, Optional, Any

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, register_function
except ImportError:
    print("❌ Error: AutoGen not installed")
    print("Install with: pip install pyautogen")
    sys.exit(1)

import networkx as nx

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
5. Output your clustering in JSON format:

Output your clustering as JSON:
{{
  "clusters": {{
    "cluster_1": ["node1", "node2", ...],
    "cluster_2": ["node3", "node4", ...],
    ...
  }}
}}"""


INSPECTOR_SYSTEM_PROMPT = """You are the Lead Software Architect and Quality Inspector. 
You are critical, strict, and detail-oriented.

Your goal: "Battle" the clustering agent to force improvements.

You have access to the same graph tools as the clustering agent:
- get_node_dependencies(node)
- get_node_dependents(node)
- get_all_edges()

Process:
1. Receive the proposed clusters and the TurboMQ/MoJo scores.
2. If the score is low (TurboMQ < 0.7), DO NOT just accept it.
3. INVESTIGATE:
   - Pick clusters that seem confused or too large.
   - Use `get_node_dependencies` to check edges between them.
   - Find specific nodes that are in the wrong place.
4. ATTACK THE CLUSTERING:
   - "You placed 'OrderService' in Cluster A, but it calls 'PaymentService' in Cluster B 50 times. Move it!"
   - "Cluster 3 is a God Cluster with 500 nodes. Split it!"

Output your feedback as a clear, bulleted list of orders.
Finally, provide the JSON format:
{
  "analysis": "Your textual analysis here...",
  "specific_orders": ["Move node X to Cluster Y", "Merge Cluster A and B"]
}
"""

class GraphDataProvider:
    """Holds the graph data and provides query functions"""
    
    def __init__(self, graph):
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
        return json.dumps(info)
    
    def get_all_nodes(self) -> str:
        """Get list of all nodes"""
        return json.dumps(list(self.graph.nodes()))
    
    def get_all_edges(self) -> str:
        """Get all dependency edges"""
        edges = [{"source": u, "target": v} for u, v in self.graph.edges()]
        return json.dumps(edges)
    
    def get_node_dependencies(self, node: str) -> str:
        """Get what a node depends on"""
        if node not in self.graph:
            return json.dumps([])
        return json.dumps(list(self.graph.successors(node)))
    
    def get_node_dependents(self, node: str) -> str:
        """Get what depends on this node"""
        if node not in self.graph:
            return json.dumps([])
        return json.dumps(list(self.graph.predecessors(node)))
    
    def get_hub_nodes(self, top_n: int = 10) -> str:
        """Get most depended-on nodes"""
        in_degrees = dict(self.graph.in_degree())
        sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        result = [{"node": node, "dependents": degree} for node, degree in sorted_nodes[:top_n]]
        return json.dumps(result)


class MetricsCalculator:
    """Calculates TurboMQ and MoJo-FM metrics"""
    
    def __init__(self, dependency_rsf_path, experiments_dir=None):
        self.dependency_rsf_path = os.path.abspath(dependency_rsf_path) if dependency_rsf_path else None
        self.experiments_dir = experiments_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "experiments"
        )
        
        # Try to find Java executable
        self.java_cmd = self._find_java()
    
    def _find_java(self):
        """Find the Java executable, trying common locations"""
        import shutil
        
        # Try to find java in PATH
        java_path = shutil.which('java')
        if java_path:
            return java_path
        
        # Try common macOS locations
        common_paths = [
            '/usr/bin/java',
            '/Library/Java/JavaVirtualMachines/*/Contents/Home/bin/java',
            '/opt/homebrew/bin/java',
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Default to 'java' and hope it's in PATH
        return 'java'
    
    def clusters_to_rsf(self, clusters, output_path):
        """Convert clusters dict to RSF format"""
        with open(output_path, 'w') as f:
            for cluster_name, nodes in clusters.items():
                for node in nodes:
                    f.write(f"contain {cluster_name} {node}\n")
    
    def calculate_turbomq(self, clustering_rsf_path):
        """Calculate TurboMQ score using Java jar"""
        import subprocess
        import shutil
        
        turbomq_jar = os.path.join(self.experiments_dir, "turbomq.jar")
        
        if not os.path.exists(turbomq_jar):
            print(f"TurboMQ jar not found: {turbomq_jar}")
            print(f"   Experiments dir: {self.experiments_dir}")
            return None
        
        if not os.path.exists(self.dependency_rsf_path):
            print(f"Dependency RSF not found: {self.dependency_rsf_path}")
            return None
        
        if not os.path.exists(clustering_rsf_path):
            print(f"Clustering RSF not found: {clustering_rsf_path}")
            return None
        
        try:
            dep_rsf_temp = os.path.join(self.experiments_dir, "temp_dependency.rsf")
            clust_rsf_temp = os.path.join(self.experiments_dir, "temp_clustering.rsf")
            
            shutil.copy2(self.dependency_rsf_path, dep_rsf_temp)
            shutil.copy2(clustering_rsf_path, clust_rsf_temp)
            
            cmd = ["java", "-jar", "turbomq.jar", "temp_dependency.rsf", "temp_clustering.rsf"]
            print(f"   Executing: {' '.join(cmd)}")
            print(f"   CWD: {self.experiments_dir}")
            print(f"   Java: {self.java_cmd}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir,
                env={**os.environ, 'PATH': os.path.dirname(self.java_cmd) + ':' + os.environ.get('PATH', '')}
            )
            
            # Clean up
            for f in [dep_rsf_temp, clust_rsf_temp]:
                if os.path.exists(f):
                    os.remove(f)
            
            if result.returncode == 0:
                try:
                    return float(result.stdout.strip())
                except ValueError:
                    print(f"TurboMQ: Could not parse output: {result.stdout.strip()}")
                    return None
            else:
                print(f"TurboMQ error: {result.stderr}")
                return None
        except Exception as e:
            print(f"TurboMQ calculation failed: {e}")
            return None
    
    def calculate_mojo_fm(self, clustering_rsf_path, reference_rsf_path=None):
        """Calculate MoJo-FM distance using Java jar"""
        import subprocess
        
        mojo_jar = os.path.join(self.experiments_dir, "mojo.jar")
        
        if not os.path.exists(mojo_jar):
            print(f"MoJo jar not found: {mojo_jar}")
            print(f"   Experiments dir: {self.experiments_dir}")
            return None
            
        if not reference_rsf_path:
            print(f"No reference RSF provided for MoJo-FM")
            return None
        
        if not os.path.exists(reference_rsf_path):
            print(f"Reference RSF not found: {reference_rsf_path}")
            return None
        
        try:
            clust_rsf_abs = os.path.abspath(clustering_rsf_path)
            ref_rsf_abs = os.path.abspath(reference_rsf_path)
            
            # Verify the files exist
            if not os.path.exists(clust_rsf_abs):
                print(f"MoJo-FM error: Clustering RSF not found: {clust_rsf_abs}")
                return None
            
            cmd = ["java", "-jar", "mojo.jar", clust_rsf_abs, ref_rsf_abs, "-fm"]
            print(f"   Executing MoJo: {' '.join(cmd)}")
            print(f"   CWD: {self.experiments_dir}")
            print(f"   Java: {self.java_cmd}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir,
                env={**os.environ, 'PATH': os.path.dirname(self.java_cmd) + ':' + os.environ.get('PATH', '')}
            )
            
            # Debug output
            if result.stderr:
                print(f"   MoJo stderr: {result.stderr[:200]}")
            if result.stdout:
                print(f"   MoJo stdout: {result.stdout[:200]}")
            
            if result.returncode == 0:
                output = result.stdout.strip()
                match = re.search(r'[\d.]+', output)
                if match:
                    score = float(match.group())
                    print(f"   ✓ MoJo-FM score: {score}")
                    return score
                else:
                    print(f"MoJo-FM: Could not parse score from output: {output}")
                    return None
            else:
                print(f"MoJo-FM error (exit code {result.returncode}): {result.stderr}")
                return None
        except Exception as e:
            print(f"MoJo-FM calculation failed: {e}")
            import traceback
            traceback.print_exc()
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
                "mojo_fm": mojo_fm,
                "clustering_rsf": temp_rsf
            }
        except Exception as e:
            return {"turbomq": None, "mojo_fm": None, "clustering_rsf": temp_rsf}


class InteractiveAnalyzerImproved:
    """Analyzer with integrated metrics evaluation and improvement loop"""
    
    def __init__(self, ollama_host="http://localhost:11434", model="gpt-oss:120b-cloud",
                 dependency_rsf_path=None, reference_rsf_path=None, verbose=False, api_key=None):
        self.model = model
        self.graph = None
        self.graph_provider = None
        self.metrics_calculator = None
        self.dependency_rsf_path = dependency_rsf_path
        self.reference_rsf_path = reference_rsf_path
        self.verbose = verbose
        
        # Detect cloud vs local model and configure accordingly
        if model.endswith(':cloud') or '-cloud' in model:
            # Cloud model - still uses local Ollama but needs API key for authentication
            self.is_cloud = True
            self.ollama_host = ollama_host  # Use local Ollama endpoint
            self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "ollama")
            if self.verbose or verbose:
                print(f"[INFO] Detected cloud model: {model}")
                print(f"[INFO] Using local Ollama with cloud authentication: {self.ollama_host}")
                if api_key:
                    print(f"[INFO] API key provided for cloud authentication")
        else:
            # Local model
            self.is_cloud = False
            self.ollama_host = ollama_host
            self.api_key = "ollama"
            if self.verbose or verbose:
                print(f"[INFO] Detected local model: {model}")
                print(f"[INFO] Using local endpoint: {self.ollama_host}")
        
        # Track LLM outputs for debugging
        self.last_llm_output = None
        
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
        if num_clusters is None or num_clusters <= 0:
            # Let LLM decide optimal number
            cluster_instruction = """

Your goal: Analyze the graph and determine the OPTIMAL number of clusters.

IMPORTANT: 
- YOU decide how many clusters based on:
  * Module boundaries and cohesion
  * Dependency patterns
  * Natural groupings in the code
  * Minimizing inter-cluster dependencies
- Create as many clusters as makes sense (could be 3, could be 20+)
- Quality over quantity - don't force a specific number"""
        else:
            cluster_instruction = f"""

Your goal: Create exactly {num_clusters} clusters from the graph."""

        system_prompt = CLUSTERING_SYSTEM_PROMPT + cluster_instruction + """

IMPORTANT OUTPUT FORMAT - Use simple text (NOT JSON):
```
CLUSTER cluster_1:
node_a
node_b
node_c

CLUSTER cluster_2:
node_d
node_e
```

Make sure ALL nodes are assigned to exactly one cluster."""
        
        config_list = [{
            "model": self.model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": self.api_key,
            "api_type": "openai",
            "temperature": 0.1,  
        }]
        
        self.clustering_agent = AssistantAgent(
            name="ClusteringExpert",
            system_message=system_prompt,
            llm_config={
                "config_list": config_list,
                "timeout": 900,  
                "cache_seed": None,
            }
        )
        
        def termination_check(msg):
            content = msg.get("content", "")
            self.last_llm_output = content  
            
            
            if "CLUSTER" in content or ("clusters" in content.lower() and "{" in content):
                try:
                    clusters = self._extract_clusters_from_content(content)
                    if clusters and len(clusters) > 0:
                        if self.verbose:
                            print(f"[DEBUG] Termination: Found valid clusters ({len(clusters)} clusters)")
                        return True
                except:
                    pass
            
            
            if self.verbose:
                print(f"[DEBUG] Termination check: No valid clusters yet")
            return False
        
        self.user_proxy = UserProxyAgent(
            name="GraphProvider",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=25,  
            code_execution_config=False,
            default_auto_reply="Please continue with your analysis and output the final clustering in simple text format (CLUSTER cluster_X: followed by node names).",
            is_termination_msg=termination_check,
            silent=not self.verbose,
        )
        
        self._register_graph_functions(self.clustering_agent, self.user_proxy)
    
    def setup_metrics_agent(self):
        """Setup metrics evaluation agent (The Inspector)"""
        config_list = [{
            "model": self.model,
            "base_url": f"{self.ollama_host}/v1",
            "api_key": self.api_key,
            "api_type": "openai",
            "temperature": 0,
        }]
        
        self.metrics_agent = AssistantAgent(
            name="InspectorArchitecture",
            system_message=INSPECTOR_SYSTEM_PROMPT,
            llm_config={
                "config_list": config_list,
                "timeout": 600,
                "cache_seed": None,
            }
        )
        
        self.metrics_proxy = UserProxyAgent(
            name="InspectorExecutor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,  
            code_execution_config=False,
            is_termination_msg=lambda msg: "specific_orders" in msg.get("content", ""),
            silent=not self.verbose,
        )
        
        self._register_metrics_functions()
        self._register_graph_functions(self.metrics_agent, self.metrics_proxy)
    
    def _register_graph_functions(self, agent, executor):
        """Register graph query functions with proper return type annotations"""
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
        
        def get_hub_nodes(top_n: int = 10) -> str:
            return provider.get_hub_nodes(top_n)
        
        functions = [
            (get_graph_info, "get_graph_info", "Get basic graph statistics (returns JSON)"),
            (get_all_nodes, "get_all_nodes", "Get list of all nodes (returns JSON array)"),
            (get_all_edges, "get_all_edges", "Get all dependency edges (returns JSON array)"),
            (get_node_dependencies, "get_node_dependencies", "Get what a node depends on (returns JSON array)"),
            (get_node_dependents, "get_node_dependents", "Get what depends on a node (returns JSON array)"),
            (get_hub_nodes, "get_hub_nodes", "Get most depended-on nodes (returns JSON array)"),
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
        if not self.metrics_calculator:
            return
        
        calculator = self.metrics_calculator
        ref_path = self.reference_rsf_path
        
        def calculate_turbomq(clusters_json: str) -> str:
            clusters = json.loads(clusters_json)
            result = calculator.calculate_metrics(clusters, ref_path)
            return json.dumps({
                "turbomq": result.get("turbomq"),
                "mojo_fm": result.get("mojo_fm"),
                "note": "TurboMQ: higher is better (0-1). MoJo-FM: lower is better."
            })
        
        register_function(
            calculate_turbomq,
            caller=self.metrics_agent,
            executor=self.metrics_proxy,
            name="calculate_turbomq",
            description="Calculate TurboMQ and MoJo-FM metrics for clustering (input: JSON string)"
        )

    def extract_clusters(self, agent) -> Optional[Dict]:
        """IMPROVED: Extract clusters from agent conversation - supports both text and JSON formats"""
        try:
            all_content = []
            for conv_id, messages in agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        all_content.append(content)
                        
                        if self.verbose:
                            print(f"\n[DEBUG] Checking message ({len(content)} chars): {content[:500]}...")
                        
                        
                        clusters = self._extract_clusters_from_content(content)
                        if clusters:
                            return clusters
            
            if all_content and self.verbose:
                print(f"\n[DEBUG] Full last message:\n{all_content[0]}")
                
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Extract error: {e}")
                import traceback
                traceback.print_exc()
        return None
    
    def _extract_clusters_from_content(self, content: str) -> Optional[Dict]:
        """Extract clusters from content - tries simple text format first, then JSON"""
        
        
        clusters = self._extract_text_clusters(content)
        if clusters:
            if self.verbose:
                print(f"[DEBUG] Extracted using simple text format")
            return clusters
        
        clusters = self._extract_json_clusters(content)
        if clusters:
            return clusters
        
        return None
    
    def _extract_text_clusters(self, content: str) -> Optional[Dict]:
        """Extract clusters from simple text format:
        CLUSTER cluster_1:
        node1
        node2
        
        CLUSTER cluster_2:
        node3
        """
        try:
            clusters = {}
            lines = content.split('\n')
            current_cluster = None
            
            for line in lines:
                line = line.strip()
                
                
                if line.upper().startswith('CLUSTER'):
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        cluster_name = parts[1].rstrip(':')
                        current_cluster = cluster_name
                        if current_cluster not in clusters:
                            clusters[current_cluster] = []
                
                
                elif current_cluster and line and not line.startswith('#') and not line.startswith('//'):
                    
                    if line not in ['---', '===', '```', '```text']:
                        clusters[current_cluster].append(line)
            
            if clusters and any(len(nodes) > 0 for nodes in clusters.values()):
                if self.verbose:
                    print(f"[DEBUG] Text format found {len(clusters)} clusters")
                return clusters
                
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Text extraction failed: {e}")
        
        return None
    
    def _extract_json_clusters(self, content: str) -> Optional[Dict]:
        """Try multiple strategies to extract clusters from content"""
        
        json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_block_match:
            try:
                data = json.loads(json_block_match.group(1))
                if 'clusters' in data:
                    if self.verbose:
                        print("[DEBUG] Extracted from ```json block")
                    return data['clusters']
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG] Failed to parse ```json block: {e}")
        
        code_block_match = re.search(r'```\w*\s*(\{.*?\})\s*```', content, re.DOTALL)
        if code_block_match:
            try:
                data = json.loads(code_block_match.group(1))
                if 'clusters' in data:
                    if self.verbose:
                        print("[DEBUG] Extracted from ``` block")
                    return data['clusters']
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG] Failed to parse ``` block: {e}")
        
        brace_count = 0
        start_idx = None
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    potential_json = content[start_idx:i+1]
                    if '"clusters"' in potential_json:
                        try:
                            data = json.loads(potential_json)
                            if 'clusters' in data and isinstance(data['clusters'], dict):
                                if self.verbose:
                                    print("[DEBUG] Extracted using brace matching")
                                return data['clusters']
                        except Exception as e:
                            if self.verbose:
                                print(f"[DEBUG] Failed brace match parse: {e}")
                    start_idx = None
    
        patterns = [
            r'\{\s*"clusters"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\}',
            r'"clusters"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
        ]
        
        for pattern in patterns:
            json_match = re.search(pattern, content, re.DOTALL)
            if json_match:
                try:
                    if len(json_match.groups()) > 0:
                        clusters_json = json_match.group(1)
                        data = json.loads(clusters_json)
                    else:
                        data = json.loads(json_match.group())
                        if 'clusters' in data:
                            data = data['clusters']
                    
                    if isinstance(data, dict):
                        if self.verbose:
                            print(f"[DEBUG] Extracted using pattern: {pattern[:50]}...")
                        return data
                except Exception as e:
                    if self.verbose:
                        print(f"[DEBUG] Failed pattern {pattern[:30]}: {e}")
        
        if self.verbose:
            print("[DEBUG] No valid JSON clusters found")
            if len(content) > 0:
                print(f"[DEBUG] Content preview: {content[:1000]}...")
        return None
    
    def analyze_with_metrics(self, num_clusters=10, max_iterations=1):
        """Complete pipeline with improvements"""
        print(f"\n{'='*70}")
        print(f"IMPROVED INTERACTIVE ANALYSIS (INSPECTOR MODE)")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        if num_clusters is None or num_clusters <= 0:
            print(f"Target clusters: LLM will decide optimal number")
        else:
            print(f"Target clusters: {num_clusters}")
        print(f"Max iterations: {max_iterations}")
        print(f"Verbose: {self.verbose}")
        
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
            
           
            print(f"\n📍 Step 1: Clustering...")
            
            max_retries = 3
            clusters = None
            for retry in range(max_retries):
                if retry > 0:
                    print(f"   Retry {retry}/{max_retries}...")
                
                clusters = self._run_clustering(
                    num_clusters, 
                    iteration > 0, 
                    best_scores, 
                    best_clusters, 
                    analysis_feedback=last_analysis
                )
                
                if clusters:
                    print(f"✓ LLM produced {len(clusters)} clusters")
                    break
                else:
                    if retry < max_retries - 1:
                        print(f"LLM clustering failed, retrying...")
                        self.clustering_agent.reset()
                    else:
                        print(f"❌ LLM clustering failed after {max_retries} attempts")
                        print(f"   Please check:")
                        print(f"   1. Model '{self.model}' is running (ollama list)")
                        print(f"   2. Ollama server is accessible at {self.ollama_host}")
                        print(f"   3. Model has enough context window for {self.graph.number_of_nodes()} nodes")
                        raise RuntimeError(f"LLM failed to produce valid clustering after {max_retries} attempts")
            
           
            print(f"\n📍 Step 2: Evaluating metrics...")
            scores = self._evaluate_metrics(clusters)
            
            iteration_history.append({
                "iteration": iteration + 1,
                "num_clusters": len(clusters),
                "turbomq": scores.get('turbomq') if scores else None,
                "mojo_fm": scores.get('mojo_fm') if scores else None,
                "used_fallback": clusters == best_clusters  # Track if we used fallback
            })
            
           
            if self.metrics_agent and scores:
                print(f"\n📍 Step 3: Inspector is investigating...")
                analysis = self._analyze_scores(scores, clusters)
                
                
                if iteration < max_iterations - 1:
                    turbomq = scores.get('turbomq', 0) or 0
                    
                    
                    if turbomq < 0.85:  # Target threshold
                        print(f"\n📍 Step 4: Preparing for improvement (TurboMQ={turbomq:.4f} < 0.85)...")
                        best_clusters = clusters
                        best_scores = scores
                        last_analysis = analysis
                        
                        print("   (Resetting clustering agent memory for fresh attempt)")
                        self.clustering_agent.reset()
                        continue
                    else:
                        print(f"✓ TurboMQ={turbomq:.4f} meets threshold, stopping")
                
                best_clusters = clusters
                best_scores = scores
                break
            else:
                best_clusters = clusters
                best_scores = scores
                break
        
        
        stats = self._compute_stats(best_clusters)
        
        return {
            "model": self.model,
            "approach": "interactive_with_metrics_improved",
            "clusters": best_clusters,
            "statistics": stats,
            "metrics": best_scores,
            "iteration_history": iteration_history,
            "metadata": {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "iterations": iteration + 1,
            }
        }

    def _run_clustering(self, num_clusters, is_improvement, previous_scores, 
                       current_clusters=None, analysis_feedback=None):
        """Run clustering step"""
        if is_improvement and previous_scores and current_clusters:
            clusters_str = json.dumps(current_clusters, indent=2)
            
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
3. Do NOT return the exact same clustering.
4. Output the complete, improved clustering in SIMPLE TEXT FORMAT:

```
CLUSTER cluster_1:
node1
node2

CLUSTER cluster_2:
node3
node4
```

One node per line, blank line between clusters."""
        else:
            total_nodes = self.graph.number_of_nodes()
            
            if num_clusters is None or num_clusters <= 0:
                cluster_goal = f"""You will analyze the graph and decide the OPTIMAL number of clusters.

The graph has {total_nodes} nodes. Consider:
- Natural module boundaries
- Dependency patterns and cohesion
- Minimizing coupling between clusters
- Maximizing cohesion within clusters

YOU decide how many clusters makes sense - it could be 5, 10, 15, or more!"""
            else:
                cluster_goal = f"""Please analyze the dependency graph and create {num_clusters} microservice clusters.

CRITICAL: The graph has {total_nodes} nodes. You MUST assign ALL {total_nodes} nodes to clusters."""
            
            message = f"""{cluster_goal}

Steps:
1. Use get_graph_info() to understand the graph structure
2. Use get_all_nodes() to get the complete list of all {total_nodes} nodes
3. Use get_all_edges() to see all dependencies
4. Analyze the dependency patterns
5. Create clusters that minimize inter-cluster dependencies

IMPORTANT RULES:
- Every single node must appear in exactly ONE cluster
- Do not skip any nodes
- Verify your output includes all {total_nodes} nodes

Output your clustering in SIMPLE TEXT FORMAT (easier than JSON):
```
CLUSTER cluster_1:
node1
node2
node3

CLUSTER cluster_2:
node4
node5
node6
```

Each cluster starts with "CLUSTER cluster_X:" then list one node per line.
Put a blank line between clusters.

Start by calling get_all_nodes() to see the complete list of nodes you must cluster!"""
        
        try:
            if not self.verbose:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.user_proxy.initiate_chat(
                        self.clustering_agent,
                        message=message,
                        max_turns=20,
                    )
            else:
                self.user_proxy.initiate_chat(
                    self.clustering_agent,
                    message=message,
                    max_turns=20,
                )
            
            clusters = self.extract_clusters(self.clustering_agent)
            
            if clusters:
                all_nodes = set(self.graph.nodes())
                clustered_nodes = set()
                for nodes in clusters.values():
                    clustered_nodes.update(nodes)
                
                missing = all_nodes - clustered_nodes
                if missing:
                    print(f"{len(missing)} nodes not assigned, distributing by dependencies...")
                    clusters = self._distribute_missing_nodes(clusters, missing)
                
                extra_nodes = clustered_nodes - all_nodes
                if extra_nodes:
                    print(f"Removing {len(extra_nodes)} non-existent nodes from clusters")
                    for cluster_name in clusters:
                        clusters[cluster_name] = [n for n in clusters[cluster_name] if n in all_nodes]
            
            return clusters
            
        except Exception as e:
            print(f"Clustering error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _distribute_missing_nodes(self, clusters: Dict, missing: set) -> Dict:
        """Distribute missing nodes to clusters based on their dependencies"""
        node_to_cluster = {}
        for cname, nodes in clusters.items():
            for node in nodes:
                node_to_cluster[node] = cname
        
        for node in missing:
            cluster_connections = {}
            
            for target in self.graph.successors(node):
                if target in node_to_cluster:
                    cluster = node_to_cluster[target]
                    cluster_connections[cluster] = cluster_connections.get(cluster, 0) + 1
            
            for source in self.graph.predecessors(node):
                if source in node_to_cluster:
                    cluster = node_to_cluster[source]
                    cluster_connections[cluster] = cluster_connections.get(cluster, 0) + 1
            
            if cluster_connections:
                best_cluster = max(cluster_connections, key=cluster_connections.get)
                clusters[best_cluster].append(node)
                node_to_cluster[node] = best_cluster
            else:
                smallest_cluster = min(clusters.keys(), key=lambda c: len(clusters[c]))
                clusters[smallest_cluster].append(node)
                node_to_cluster[node] = smallest_cluster
        
        return clusters
    
    def _evaluate_metrics(self, clusters):
        if not self.metrics_calculator:
            return None
        try:
            result = self.metrics_calculator.calculate_metrics(clusters, self.reference_rsf_path)
            turbomq = result.get('turbomq')
            mojo = result.get('mojo_fm')
            
            # Format output safely handling None values
            turbomq_str = f"{turbomq:.4f}" if turbomq is not None else "N/A"
            mojo_str = f"{mojo:.2f}" if mojo is not None else "N/A"
            print(f"✓ Metrics: TurboMQ={turbomq_str}, MoJo={mojo_str}")
            return result
        except Exception as e:
            print(f"Metrics evaluation error: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return None
    
    def _analyze_scores(self, scores, clusters):
        """Have Inspector agent analyze the scores"""
        if not self.metrics_agent:
            return None
        
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

Output JSON: {{"specific_orders": ["Move X to cluster_Y", ...], "analysis": "..."}}"""
        
        try:
            if not self.verbose:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.metrics_proxy.initiate_chat(
                        self.metrics_agent,
                        message=message,
                        max_turns=2,  
                    )
            else:
                self.metrics_proxy.initiate_chat(
                    self.metrics_agent,
                    message=message,
                    max_turns=2,  
                )
            
            for conv_id, messages in self.metrics_agent.chat_messages.items():
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if "specific_orders" in content or "Move" in content:
                            return content
                        return content
        except Exception as e:
            print(f"Analysis error: {e}")
            return None
    
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

def main():
    parser = argparse.ArgumentParser(description="Improved Interactive analyzer with Inspector Agent")
    parser.add_argument("graph_file", help="Path to pickled graph file")
    parser.add_argument("--dependency-rsf", help="Path to dependency RSF file (for metrics)")
    parser.add_argument("--reference-rsf", help="Path to reference clustering RSF (for MoJo-FM)")
    parser.add_argument("--model", default="gpt-oss:120b-cloud", help="Ollama model (use :cloud suffix for cloud models)")
    parser.add_argument("--clusters", type=int, default=None, help="Number of clusters (leave empty to let LLM decide)")
    parser.add_argument("--iterations", type=int, default=1, help="Max improvement iterations")
    parser.add_argument("--output", default="clusters_with_metrics.json", help="Output file")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host (ignored for cloud models)")
    parser.add_argument("--api-key", help="API key for cloud Ollama (optional, uses OLLAMA_API_KEY env var if not provided)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output for debugging")

    args = parser.parse_args()
    
    
    api_key = args.api_key or os.environ.get("OLLAMA_API_KEY")

    analyzer = InteractiveAnalyzerImproved(
        ollama_host=args.host,
        model=args.model,
        dependency_rsf_path=args.dependency_rsf,
        reference_rsf_path=args.reference_rsf,
        verbose=args.verbose,
        api_key=api_key,
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
            if first.get('turbomq') and last.get('turbomq'):
                imp = last['turbomq'] - first['turbomq']
                print(f"TurboMQ Change: {imp:+.4f}")
    else:
        print("❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()