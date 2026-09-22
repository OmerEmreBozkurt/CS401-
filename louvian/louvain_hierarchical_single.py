#!/usr/bin/env python3
"""
Single-agent Louvain + LLM clustering pipeline.

Purpose
-------
This file is intended to be the SINGLE-AGENT alternative to a multi-agent
hierarchical_analyzer.py pipeline.

Flow:
1. Load a graph from a .pkl NetworkX graph OR build it from dependency RSF.
2. Apply Hierarchical Louvain with adjustable --louvain-levels.
3. Convert Louvain communities into super-nodes.
4. Send the super-node graph ONCE to one LLM.
5. Expand the LLM clustering back to original nodes.
6. Write the final clustering as RSF.
7. Print:
   - cluster count
   - super-node count
   - TurboMQ, if --turbomq-jar is provided
   - MoJo-FM, if --reference-rsf is provided

Expected RSF formats
--------------------
Dependency RSF:
    depends source target

Clustering RSF:
    contain cluster_name node_name

Example command
---------------
python3 louvain_hierarchical_single.py test_results/bash-g.pkl \
  --dependency-rsf dataset/bash/bash-dependency.rsf \
  --reference-rsf dataset/bash/bash-clustering.rsf \
  --louvain-resolution 1.0 \
  --louvain-levels 1 \
  --model qwen3-coder:480b-cloud \
  --iterations 1 \
  --output test_results/bash_single_louvain_L1_qwen3coder.json \
  --output-rsf test_results/bash_single_louvain_L1_qwen3coder.rsf \
  --turbomq-jar turbomq.jar

Notes
-----
- --iterations is kept for command compatibility, but this single-agent file
  sends the graph to the LLM only once. It does NOT run multi-agent feedback.
- TurboMQ is usually computed by an external Java tool, so this script calls
  the jar when --turbomq-jar is supplied.
"""

import argparse
import json
import os
import pickle
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx is required. Install with: pip install networkx", file=sys.stderr)
    raise

try:
    import requests
except ImportError:
    requests = None


# ============================================================================
# Louvain graph implementation
# ============================================================================

class LouvainGraph:
    def __init__(self):
        self.adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.node_weights: Dict[str, float] = defaultdict(float)
        self.self_loops: Dict[str, float] = defaultdict(float)
        self.total_weight: float = 0.0
        self.nodes: Set[str] = set()

    def add_edge(self, u: str, v: str, weight: float = 1.0):
        u, v = str(u), str(v)
        self.nodes.add(u)
        self.nodes.add(v)

        if u == v:
            self.self_loops[u] += weight
            self.node_weights[u] += weight
            self.total_weight += weight
        else:
            self.adj[u][v] += weight
            self.adj[v][u] += weight
            self.node_weights[u] += weight
            self.node_weights[v] += weight
            self.total_weight += weight

    def get_neighbors(self, node: str) -> Dict[str, float]:
        return dict(self.adj[node])

    def get_edge_weight(self, u: str, v: str) -> float:
        if u == v:
            return self.self_loops.get(u, 0.0)
        return self.adj[u].get(v, 0.0)

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def number_of_edges(self) -> int:
        count = sum(len(neighbors) for neighbors in self.adj.values()) // 2
        count += len(self.self_loops)
        return count

    @classmethod
    def from_networkx(cls, nx_graph) -> "LouvainGraph":
        lg = cls()
        for u, v, data in nx_graph.edges(data=True):
            weight = data.get("weight", 1.0)
            lg.add_edge(str(u), str(v), float(weight))

        for node in nx_graph.nodes():
            node = str(node)
            lg.nodes.add(node)
            lg.node_weights.setdefault(node, 0.0)

        return lg


class Community:
    def __init__(self):
        self.nodes: Set[str] = set()
        self.total_weight: float = 0.0
        self.internal_weight: float = 0.0

    def add_node(self, node: str, node_weight: float, internal_edges: float = 0.0):
        self.nodes.add(node)
        self.total_weight += node_weight
        self.internal_weight += internal_edges

    def remove_node(self, node: str, node_weight: float, internal_edges: float = 0.0):
        self.nodes.discard(node)
        self.total_weight -= node_weight
        self.internal_weight -= internal_edges


class LouvainAlgorithm:
    def __init__(
        self,
        resolution: float = 1.0,
        min_modularity_gain: float = 1e-7,
        max_passes: int = 10,
        random_seed: Optional[int] = None,
    ):
        self.resolution = resolution
        self.min_modularity_gain = min_modularity_gain
        self.max_passes = max_passes
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)

    def _modularity_gain(
        self,
        graph: LouvainGraph,
        node: str,
        community: Community,
        node_to_comm_weight: float,
    ) -> float:
        m = graph.total_weight
        if m == 0:
            return 0.0

        k_i = graph.node_weights[node]
        sigma_tot = community.total_weight
        k_i_in = node_to_comm_weight

        return k_i_in / m - (sigma_tot * k_i / (2 * m * m)) * self.resolution

    def _get_node_to_community_weight(
        self, graph: LouvainGraph, node: str, community: Community
    ) -> float:
        total = 0.0
        for neighbor, weight in graph.get_neighbors(node).items():
            if neighbor in community.nodes and neighbor != node:
                total += weight
        return total

    def _one_level(self, graph: LouvainGraph) -> Tuple[Dict[str, int], bool]:
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

        # This already means: continue while better clustering is found,
        # but with a safety cap of max_passes.
        while improved and pass_count < self.max_passes:
            improved = False
            pass_count += 1

            nodes = list(graph.nodes)
            random.shuffle(nodes)

            for node in nodes:
                current_comm_id = node_to_community[node]
                current_comm = communities[current_comm_id]

                current_weight = self._get_node_to_community_weight(graph, node, current_comm)
                node_weight = graph.node_weights[node]
                self_loop = graph.self_loops.get(node, 0.0)

                current_comm.remove_node(node, node_weight, self_loop + current_weight)

                best_comm_id = current_comm_id
                best_gain = 0.0

                neighbor_comms: Set[int] = set()
                for neighbor in graph.get_neighbors(node):
                    neighbor_comms.add(node_to_community[neighbor])
                neighbor_comms.add(current_comm_id)

                for comm_id in neighbor_comms:
                    comm = communities[comm_id]
                    weight_to_comm = self._get_node_to_community_weight(graph, node, comm)
                    gain = self._modularity_gain(graph, node, comm, weight_to_comm)

                    if gain > best_gain + self.min_modularity_gain:
                        best_gain = gain
                        best_comm_id = comm_id

                best_comm = communities[best_comm_id]
                best_weight = self._get_node_to_community_weight(graph, node, best_comm)
                best_comm.add_node(node, node_weight, self_loop + best_weight)
                node_to_community[node] = best_comm_id

                if best_comm_id != current_comm_id:
                    improved = True
                    total_improvement = True

        unique_comms = set(node_to_community.values())
        comm_mapping = {old: new for new, old in enumerate(sorted(unique_comms))}
        node_to_community = {
            node: comm_mapping[comm] for node, comm in node_to_community.items()
        }

        return node_to_community, total_improvement

    def _aggregate(self, graph: LouvainGraph, node_to_community: Dict[str, int]) -> LouvainGraph:
        new_graph = LouvainGraph()
        comm_names = {comm: f"comm_{comm}" for comm in set(node_to_community.values())}

        for node in graph.nodes:
            node_comm_name = comm_names[node_to_community[node]]
            new_graph.nodes.add(node_comm_name)

            if node in graph.self_loops:
                new_graph.add_edge(node_comm_name, node_comm_name, graph.self_loops[node])

            for neighbor, weight in graph.get_neighbors(node).items():
                neighbor_comm_name = comm_names[node_to_community[neighbor]]
                if node <= neighbor:
                    new_graph.add_edge(node_comm_name, neighbor_comm_name, weight)

        return new_graph

    def run(self, graph: LouvainGraph, num_levels: Optional[int] = None) -> List[Dict[str, int]]:
        partitions: List[Dict[str, int]] = []
        current_graph = graph
        node_mapping: Dict[str, Set[str]] = {node: {node} for node in graph.nodes}

        level = 0
        while True:
            if num_levels is not None and level >= num_levels:
                break

            partition, improved = self._one_level(current_graph)

            if not improved and level > 0:
                break

            original_partition: Dict[str, int] = {}
            for current_node, comm_id in partition.items():
                for original_node in node_mapping.get(current_node, {current_node}):
                    original_partition[original_node] = comm_id

            partitions.append(original_partition)

            num_communities = len(set(partition.values()))
            if num_communities == current_graph.number_of_nodes():
                break
            if num_communities == 1:
                break

            new_graph = self._aggregate(current_graph, partition)

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


class HierarchicalLouvain:
    def __init__(
        self,
        graph,
        num_coarsening_levels: int = 1,
        resolution: float = 1.0,
        random_seed: int = 42,
    ):
        self.original_graph = graph
        self.num_levels = num_coarsening_levels
        self.resolution = resolution
        self.random_seed = random_seed
        self.louvain_graph = LouvainGraph.from_networkx(graph)

        self.partitions: List[Dict[str, int]] = []
        self.super_node_to_original: Dict[str, List[str]] = {}
        self.coarse_edges: List[Dict[str, Any]] = []

    def coarsen(self) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
        print(f"\n📊 Running Louvain coarsening ({self.num_levels} levels)...")
        print(
            f"   Original graph: {self.louvain_graph.number_of_nodes()} nodes, "
            f"{self.louvain_graph.number_of_edges()} edges"
        )

        louvain = LouvainAlgorithm(
            resolution=self.resolution,
            max_passes=10,
            random_seed=self.random_seed,
        )

        self.partitions = louvain.run(self.louvain_graph, num_levels=self.num_levels)

        if not self.partitions:
            print("   ⚠ Louvain produced no partitions, using original graph")
            self.super_node_to_original = {node: [node] for node in self.louvain_graph.nodes}
            return self.super_node_to_original, []

        final_partition = self.partitions[-1]

        super_node_to_original: Dict[str, List[str]] = defaultdict(list)
        for node, comm_id in final_partition.items():
            super_node_to_original[f"super_{comm_id}"].append(node)

        self.super_node_to_original = dict(super_node_to_original)
        self._build_coarse_graph()

        print(f"   Coarsened graph: {len(self.super_node_to_original)} super-nodes")

        sizes = [len(nodes) for nodes in self.super_node_to_original.values()]
        if sizes:
            print(
                f"   Super-node sizes: min={min(sizes)}, max={max(sizes)}, "
                f"avg={sum(sizes) / len(sizes):.1f}"
            )

        return self.super_node_to_original, self.coarse_edges

    def _build_coarse_graph(self):
        node_to_super = {}
        for super_node, original_nodes in self.super_node_to_original.items():
            for node in original_nodes:
                node_to_super[str(node)] = super_node

        edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)

        for u, v in self.original_graph.edges():
            u_str, v_str = str(u), str(v)
            super_u = node_to_super.get(u_str, u_str)
            super_v = node_to_super.get(v_str, v_str)

            edge_key = tuple(sorted([super_u, super_v]))
            edge_weights[edge_key] += 1.0

        self.coarse_edges = [
            {"source": u, "target": v, "weight": w}
            for (u, v), w in sorted(edge_weights.items())
        ]

    def expand_clustering(self, super_node_clustering: Dict[str, List[str]]) -> Dict[str, List[str]]:
        expanded_clustering: Dict[str, List[str]] = {}

        for cluster_name, super_nodes in super_node_clustering.items():
            expanded_nodes: List[str] = []
            for super_node in super_nodes:
                original_nodes = self.super_node_to_original.get(super_node, [super_node])
                expanded_nodes.extend(original_nodes)

            expanded_clustering[cluster_name] = sorted(set(map(str, expanded_nodes)))

        return expanded_clustering


# ============================================================================
# Input / output helpers
# ============================================================================

def load_graph(graph_file: Optional[str], dependency_rsf: Optional[str]):
    if graph_file:
        path = Path(graph_file)
        if path.exists():
            if path.suffix == ".pkl":
                with path.open("rb") as f:
                    graph = pickle.load(f)
                print(f"Loaded graph from pickle: {path}")
                return graph
            raise ValueError(f"Unsupported graph file format: {path.suffix}")

    if dependency_rsf:
        print(f"Building graph from dependency RSF: {dependency_rsf}")
        return build_graph_from_dependency_rsf(dependency_rsf)

    raise ValueError("Provide either a graph_file positional argument or --dependency-rsf.")


def build_graph_from_dependency_rsf(path: str):
    graph = nx.DiGraph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            relation, source, target = parts[0], parts[1], parts[2]
            if relation.lower() in {"depends", "depend", "dependency", "call", "use"}:
                graph.add_edge(str(source), str(target), weight=1.0)

    return graph


def write_clustering_rsf(clustering: Dict[str, List[str]], output_rsf: str):
    with open(output_rsf, "w", encoding="utf-8") as f:
        for cluster_name, nodes in sorted(clustering.items()):
            for node in sorted(nodes):
                f.write(f"contain {cluster_name} {node}\n")


def read_clustering_rsf(path: str) -> Dict[str, List[str]]:
    clusters: Dict[str, List[str]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            relation, cluster, node = parts[0], parts[1], parts[2]
            if relation.lower() in {"contain", "contains"}:
                clusters[str(cluster)].append(str(node))

    return dict(clusters)


# ============================================================================
# Single LLM call
# ============================================================================

def make_single_agent_prompt(
    super_node_to_original: Dict[str, List[str]],
    coarse_edges: List[Dict[str, Any]],
) -> str:
    compact_supernodes = {
        sn: {
            "size": len(nodes),
            "sample_nodes": nodes[:8],
        }
        for sn, nodes in sorted(super_node_to_original.items())
    }

    prompt_payload = {
        "super_nodes": compact_supernodes,
        "coarse_edges": coarse_edges,
    }

    return f"""
You are clustering a software dependency graph.

You are given:
1. Super-nodes produced by Louvain coarsening.
2. Weighted edges between those super-nodes.

Task:
Cluster the super-nodes into architecture-level clusters.

Important rules:
- Use each super-node exactly once.
- Do not invent new super-node names.
- Return ONLY valid JSON.
- JSON format must be:
  {{
    "cluster_0": ["super_0", "super_1"],
    "cluster_1": ["super_2"]
  }}

Graph data:
{json.dumps(prompt_payload, indent=2)}
""".strip()


def call_ollama_once(model: str, prompt: str, ollama_url: str, temperature: float = 0.0) -> str:
    if requests is None:
        raise RuntimeError("requests is required for Ollama calls. Install with: pip install requests")

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def extract_json_object(text: str) -> Dict[str, List[str]]:
    text = text.strip()

    # Remove markdown fences if the model used them.
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Extract first JSON object.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")

    data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object/dict.")

    cleaned: Dict[str, List[str]] = {}
    for cluster_name, super_nodes in data.items():
        if isinstance(super_nodes, list):
            cleaned[str(cluster_name)] = [str(x) for x in super_nodes]

    if not cleaned:
        raise ValueError("LLM JSON did not contain any cluster lists.")

    return cleaned


def validate_and_repair_supernode_clustering(
    clustering: Dict[str, List[str]],
    all_super_nodes: Set[str],
) -> Dict[str, List[str]]:
    repaired: Dict[str, List[str]] = defaultdict(list)
    used: Set[str] = set()

    for cluster_name, super_nodes in clustering.items():
        for sn in super_nodes:
            if sn in all_super_nodes and sn not in used:
                repaired[cluster_name].append(sn)
                used.add(sn)

    missing = sorted(all_super_nodes - used)
    if missing:
        repaired["cluster_missing_from_llm"].extend(missing)

    return {k: v for k, v in repaired.items() if v}


def fallback_louvain_supernode_clustering(super_node_to_original: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {f"cluster_{i}": [sn] for i, sn in enumerate(sorted(super_node_to_original.keys()))}


# ============================================================================
# Metrics
# ============================================================================

def run_turbomq(turbomq_jar: str, dependency_rsf: str, clustering_rsf: str) -> Optional[float]:
    if not turbomq_jar:
        return None

    cmd = ["java", "-jar", turbomq_jar, dependency_rsf, clustering_rsf]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        print("⚠ Java not found, cannot compute TurboMQ.")
        return None

    combined = (result.stdout or "") + "\n" + (result.stderr or "")

    if result.returncode != 0:
        print("⚠ TurboMQ command failed.")
        print(combined.strip())
        return None

    # Parse the last numeric value from output.
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", combined)
    if not numbers:
        print("⚠ TurboMQ ran, but no numeric score was found in output.")
        print(combined.strip())
        return None

    return float(numbers[-1])


def mojofm(predicted: Dict[str, List[str]], reference: Dict[str, List[str]]) -> Optional[float]:
    """
    Compute a practical MoJo-FM-style score.

    This uses a common formulation based on the minimum number of node moves
    needed to transform predicted clusters toward the reference clustering:

        moves = N - sum_for_each_predicted_cluster(max overlap with any reference cluster)
        max_moves = N - number_of_reference_clusters
        MoJo-FM = (1 - moves / max_moves) * 100

    If your course/project requires a specific external MoJo tool, prefer that
    tool for exact comparability.
    """
    pred_sets = [set(map(str, nodes)) for nodes in predicted.values() if nodes]
    ref_sets = [set(map(str, nodes)) for nodes in reference.values() if nodes]

    all_nodes = set().union(*pred_sets) if pred_sets else set()
    ref_nodes = set().union(*ref_sets) if ref_sets else set()
    common_nodes = all_nodes & ref_nodes

    if not common_nodes:
        return None

    pred_sets = [s & common_nodes for s in pred_sets if s & common_nodes]
    ref_sets = [s & common_nodes for s in ref_sets if s & common_nodes]

    n = len(common_nodes)
    if n == 0:
        return None

    preserved = 0
    for pred in pred_sets:
        preserved += max((len(pred & ref) for ref in ref_sets), default=0)

    moves = n - preserved
    max_moves = max(n - len(ref_sets), 1)

    score = (1.0 - (moves / max_moves)) * 100.0
    return max(0.0, min(100.0, score))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-agent Louvain hierarchical clustering pipeline."
    )

    parser.add_argument(
        "graph_file",
        nargs="?",
        help="Optional NetworkX graph pickle file. If omitted, --dependency-rsf is used.",
    )
    parser.add_argument("--dependency-rsf", required=True, help="Dependency RSF file.")
    parser.add_argument("--reference-rsf", help="Reference clustering RSF file for MoJo-FM.")
    parser.add_argument("--louvain-resolution", type=float, default=1.0)
    parser.add_argument("--louvain-levels", type=int, default=1)
    parser.add_argument("--model", default=None, help="LLM model name, e.g. qwen3-coder:480b-cloud")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Kept for compatibility. Single-agent mode sends to LLM once.",
    )
    parser.add_argument("--output", required=True, help="JSON output file.")
    parser.add_argument("--output-rsf", help="Final clustering RSF output file.")
    parser.add_argument("--turbomq-jar", help="Path to turbomq.jar.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Do not call LLM; use each super-node as one final cluster.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_rsf = args.output_rsf
    if not output_rsf:
        output_rsf = str(output_path.with_suffix(".rsf"))

    print("\n======================================================================")
    print("SINGLE-AGENT LOUVAIN + LLM PIPELINE")
    print("======================================================================")
    print(f"Louvain resolution: {args.louvain_resolution}")
    print(f"Louvain levels: {args.louvain_levels}")
    print(f"Model: {args.model if args.model else 'None'}")
    print(f"Output JSON: {args.output}")
    print(f"Output RSF: {output_rsf}")

    graph = load_graph(args.graph_file, args.dependency_rsf)

    hier = HierarchicalLouvain(
        graph,
        num_coarsening_levels=args.louvain_levels,
        resolution=args.louvain_resolution,
        random_seed=args.random_seed,
    )

    super_node_to_original, coarse_edges = hier.coarsen()

    all_super_nodes = set(super_node_to_original.keys())

    llm_response = None
    if args.skip_llm or not args.model:
        print("\n⚠ Skipping LLM call. Using Louvain super-nodes as final clusters.")
        supernode_clustering = fallback_louvain_supernode_clustering(super_node_to_original)
    else:
        print("\n🤖 Sending coarse graph to LLM once...")
        prompt = make_single_agent_prompt(super_node_to_original, coarse_edges)

        try:
            llm_response = call_ollama_once(args.model, prompt, args.ollama_url)
            raw_supernode_clustering = extract_json_object(llm_response)
            supernode_clustering = validate_and_repair_supernode_clustering(
                raw_supernode_clustering,
                all_super_nodes,
            )
        except Exception as exc:
            print(f"⚠ LLM step failed: {exc}")
            if hasattr(exc, "response") and exc.response is not None:
                print(f"   Details: {exc.response.text}")
            print("⚠ Falling back to Louvain super-nodes as final clusters.")
            supernode_clustering = fallback_louvain_supernode_clustering(super_node_to_original)

    final_clustering = hier.expand_clustering(supernode_clustering)

    write_clustering_rsf(final_clustering, output_rsf)

    reference_clustering = None
    mojo_score = None
    if args.reference_rsf:
        reference_clustering = read_clustering_rsf(args.reference_rsf)
        mojo_score = mojofm(final_clustering, reference_clustering)

    turbomq_score = None
    if args.turbomq_jar:
        turbomq_score = run_turbomq(args.turbomq_jar, args.dependency_rsf, output_rsf)

    result = {
        "mode": "single_agent_louvain_then_one_llm_call",
        "graph_file": args.graph_file,
        "dependency_rsf": args.dependency_rsf,
        "reference_rsf": args.reference_rsf,
        "louvain_resolution": args.louvain_resolution,
        "louvain_levels": args.louvain_levels,
        "model": args.model,
        "iterations_requested": args.iterations,
        "note": "--iterations is kept for compatibility; this script calls the LLM once.",
        "super_nodes_used": len(super_node_to_original),
        "clusters": len(final_clustering),
        "turbomq": turbomq_score,
        "mojofm": mojo_score,
        "super_node_clustering": supernode_clustering,
        "final_clustering": final_clustering,
        "coarse_edges": coarse_edges,
        "llm_response": llm_response,
        "output_rsf": output_rsf,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n======================================================================")
    print("FINAL RESULTS")
    print("======================================================================")
    print(f"Clusters: {len(final_clustering)}")
    print(f"Super-nodes used: {len(super_node_to_original)}")

    if turbomq_score is None:
        print("TurboMQ: N/A")
    else:
        print(f"TurboMQ: {turbomq_score}")

    if mojo_score is None:
        print("MoJo-FM: N/A")
    else:
        print(f"MoJo-FM: {mojo_score:.2f}")

    print(f"Saved JSON: {args.output}")
    print(f"Saved RSF: {output_rsf}")


if __name__ == "__main__":
    main()
