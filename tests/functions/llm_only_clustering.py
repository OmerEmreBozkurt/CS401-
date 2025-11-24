#!/usr/bin/env python3
"""
Pure LLM Clustering with Iterative Validation
Forces LLM to cluster ALL nodes through an iterative feedback loop
"""

import sys
import os
import pickle
import json
import requests
import networkx as nx
import time


class PureLLMClusterer:
    def __init__(self, ollama_host="http://localhost:11434"):
        self.ollama_host = ollama_host
        self.graph = None

    def load_graph(self, graph_file):
        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"✓ Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def query_llm(self, prompt, model, max_tokens=32000):
        """Query LLM with large output capacity"""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0,
                        "num_ctx": 65536,  # Large context
                        "num_predict": max_tokens,  # Allow long output
                        "stop": []  # Don't stop early
                    }
                },
                timeout=300  # 5 minutes
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Error: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None

    def create_initial_prompt(self, edges_tok, allowed_ids, num_clusters=10):
        """Create initial clustering prompt"""

        prompt = f"""You are a dependency graph clustering expert.

CRITICAL TASK: Cluster ALL {len(allowed_ids)} nodes into exactly {num_clusters} clusters.

STRICT REQUIREMENTS:
1. Output MUST include ALL {len(allowed_ids)} nodes
2. Each node appears in EXACTLY ONE cluster  
3. Use EXACT node IDs from the list below
4. Return valid JSON only (no markdown, no explanations outside JSON)

STRATEGY:
- Analyze dependency patterns
- Group tightly connected nodes together
- Minimize cross-cluster dependencies
- Distribute nodes across {num_clusters} clusters

OUTPUT FORMAT (this is the ONLY acceptable format):
{{
  "clusters": {{
    "cluster_1": ["N0", "N5", "N12", ...],
    "cluster_2": ["N1", "N8", "N15", ...],
    "cluster_3": ["N2", "N9", "N18", ...],
    ...
    "cluster_{num_clusters}": ["...", "..."]
  }},
  "verification": {{
    "total_nodes_assigned": <must be {len(allowed_ids)}>,
    "clusters_created": {num_clusters}
  }}
}}

ALL {len(allowed_ids)} NODE IDS (copy exactly):
{json.dumps(allowed_ids, indent=2)}

DEPENDENCY EDGES (source->target):
{chr(10).join(edges_tok)}

BEGIN YOUR RESPONSE NOW. Include all {len(allowed_ids)} nodes:"""

        return prompt

    def create_completion_prompt(self, edges_tok, all_ids, existing_clusters, missing_ids):
        """Create prompt to assign missing nodes"""

        # Show existing cluster sizes
        cluster_summary = {k: len(v) for k, v in existing_clusters.items()}

        prompt = f"""URGENT: You previously clustered nodes but missed {len(missing_ids)} nodes.

TASK: Assign these {len(missing_ids)} missing nodes to the existing clusters.

EXISTING CLUSTERS (with current sizes):
{json.dumps(cluster_summary, indent=2)}

MISSING NODES THAT MUST BE ASSIGNED:
{json.dumps(sorted(missing_ids), indent=2)}

EDGES involving missing nodes:
{chr(10).join([e for e in edges_tok if any(n in e for n in missing_ids)][:200])}

INSTRUCTIONS:
1. For each missing node, choose the most appropriate existing cluster
2. Consider which cluster members it has edges to/from
3. Distribute evenly if no clear preference

OUTPUT FORMAT:
{{
  "assignments": {{
    "cluster_1": ["N123", "N145", ...],
    "cluster_2": ["N67", "N89", ...],
    ...
  }}
}}

ALL {len(missing_ids)} missing nodes MUST appear in your response:"""

        return prompt

    def iterative_cluster(self, model="llama3.1:8b", max_iterations=5, num_clusters=10):
        """
        Iteratively query LLM until all nodes are assigned
        """
        G = self.graph
        nodes = set(G.nodes())

        # Build token maps
        id2tok = {nid: f"N{idx}" for idx, nid in enumerate(sorted(nodes))}
        tok2id = {v: k for k, v in id2tok.items()}

        edges_tok = [f"{id2tok[u]}->{id2tok[v]}" for u, v in G.edges()]
        allowed_ids = sorted(id2tok.values())

        print(f"\n🎯 Goal: Cluster ALL {len(nodes)} nodes using ONLY LLM")
        print(f"   Strategy: Iterative refinement until 100% coverage")

        clusters_real = {}
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'=' * 70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'=' * 70}")

            if iteration == 1:
                # Initial clustering attempt
                print("Sending initial clustering request to LLM...")
                prompt = self.create_initial_prompt(edges_tok, allowed_ids, num_clusters)

            else:
                # Completion attempt for missing nodes
                assigned = {n for members in clusters_real.values() for n in members}
                missing = nodes - assigned
                missing_toks = [id2tok[n] for n in missing]

                print(f"LLM missed {len(missing)} nodes. Requesting assignment...")
                prompt = self.create_completion_prompt(
                    edges_tok, allowed_ids,
                    {k: [id2tok[n] for n in v] for k, v in clusters_real.items()},
                    missing_toks
                )

            # Query LLM
            start_time = time.time()
            raw_response = self.query_llm(prompt, model, max_tokens=32000)
            elapsed = time.time() - start_time

            if not raw_response:
                print(f"❌ No response from LLM")
                break

            print(f"✓ Response received in {elapsed:.1f}s ({len(raw_response)} chars)")

            # Parse response
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError:
                print("⚠ Invalid JSON, attempting to extract...")
                import re
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except:
                        print("❌ Could not parse response")
                        break
                else:
                    print("❌ No JSON found in response")
                    break

            # Extract clusters
            if iteration == 1:
                if "clusters" in data:
                    clusters_tok = data["clusters"]
                    clusters_real = {}

                    for cname, members in clusters_tok.items():
                        if isinstance(members, list):
                            mapped = [tok2id.get(tok) for tok in members if tok in tok2id]
                            if mapped:
                                clusters_real[cname] = mapped
            else:
                # Merge completion assignments
                if "assignments" in data:
                    for cname, new_members in data["assignments"].items():
                        if cname in clusters_real:
                            mapped = [tok2id.get(tok) for tok in new_members if tok in tok2id]
                            clusters_real[cname].extend(mapped)

            # Check coverage
            assigned = {n for members in clusters_real.values() for n in members}
            missing = nodes - assigned
            coverage = len(assigned) / len(nodes) * 100

            print(f"\n📊 Coverage: {len(assigned)}/{len(nodes)} ({coverage:.1f}%)")

            if len(missing) == 0:
                print("🎉 SUCCESS! All nodes assigned!")
                break
            else:
                print(f"⚠ Still missing {len(missing)} nodes")

                if iteration >= max_iterations:
                    print(f"\n❌ Max iterations reached. Missing nodes: {list(missing)[:10]}")

                    # Final attempt: assign remaining to largest cluster
                    if clusters_real:
                        largest = max(clusters_real.keys(), key=lambda k: len(clusters_real[k]))
                        print(f"   Auto-assigning {len(missing)} nodes to {largest} (fallback)")
                        clusters_real[largest].extend(list(missing))
                    break

        # Compute statistics
        stats = self.compute_stats(clusters_real)

        return {
            "method": "pure_llm_iterative",
            "model": model,
            "iterations": iteration,
            "clusters": clusters_real,
            "statistics": stats
        }

    def compute_stats(self, clusters):
        """Compute clustering quality"""
        node_to_cluster = {}
        for cname, members in clusters.items():
            for node in members:
                node_to_cluster[node] = cname

        intra = 0
        inter = 0

        for u, v in self.graph.edges():
            if u in node_to_cluster and v in node_to_cluster:
                if node_to_cluster[u] == node_to_cluster[v]:
                    intra += 1
                else:
                    inter += 1

        total = intra + inter

        return {
            "num_clusters": len(clusters),
            "cluster_sizes": {k: len(v) for k, v in clusters.items()},
            "intra_cluster_edges": intra,
            "inter_cluster_edges": inter,
            "intra_ratio": intra / total if total > 0 else 0,
            "inter_ratio": inter / total if total > 0 else 0
        }

    def save_results(self, result, output_file):
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        stats = result["statistics"]
        print(f"\n{'=' * 70}")
        print("FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Clusters: {stats['num_clusters']}")
        print(f"Intra-cluster edges: {stats['intra_cluster_edges']}")
        print(f"Inter-cluster edges: {stats['inter_cluster_edges']}")
        print(f"Quality (intra-ratio): {stats['intra_ratio']:.2%}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pure LLM clustering with iteration")
    parser.add_argument("graph_file", help="Pickled graph file")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max iterations")
    parser.add_argument("--output", default="llm_clusters.json")

    args = parser.parse_args()

    clusterer = PureLLMClusterer()
    clusterer.load_graph(args.graph_file)

    result = clusterer.iterative_cluster(
        model=args.model,
        max_iterations=args.max_iterations,
        num_clusters=args.clusters
    )

    clusterer.save_results(result, args.output)


if __name__ == "__main__":
    main()