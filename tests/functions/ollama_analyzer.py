#!/usr/bin/env python3
"""
Ollama Graph Analyzer - Strict Mode (No Auto-Assignment)
Only uses LLM's clustering output without correction
"""

import sys
import os
import pickle
import json
import requests
import networkx as nx
from typing import Dict, List, Tuple, Set
import time


class OllamaGraphAnalyzer:
    def __init__(self, ollama_host="http://localhost:11434", verbose=True):
        self.ollama_host = ollama_host
        self.graph = None
        self.verbose = verbose
        self.check_ollama_connection()

    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=2)
            if response.status_code == 200:
                if self.verbose:
                    print(f"✓ Connected to Ollama at {self.ollama_host}")
                return True
        except:
            pass

        print(f"⚠ Warning: Cannot connect to Ollama at {self.ollama_host}")
        print("  Make sure Ollama is running: 'ollama serve'")
        return False

    def load_graph(self, graph_file):
        """Load the compact graph"""
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")

        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)

        print(f"✓ Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        if self.graph.number_of_edges() > 50000:
            print(f"⚠ Warning: Graph has {self.graph.number_of_edges()} edges, which may be too large for some models")
            print("  Consider using a smaller model context or sampling the graph")

    def get_available_models(self):
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        except:
            pass
        return []

    def estimate_token_count(self, edges_list):
        """Rough estimate of token count for the edge list"""
        return len(edges_list) * 3

    def build_token_maps(self, nodes):
        """Map raw node IDs to safe tokens and back"""
        nodes_sorted = sorted(nodes)
        id2tok = {nid: f"N{idx}" for idx, nid in enumerate(nodes_sorted)}
        tok2id = {v: k for k, v in id2tok.items()}
        return id2tok, tok2id

    def prepare_graph_summary(self):
        """Prepare a human-readable summary of the graph"""
        G = self.graph

        summary = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'components': nx.number_weakly_connected_components(G)
        }

        in_degrees = [G.in_degree(n) for n in G.nodes()]
        out_degrees = [G.out_degree(n) for n in G.nodes()]

        summary['degree_stats'] = {
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0
        }

        node_info = [(n, G.in_degree(n), G.out_degree(n)) for n in G.nodes()]
        by_in = sorted(node_info, key=lambda x: x[1], reverse=True)[:10]
        by_out = sorted(node_info, key=lambda x: x[2], reverse=True)[:10]

        summary['top_nodes'] = {
            'most_depended_on': [(n, ind) for n, ind, _ in by_in],
            'most_dependencies': [(n, outd) for n, _, outd in by_out]
        }

        return summary

    def create_clustering_prompt(self, edges_tok, allowed_ids, num_clusters_hint=None):
        """Create the clustering prompt for Ollama"""
        if num_clusters_hint:
            cluster_instruction = f"- Create exactly {num_clusters_hint} clusters."
        else:
            cluster_instruction = "Choose 5 to 15 clusters based on the graph structure."

        prompt = f"""You are a clustering engine for dependency graphs.

TASK: Analyze the dependency edges and group nodes into clusters that minimize cross-cluster dependencies.

RULES:
- IDs are opaque tokens (N0, N1, etc). Do NOT modify them.
- Return ONLY valid JSON with no markdown or comments.
- Use ONLY the IDs from ALLOWED_NODE_IDS.
- Every ID MUST appear in exactly one cluster.
{cluster_instruction}
- Optimize for high cohesion within clusters and low coupling between clusters.



OUTPUT FORMAT:
{{
  "analysis": {{
    "recommended_clusters": <number>,
    "rationale": "<brief explanation>"
  }},
  "clusters": {{
    "cluster_1": ["N0", "N1", ...],
    "cluster_2": ["N2", "N3", ...],
    ...
  }}
}}

ALLOWED_NODE_IDS: {json.dumps(allowed_ids)}

DEPENDENCY EDGES (source->target):
{chr(10).join(edges_tok)}"""

        return prompt

    def query_ollama_with_retry(self, model, prompt, max_retries=2):
        """Query Ollama with retry logic"""
        for attempt in range(max_retries):
            try:
                if self.verbose and attempt > 0:
                    print(f"  Retry attempt {attempt + 1}/{max_retries}")

                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0,
                            "num_ctx": 65536,
                            "num_predict": 16384
                        }
                    },
                    timeout=300
                )

                if response.status_code == 200:
                    return response.json().get("response", "")
                elif response.status_code == 404:
                    raise ValueError(f"Model '{model}' not found")
                else:
                    print(f"  Error: HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                print(f"  Timeout on attempt {attempt + 1}")
            except Exception as e:
                print(f"  Error on attempt {attempt + 1}: {e}")

        return None

    def query_ollama(self, model="llama3.2:3b", sample_size=None):
        """
        Send graph to Ollama for clustering analysis
        STRICT MODE: Does not auto-assign missing nodes
        """
        G = self.graph

        available_models = self.get_available_models()
        if available_models and model not in available_models:
            print(f"\n⚠ Model '{model}' is not installed.")
            print(f"Available models: {', '.join(available_models)}")
            print(f"Install with: ollama pull {model}")
            return None

        if sample_size and sample_size < G.number_of_nodes():
            print(f"Sampling {sample_size} nodes from graph...")
            node_degrees = [(n, G.degree(n)) for n in G.nodes()]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            sampled_nodes = [n for n, _ in node_degrees[:sample_size]]
            G = G.subgraph(sampled_nodes).copy()
            print(f"  Using subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        nodes = set(G.nodes())
        id2tok, tok2id = self.build_token_maps(nodes)

        edges_tok = [f"{id2tok[u]}->{id2tok[v]}" for u, v in G.edges()]
        allowed_ids = sorted(id2tok.values())

        estimated_tokens = self.estimate_token_count(edges_tok)
        print(f"\n📊 Graph statistics:")
        print(f"  - Nodes: {len(nodes)}")
        print(f"  - Edges: {len(edges_tok)}")
        print(f"  - Estimated tokens: ~{estimated_tokens}")

        if estimated_tokens > 20000:
            print(f"⚠ Warning: Large token count may exceed model context")
            print(f"  Consider using sample_size parameter or a model with larger context")

        prompt = self.create_clustering_prompt(edges_tok, allowed_ids)

        print(f"\n🤖 Sending to Ollama model: {model}")
        print(f"  This may take a few minutes for large graphs...")

        start_time = time.time()
        raw_response = self.query_ollama_with_retry(model, prompt)

        if not raw_response:
            return json.dumps({
                "error": "Failed to get response from Ollama after retries",
                "suggestion": "Try a different model or check Ollama logs"
            }, indent=2)

        elapsed = time.time() - start_time
        print(f"  ✓ Response received in {elapsed:.1f} seconds")

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            print("⚠ Warning: Response is not valid JSON, attempting to fix...")
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except:
                    data = {"error": "Could not parse response as JSON", "raw": raw_response[:500]}
            else:
                data = {"error": "No JSON found in response", "raw": raw_response[:500]}

        # Process clusters WITHOUT auto-assignment
        if "clusters" in data:
            clusters_tok = data.get("clusters", {})
            clusters_real = {}

            for cname, members in clusters_tok.items():
                if isinstance(members, list):
                    mapped = []
                    for tok in members:
                        real_id = tok2id.get(tok)
                        if real_id:
                            mapped.append(real_id)
                    clusters_real[cname] = mapped

            # STRICT: Do NOT auto-assign - only validate
            validation_result = self.validate_clusters(nodes, clusters_real)

            # Print warning if nodes are missing
            assigned = {n for members in clusters_real.values() for n in members}
            missing = nodes - assigned
            if missing:
                print(f"\n⚠ WARNING: LLM failed to assign {len(missing)} nodes to clusters")
                print(f"  Missing nodes will NOT be included in output (strict mode)")
                print(f"  Use --auto-assign flag if you want automatic assignment")

            result = {
                "analysis": data.get("analysis", {
                    "recommended_clusters": len(clusters_real),
                    "model_used": model,
                    "processing_time": f"{elapsed:.1f}s"
                }),
                "validation": validation_result,
                "clusters": clusters_real,
                "statistics": self.compute_cluster_stats(G, clusters_real)
            }

            return json.dumps(result, indent=2)
        else:
            return json.dumps(data, indent=2)

    def validate_clusters(self, all_nodes, clusters):
        """Validate that clusters cover all nodes exactly once"""
        seen = set()
        issues = []

        for cname, members in clusters.items():
            for node in members:
                if node not in all_nodes:
                    issues.append(f"Unknown node '{node}' in {cname}")
                if node in seen:
                    issues.append(f"Duplicate node '{node}'")
                seen.add(node)

        missing = all_nodes - seen
        if missing:
            issues.append(f"{len(missing)} nodes not assigned to any cluster")

        return {
            "valid": len(issues) == 0,
            "coverage": f"{len(seen)}/{len(all_nodes)}",
            "issues": issues[:5] if issues else []
        }

    def compute_cluster_stats(self, G, clusters):
        """Compute statistics about the clustering quality"""
        stats = {
            "num_clusters": len(clusters),
            "cluster_sizes": {},
            "modularity": 0,
            "inter_cluster_edges": 0,
            "intra_cluster_edges": 0
        }

        node_to_cluster = {}
        for cname, members in clusters.items():
            stats["cluster_sizes"][cname] = len(members)
            for node in members:
                node_to_cluster[node] = cname

        for u, v in G.edges():
            if u in node_to_cluster and v in node_to_cluster:
                if node_to_cluster[u] == node_to_cluster[v]:
                    stats["intra_cluster_edges"] += 1
                else:
                    stats["inter_cluster_edges"] += 1

        total_edges = stats["intra_cluster_edges"] + stats["inter_cluster_edges"]
        if total_edges > 0:
            stats["clustering_quality"] = {
                "intra_cluster_ratio": stats["intra_cluster_edges"] / total_edges,
                "inter_cluster_ratio": stats["inter_cluster_edges"] / total_edges
            }

        return stats

    def save_analysis(self, analysis, output_file):
        """Save analysis results to file"""
        with open(output_file, 'w') as f:
            f.write("# Dependency Graph Clustering Analysis\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(analysis)
        print(f"\n✓ Analysis saved to {output_file}")

    def export_clusters_for_visualization(self, clusters, output_file):
        """Export clusters in a format suitable for visualization tools"""
        viz_data = {
            "nodes": [],
            "clusters": {}
        }

        node_to_cluster = {}
        for cname, members in clusters.items():
            viz_data["clusters"][cname] = members
            for node in members:
                node_to_cluster[node] = cname

        for node in self.graph.nodes():
            viz_data["nodes"].append({
                "id": node,
                "cluster": node_to_cluster.get(node, "unassigned"),
                "in_degree": self.graph.in_degree(node),
                "out_degree": self.graph.out_degree(node)
            })

        with open(output_file, 'w') as f:
            json.dump(viz_data, f, indent=2)

        print(f"✓ Visualization data exported to {output_file}")


def pick_best_model(ollama_host="http://localhost:11434"):
    """Pick the best available model for clustering"""
    preferred = [
        "llama3.2:3b",
        "llama3.1:8b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "mistral:7b",
        "deepseek-r1:8b"
    ]

    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available = {m["name"] for m in models}

            for model in preferred:
                if model in available:
                    return model

            if models:
                return models[0]["name"]
    except:
        pass

    return "llama3.2:3b"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze dependency graph with Ollama (Strict Mode)")
    parser.add_argument("graph_file", help="Path to pickled graph file (e.g., compact_graph.pkl)")
    parser.add_argument("--model", help="Ollama model to use", default=None)
    parser.add_argument("--sample", type=int, help="Sample N most connected nodes", default=None)
    parser.add_argument("--output", help="Output file name", default="clusters_analysis.json")
    parser.add_argument("--host", help="Ollama host URL", default="http://localhost:11434")

    args = parser.parse_args()

    analyzer = OllamaGraphAnalyzer(ollama_host=args.host)

    try:
        analyzer.load_graph(args.graph_file)
    except FileNotFoundError:
        print(f"❌ Error: Graph file '{args.graph_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GRAPH SUMMARY")
    print("=" * 70)
    summary = analyzer.prepare_graph_summary()
    print(f"Nodes: {summary['nodes']}")
    print(f"Edges: {summary['edges']}")
    print(f"Density: {summary['density']:.6f}")
    print(f"Is DAG: {summary['is_dag']}")
    print(f"Components: {summary['components']}")

    if not args.model:
        args.model = pick_best_model(args.host)
        print(f"\n✓ Auto-selected model: {args.model}")

    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS (STRICT MODE)")
    print("=" * 70)
    print("Note: Missing nodes will NOT be auto-assigned")

    result = analyzer.query_ollama(model=args.model, sample_size=args.sample)

    if result:
        analyzer.save_analysis(result, args.output)

        try:
            data = json.loads(result)
            if "clusters" in data and "error" not in data:
                print("\n" + "=" * 70)
                print("RESULTS SUMMARY")
                print("=" * 70)
                print(f"Number of clusters: {len(data['clusters'])}")

                if "validation" in data:
                    val = data["validation"]
                    print(f"Validation: {'✓ PASS' if val['valid'] else '❌ FAIL'}")
                    print(f"Coverage: {val['coverage']}")
                    if val['issues']:
                        print("Issues:")
                        for issue in val['issues']:
                            print(f"  - {issue}")

                if "statistics" in data:
                    stats = data["statistics"]
                    print(f"Intra-cluster edges: {stats.get('intra_cluster_edges', 'N/A')}")
                    print(f"Inter-cluster edges: {stats.get('inter_cluster_edges', 'N/A')}")

                    if "clustering_quality" in stats:
                        quality = stats["clustering_quality"]
                        print(f"Clustering quality:")
                        print(f"  - Intra-cluster ratio: {quality.get('intra_cluster_ratio', 0):.2%}")
                        print(f"  - Inter-cluster ratio: {quality.get('inter_cluster_ratio', 0):.2%}")

                viz_file = args.output.replace('.json', '_viz.json')
                analyzer.export_clusters_for_visualization(data['clusters'], viz_file)
        except:
            pass
    else:
        print("❌ Failed to get clustering results")
        sys.exit(1)


if __name__ == "__main__":
    main()