#!/usr/bin/env python3
"""
Simple Clustering Pipeline (Test Version)

A minimal pipeline that:
1. Loads a dependency graph
2. Runs Louvain coarsening to create super-nodes
3. Sends super-nodes directly to a single LLM
4. Gets clustering result back

No multi-agent structure, no iterations - just straightforward clustering.
"""

import sys
import os
import pickle
import json
import argparse
import re
from typing import Dict, List, Optional
from collections import defaultdict

import networkx as nx

# Import Louvain implementation (must be in same directory or PYTHONPATH)
from louvain_hierarchical import LouvainGraph, LouvainAlgorithm, HierarchicalLouvain


def load_graph(graph_path: str) -> nx.DiGraph:
    """Load a pickled NetworkX graph"""
    print(f"📂 Loading graph from: {graph_path}")
    
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    # Ensure it's a DiGraph
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)
    
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    
    return graph


def coarsen_graph(graph: nx.DiGraph, 
                  num_levels: int = 2, 
                  resolution: float = 1.0,
                  random_seed: int = 42) -> HierarchicalLouvain:
    
    
    hier = HierarchicalLouvain(
        graph, 
        num_coarsening_levels=num_levels,
        resolution=resolution,
        random_seed=random_seed
    )
    
    hier.coarsen()
    
    return hier


def build_llm_prompt(hier: HierarchicalLouvain, num_clusters: Optional[int] = None) -> str:
    """Build a prompt for the LLM with super-node information"""
    
    
    super_nodes_info = []
    for super_node, original_nodes in hier.super_node_to_original.items():
        super_nodes_info.append({
            "name": super_node,
            "size": len(original_nodes),
            "members": original_nodes[:10],  # First 10 as sample
            "total_members": len(original_nodes)
        })
    
    
    super_nodes_info.sort(key=lambda x: x["size"], reverse=True)
    
    
    edges = [e for e in hier.coarse_edges if e["source"] != e["target"]]
    edges.sort(key=lambda x: x["weight"], reverse=True)
    
    
    cluster_instruction = ""
    if num_clusters:
        cluster_instruction = f"Create exactly {num_clusters} clusters."
    else:
        cluster_instruction = "Create a reasonable number of clusters."
    
    prompt = f"""You are a software architect tasked with clustering code modules into microservices.

I have a dependency graph that has been pre-processed using the Louvain algorithm. 
The original nodes have been grouped into "super-nodes" based on their connectivity.

Your task: Cluster these super-nodes into logical microservice groups.

{cluster_instruction}

## SUPER-NODES ({len(super_nodes_info)} total):

"""
    
    for sn in super_nodes_info:
        members_str = ", ".join(sn["members"][:5])
        if sn["total_members"] > 5:
            members_str += f", ... ({sn['total_members']} total)"
        prompt += f"- {sn['name']} (size={sn['size']}): [{members_str}]\n"
    
    prompt += f"""

## EDGES BETWEEN SUPER-NODES ({len(edges)} edges):
(Higher weight = more connections between the super-nodes)

"""
    
    # Show top edges (limit to 50 for readability)
    for edge in edges[:50]:
        prompt += f"- {edge['source']} <-> {edge['target']} (weight={edge['weight']})\n"
    
    if len(edges) > 50:
        prompt += f"... and {len(edges) - 50} more edges\n"
    
    prompt += """

## OUTPUT FORMAT:

Respond with ONLY a JSON object in this exact format (no other text):

{
  "clusters": {
    "cluster_name_1": ["super_0", "super_3", "super_5"],
    "cluster_name_2": ["super_1", "super_2"],
    ...
  },
  "reasoning": "Brief explanation of your clustering logic"
}

IMPORTANT:
- Every super-node must be assigned to exactly one cluster
- Use descriptive cluster names if possible (e.g., "data_layer", "api_services")
- Group super-nodes that have high edge weights between them
"""
    
    return prompt


def call_ollama(prompt: str, model: str = "llama3:8b", host: str = "http://localhost:11434") -> str:
    """Call Ollama API to get LLM response"""
    import requests
    
    print(f"\n🤖 Calling {model} via Ollama...")
    
    url = f"{host}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  
            "num_predict": 4096
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Ollama at {host}")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        sys.exit(1)


def call_openai(prompt: str, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> str:
    """Call OpenAI API to get LLM response"""
    import requests
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    print(f"\n🤖 Calling {model} via OpenAI...")
    
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        sys.exit(1)


def parse_llm_response(response: str, hier: HierarchicalLouvain) -> Dict[str, List[str]]:
    """Parse LLM response and extract clustering"""
    
    print("\n📝 Parsing LLM response...")
    
    
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
        else:
            print("❌ Could not find JSON in LLM response")
            print("Raw response:")
            print(response[:1000])
            return {}
    
    try:
        data = json.loads(json_str)
        clusters = data.get("clusters", data)  # Handle both formats
        
        if "reasoning" in data:
            print(f"   LLM reasoning: {data['reasoning'][:200]}...")
        
        return clusters
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print("Attempted to parse:")
        print(json_str[:500])
        return {}


def expand_to_original_nodes(super_node_clusters: Dict[str, List[str]], 
                              hier: HierarchicalLouvain) -> Dict[str, List[str]]:
    """Expand super-node clustering back to original nodes"""
    
    print("\n🔄 Expanding clusters to original nodes...")
    
    expanded = hier.expand_clustering(super_node_clusters)
    
    for cluster_name, nodes in expanded.items():
        print(f"   {cluster_name}: {len(nodes)} nodes")
    
    return expanded


def compute_basic_metrics(clusters: Dict[str, List[str]], graph: nx.DiGraph) -> Dict:
    """Compute basic clustering metrics"""
    
    print("\n📊 Computing metrics...")
    
    node_to_cluster = {}
    for cname, nodes in clusters.items():
        for node in nodes:
            node_to_cluster[node] = cname
    
    intra_edges = 0
    inter_edges = 0
    
    for u, v in graph.edges():
        u_str, v_str = str(u), str(v)
        if u_str in node_to_cluster and v_str in node_to_cluster:
            if node_to_cluster[u_str] == node_to_cluster[v_str]:
                intra_edges += 1
            else:
                inter_edges += 1
    
    total_edges = intra_edges + inter_edges
    
    metrics = {
        "num_clusters": len(clusters),
        "cluster_sizes": {name: len(nodes) for name, nodes in clusters.items()},
        "intra_cluster_edges": intra_edges,
        "inter_cluster_edges": inter_edges,
        "cohesion": intra_edges / total_edges if total_edges > 0 else 0,
        "coupling": inter_edges / total_edges if total_edges > 0 else 0
    }
    
    print(f"   Clusters: {metrics['num_clusters']}")
    print(f"   Intra-cluster edges: {intra_edges} ({metrics['cohesion']*100:.1f}%)")
    print(f"   Inter-cluster edges: {inter_edges} ({metrics['coupling']*100:.1f}%)")
    
    return metrics


def run_pipeline(graph_path: str,
                 model: str = "llama3:8b",
                 backend: str = "ollama",
                 host: str = "http://localhost:11434",
                 num_clusters: Optional[int] = None,
                 louvain_levels: int = 2,
                 louvain_resolution: float = 1.0,
                 output_file: str = "simple_clusters.json",
                 api_key: Optional[str] = None) -> Dict:
    """
    Run the complete simple clustering pipeline.
    
    Parameters:
    - graph_path: Path to pickled graph file
    - model: LLM model name
    - backend: "ollama" or "openai"
    - host: Ollama host URL
    - num_clusters: Target number of clusters (None = let LLM decide)
    - louvain_levels: Number of Louvain coarsening levels
    - louvain_resolution: Louvain resolution parameter
    - output_file: Where to save results
    - api_key: OpenAI API key (if using openai backend)
    
    Returns:
    - Result dictionary with clusters and metrics
    """
    
    print("=" * 60)
    print("SIMPLE CLUSTERING PIPELINE")
    print("=" * 60)
    
    
    graph = load_graph(graph_path)
    
    
    hier = coarsen_graph(
        graph, 
        num_levels=louvain_levels,
        resolution=louvain_resolution
    )
    
    print(f"\n✅ Coarsening complete: {len(hier.super_node_to_original)} super-nodes")
    
    prompt = build_llm_prompt(hier, num_clusters)
    
   
    with open("debug_prompt.txt", "w") as f:
        f.write(prompt)
    print(f"\n💾 Prompt saved to debug_prompt.txt ({len(prompt)} chars)")
    
    
    if backend == "ollama":
        response = call_ollama(prompt, model, host)
    elif backend == "openai":
        response = call_openai(prompt, model, api_key)
    else:
        print(f"❌ Unknown backend: {backend}")
        sys.exit(1)
    
   
    with open("debug_response.txt", "w") as f:
        f.write(response)
    print(f"💾 Response saved to debug_response.txt ({len(response)} chars)")
    
    super_node_clusters = parse_llm_response(response, hier)
    
    if not super_node_clusters:
        print("❌ Failed to parse clustering from LLM response")
        return None
    
    print(f"\n✅ LLM created {len(super_node_clusters)} clusters of super-nodes")
    

    original_clusters = expand_to_original_nodes(super_node_clusters, hier)
    
    
    metrics = compute_basic_metrics(original_clusters, graph)
    
   
    result = {
        "clusters": original_clusters,
        "super_node_clusters": super_node_clusters,
        "metrics": metrics,
        "config": {
            "model": model,
            "backend": backend,
            "louvain_levels": louvain_levels,
            "louvain_resolution": louvain_resolution,
            "num_super_nodes": len(hier.super_node_to_original)
        }
    }
    
   
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Original nodes: {graph.number_of_nodes()}")
    print(f"Super-nodes: {len(hier.super_node_to_original)}")
    print(f"Final clusters: {len(original_clusters)}")
    print(f"Cohesion: {metrics['cohesion']*100:.1f}%")
    print(f"Coupling: {metrics['coupling']*100:.1f}%")
    print("\nCluster sizes:")
    for name, size in sorted(metrics['cluster_sizes'].items(), key=lambda x: -x[1]):
        print(f"  {name}: {size} nodes")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Simple Clustering Pipeline with Louvain + LLM"
    )
    parser.add_argument("graph_file", help="Path to pickled graph file")
    parser.add_argument("--model", default="llama3:8b", 
                        help="LLM model name (default: llama3:8b)")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                        help="LLM backend to use (default: ollama)")
    parser.add_argument("--host", default="http://localhost:11434", 
                        help="Ollama host URL")
    parser.add_argument("--clusters", type=int, default=None,
                        help="Target number of clusters (default: LLM decides)")
    parser.add_argument("--louvain-levels", type=int, default=2,
                        help="Louvain coarsening levels (default: 2)")
    parser.add_argument("--louvain-resolution", type=float, default=1.0,
                        help="Louvain resolution parameter (default: 1.0)")
    parser.add_argument("--output", default="simple_clusters.json",
                        help="Output file path (default: simple_clusters.json)")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Check graph file exists
    if not os.path.exists(args.graph_file):
        print(f"❌ Graph file not found: {args.graph_file}")
        sys.exit(1)
    
    result = run_pipeline(
        graph_path=args.graph_file,
        model=args.model,
        backend=args.backend,
        host=args.host,
        num_clusters=args.clusters,
        louvain_levels=args.louvain_levels,
        louvain_resolution=args.louvain_resolution,
        output_file=args.output,
        api_key=args.api_key
    )
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
