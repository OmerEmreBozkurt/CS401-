#!/usr/bin/env python3
"""
Compact Graph Builder - Fixed Version
Creates graph representation using the compact character codes directly
Handles multiple input formats including 'depends' lines
"""

import sys
import os
import networkx as nx
import json
import pickle
from collections import Counter


class CompactGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.line_formats_detected = Counter()

    def detect_line_format(self, line):
        """Detect the format of a line"""
        parts = line.strip().split()
        if not parts:
            return None

        # Check if first token looks like a relation word
        first_token = parts[0].lower()
        relation_words = {'depends', 'contain', 'include', 'import', 'use', 'call'}

        if first_token in relation_words:
            return 'rsf_with_relation'
        elif len(parts) >= 2 and all(len(p) <= 3 and p.isalnum() for p in parts[:2]):
            # Likely compact format (short alphanumeric codes)
            return 'compact_dependencies'
        else:
            return 'unknown'

    def build_graph_from_compact(self, compact_file):
        """Build graph from compact file, auto-detecting format"""
        edge_count = 0
        line_count = 0
        skipped_lines = 0

        print(f"Reading {compact_file}...")

        with open(compact_file, 'r') as f:
            lines = f.readlines()

        # Detect format from first few valid lines
        format_detected = None
        for line in lines[:20]:
            fmt = self.detect_line_format(line)
            if fmt and fmt != 'unknown':
                format_detected = fmt
                break

        if format_detected:
            print(f"Detected format: {format_detected}")
        else:
            print("Warning: Could not detect format, assuming compact dependencies")
            format_detected = 'compact_dependencies'

        # Process all lines
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            line_count += 1

            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                skipped_lines += 1
                continue

            if format_detected == 'rsf_with_relation':
                # Skip the relation word, use rest as source -> targets
                if len(parts) >= 2:
                    source = parts[1] if len(parts) > 1 else None
                    targets = parts[2:] if len(parts) > 2 else []
                    if source:
                        for target in targets:
                            self.graph.add_edge(source, target)
                            edge_count += 1
                        if not targets:
                            self.graph.add_node(source)
            else:
                # Compact format: first token is source, rest are targets
                source = parts[0]
                for target in parts[1:]:
                    self.graph.add_edge(source, target)
                    edge_count += 1

        print(f"Processed {line_count} lines ({skipped_lines} skipped)")
        print(f"Built graph with {self.graph.number_of_nodes()} nodes and {edge_count} edges")

        if self.graph.number_of_nodes() == 0:
            print("Warning: No nodes found in graph! Check input format.")

        return self.graph

    def get_stats(self):
        """Get basic graph statistics"""
        if self.graph.number_of_nodes() == 0:
            return {'error': 'Graph is empty'}

        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
            'avg_degree': sum(dict(
                self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }

        # Check if it's a DAG
        stats['is_dag'] = nx.is_directed_acyclic_graph(self.graph)

        # Component analysis
        stats['weakly_connected_components'] = nx.number_weakly_connected_components(self.graph)
        stats['strongly_connected_components'] = nx.number_strongly_connected_components(self.graph)

        # Find largest component size
        if stats['weakly_connected_components'] > 0:
            largest_wcc = max(nx.weakly_connected_components(self.graph), key=len)
            stats['largest_component_size'] = len(largest_wcc)
            stats['largest_component_percentage'] = (len(largest_wcc) / self.graph.number_of_nodes()) * 100

        return stats

    def get_node_info(self):
        """Get degree information for all nodes"""
        node_info = {}

        for node in self.graph.nodes():
            node_info[node] = {
                'in': self.graph.in_degree(node),
                'out': self.graph.out_degree(node),
                'total': self.graph.degree(node)
            }

        return node_info

    def find_important_nodes(self, top_n=10):
        """Find most connected nodes"""
        if self.graph.number_of_nodes() == 0:
            return {'error': 'Graph is empty'}

        node_info = self.get_node_info()

        # Sort by different metrics
        by_in = sorted(node_info.items(), key=lambda x: x[1]['in'], reverse=True)[:top_n]
        by_out = sorted(node_info.items(), key=lambda x: x[1]['out'], reverse=True)[:top_n]
        by_total = sorted(node_info.items(), key=lambda x: x[1]['total'], reverse=True)[:top_n]

        return {
            'highest_in_degree': [(n, info['in']) for n, info in by_in],
            'highest_out_degree': [(n, info['out']) for n, info in by_out],
            'highest_total_degree': [(n, info['total']) for n, info in by_total]
        }

    def find_isolated_nodes(self):
        """Find nodes with no connections"""
        isolated = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        return isolated

    def find_source_nodes(self):
        """Find nodes with no incoming edges (roots)"""
        sources = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        return sources

    def find_sink_nodes(self):
        """Find nodes with no outgoing edges (leaves)"""
        sinks = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        return sinks

    def save_graph(self, output_file):
        """Save the graph object"""
        with open(output_file, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved graph to {output_file}")

    def export_edge_list(self, output_file):
        """Export as simple edge list"""
        with open(output_file, 'w') as f:
            for source, target in self.graph.edges():
                f.write(f"{source} {target}\n")
        print(f"Exported edge list to {output_file} ({self.graph.number_of_edges()} edges)")

    def export_adjacency_json(self, output_file):
        """Export as adjacency list in JSON"""
        adj_dict = {}
        for node in self.graph.nodes():
            adj_dict[node] = list(self.graph.successors(node))

        with open(output_file, 'w') as f:
            json.dump(adj_dict, f, indent=2)
        print(f"Exported adjacency list to {output_file}")

    def export_graphml(self, output_file):
        """Export in GraphML format for visualization tools"""
        nx.write_graphml(self.graph, output_file)
        print(f"Exported GraphML to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python graph_builder.py <compact_file> [output_prefix]")
        print("\nExample:")
        print("  python graph_builder.py dependencies_compact.rsf my_graph")
        print("\nThe script auto-detects the input format:")
        print("  - Compact format: 'a b c d' means 'a' depends on b, c, d")
        print("  - RSF format: 'depends a b c' means 'a' depends on b, c")
        sys.exit(1)

    compact_file = sys.argv[1]

    if not os.path.exists(compact_file):
        print(f"Error: Input file '{compact_file}' not found.")
        sys.exit(1)

    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "compact_graph"

    # Build graph
    graph_builder = CompactGraph()
    graph = graph_builder.build_graph_from_compact(compact_file)

    if graph.number_of_nodes() == 0:
        print("Error: No nodes found in graph. Please check your input file format.")
        sys.exit(1)

    # Show statistics
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)
    stats = graph_builder.get_stats()
    for key, value in stats.items():
        if key == 'density':
            print(f"{key}: {value:.6f}")
        elif key == 'avg_degree':
            print(f"{key}: {value:.2f}")
        elif key == 'largest_component_percentage':
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value}")

    # Show important nodes
    print("\n" + "=" * 60)
    print("MOST CONNECTED NODES")
    print("=" * 60)
    important = graph_builder.find_important_nodes()

    if 'error' not in important:
        print("\nHighest in-degree (most depended on):")
        for char, degree in important['highest_in_degree'][:5]:
            print(f"  '{char}': {degree} incoming edges")

        print("\nHighest out-degree (most dependencies):")
        for char, degree in important['highest_out_degree'][:5]:
            print(f"  '{char}': {degree} outgoing edges")

    # Show special nodes
    sources = graph_builder.find_source_nodes()
    sinks = graph_builder.find_sink_nodes()
    isolated = graph_builder.find_isolated_nodes()

    print(f"\nRoot nodes (no incoming): {len(sources)}")
    if sources and len(sources) <= 10:
        print(f"  {', '.join(sources[:10])}")

    print(f"Leaf nodes (no outgoing): {len(sinks)}")
    if sinks and len(sinks) <= 10:
        print(f"  {', '.join(sinks[:10])}")

    print(f"Isolated nodes (no connections): {len(isolated)}")
    if isolated and len(isolated) <= 10:
        print(f"  {', '.join(isolated[:10])}")

    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # Save graph object
    graph_builder.save_graph(f"{output_prefix}.pkl")

    # Export edge list
    graph_builder.export_edge_list(f"{output_prefix}_edges.txt")

    # Export adjacency list
    graph_builder.export_adjacency_json(f"{output_prefix}_adj.json")

    # Export GraphML for visualization
    graph_builder.export_graphml(f"{output_prefix}.graphml")


if __name__ == "__main__":
    main()