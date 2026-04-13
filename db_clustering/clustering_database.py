#!/usr/bin/env python3
"""
Database Layer for Graph Clustering System

This module provides:
1. SQLite database for storing graph and clustering state
2. Efficient queries that return minimal data (token-efficient)
3. State management for iterative clustering
4. History tracking for reproducibility
"""

import sqlite3
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pickle

import networkx as nx


@dataclass
class ClusterStats:
    """Statistics for a single cluster"""
    cluster_id: str
    node_count: int
    internal_edges: int
    external_edges: int
    cohesion: float  # internal / (internal + external)
    sample_nodes: List[str]


@dataclass
class GraphSummary:
    """High-level graph summary"""
    total_nodes: int
    total_edges: int
    num_clusters: int
    avg_cluster_size: float
    total_internal_edges: int
    total_external_edges: int
    overall_cohesion: float


class ClusteringDatabase:
    """
    SQLite-backed database for graph clustering.
    
    Key features:
    - Stores graph structure (nodes, edges)
    - Maintains current clustering state
    - Tracks all changes (history)
    - Provides token-efficient query methods
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize database.
        
        Args:
            db_path: Path to SQLite file, or ":memory:" for in-memory DB
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._iteration = 0
    
    def _create_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                cluster_id TEXT,
                in_degree INTEGER DEFAULT 0,
                out_degree INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL DEFAULT 1.0
            )
        """)
        
        # Clusters summary table (cached stats)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id TEXT PRIMARY KEY,
                node_count INTEGER DEFAULT 0,
                internal_edges INTEGER DEFAULT 0,
                external_edges INTEGER DEFAULT 0,
                cohesion REAL DEFAULT 0.0
            )
        """)
        
        # History table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                action TEXT,
                details TEXT,
                timestamp TEXT,
                metrics_before TEXT,
                metrics_after TEXT
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_cluster ON nodes(cluster_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
        
        self.conn.commit()
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_from_networkx(self, graph: nx.DiGraph, initial_clustering: Optional[Dict[str, List[str]]] = None):
        """
        Load graph from NetworkX into database.
        
        Args:
            graph: NetworkX directed graph
            initial_clustering: Optional initial clustering {cluster_id: [node_ids]}
        """
        cursor = self.conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM nodes")
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM clusters")
        
        # Insert nodes
        for node in graph.nodes():
            in_deg = graph.in_degree(node)
            out_deg = graph.out_degree(node)
            cursor.execute(
                "INSERT INTO nodes (id, cluster_id, in_degree, out_degree) VALUES (?, ?, ?, ?)",
                (str(node), None, in_deg, out_deg)
            )
        
        # Insert edges
        for source, target in graph.edges():
            weight = graph[source][target].get('weight', 1.0)
            cursor.execute(
                "INSERT INTO edges (source, target, weight) VALUES (?, ?, ?)",
                (str(source), str(target), weight)
            )
        
        self.conn.commit()
        
        # Apply initial clustering if provided
        if initial_clustering:
            self.apply_clustering(initial_clustering)
        
        print(f"✓ Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges into database")
    
    def load_from_pickle(self, pickle_path: str, initial_clustering: Optional[Dict[str, List[str]]] = None):
        """Load graph from pickle file"""
        with open(pickle_path, 'rb') as f:
            graph = pickle.load(f)
        self.load_from_networkx(graph, initial_clustering)
        return graph
    
    # =========================================================================
    # Clustering State Management
    # =========================================================================
    
    def apply_clustering(self, clustering: Dict[str, List[str]]):
        """
        Apply a clustering assignment to all nodes.
        
        Args:
            clustering: {cluster_id: [node_ids]}
        """
        cursor = self.conn.cursor()
        
        # Reset all cluster assignments
        cursor.execute("UPDATE nodes SET cluster_id = NULL")
        
        # Apply new assignments
        for cluster_id, nodes in clustering.items():
            for node_id in nodes:
                cursor.execute(
                    "UPDATE nodes SET cluster_id = ? WHERE id = ?",
                    (cluster_id, str(node_id))
                )
        
        self.conn.commit()
        
        # Update cluster statistics
        self._update_cluster_stats()
    
    def get_current_clustering(self) -> Dict[str, List[str]]:
        """Get current clustering state from database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT cluster_id, id FROM nodes 
            WHERE cluster_id IS NOT NULL 
            ORDER BY cluster_id
        """)
        
        clustering = {}
        for row in cursor.fetchall():
            cluster_id = row['cluster_id']
            node_id = row['id']
            if cluster_id not in clustering:
                clustering[cluster_id] = []
            clustering[cluster_id].append(node_id)
        
        return clustering
    
    def move_node(self, node_id: str, to_cluster: str) -> bool:
        """
        Move a node to a different cluster.
        
        Args:
            node_id: Node to move
            to_cluster: Target cluster ID
            
        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        
        # Get current cluster
        cursor.execute("SELECT cluster_id FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            return False
        
        from_cluster = row['cluster_id']
        
        # Update node
        cursor.execute(
            "UPDATE nodes SET cluster_id = ? WHERE id = ?",
            (to_cluster, node_id)
        )
        
        self.conn.commit()
        
        # Log action
        self._log_action('move', {
            'node': node_id,
            'from': from_cluster,
            'to': to_cluster
        })
        
        # Update stats for affected clusters
        self._update_cluster_stats([from_cluster, to_cluster])
        
        return True
    
    def merge_clusters(self, cluster_a: str, cluster_b: str, new_name: Optional[str] = None) -> str:
        """
        Merge two clusters into one.
        
        Args:
            cluster_a: First cluster
            cluster_b: Second cluster (will be merged into cluster_a)
            new_name: Optional new name for merged cluster
            
        Returns:
            Name of merged cluster
        """
        cursor = self.conn.cursor()
        
        target_cluster = new_name or cluster_a
        
        # Move all nodes from cluster_b to target
        cursor.execute(
            "UPDATE nodes SET cluster_id = ? WHERE cluster_id = ?",
            (target_cluster, cluster_b)
        )
        
        # If renaming cluster_a
        if new_name and new_name != cluster_a:
            cursor.execute(
                "UPDATE nodes SET cluster_id = ? WHERE cluster_id = ?",
                (new_name, cluster_a)
            )
        
        self.conn.commit()
        
        # Log action
        self._log_action('merge', {
            'clusters': [cluster_a, cluster_b],
            'result': target_cluster
        })
        
        # Update stats
        self._update_cluster_stats()
        
        return target_cluster
    
    def split_cluster(self, cluster_id: str, node_groups: List[List[str]]) -> List[str]:
        """
        Split a cluster into multiple new clusters.
        
        Args:
            cluster_id: Cluster to split
            node_groups: List of node groups for new clusters
            
        Returns:
            List of new cluster IDs
        """
        cursor = self.conn.cursor()
        
        new_cluster_ids = []
        for i, nodes in enumerate(node_groups):
            new_id = f"{cluster_id}_split_{i}"
            new_cluster_ids.append(new_id)
            
            for node_id in nodes:
                cursor.execute(
                    "UPDATE nodes SET cluster_id = ? WHERE id = ?",
                    (new_id, node_id)
                )
        
        self.conn.commit()
        
        # Log action
        self._log_action('split', {
            'original': cluster_id,
            'new_clusters': new_cluster_ids
        })
        
        # Update stats
        self._update_cluster_stats()
        
        return new_cluster_ids
    
    def create_cluster(self, cluster_id: str, node_ids: List[str]) -> bool:
        """Create a new cluster with specified nodes"""
        cursor = self.conn.cursor()
        
        for node_id in node_ids:
            cursor.execute(
                "UPDATE nodes SET cluster_id = ? WHERE id = ?",
                (cluster_id, node_id)
            )
        
        self.conn.commit()
        self._update_cluster_stats([cluster_id])
        
        return True
    
    # =========================================================================
    # Query Methods (Token-Efficient!)
    # =========================================================================
    
    def get_graph_summary(self) -> str:
        """
        Get high-level graph summary.
        Returns minimal JSON - very token efficient!
        """
        cursor = self.conn.cursor()
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) as cnt FROM nodes")
        total_nodes = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(*) as cnt FROM edges")
        total_edges = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(DISTINCT cluster_id) as cnt FROM nodes WHERE cluster_id IS NOT NULL")
        num_clusters = cursor.fetchone()['cnt']
        
        # Edge statistics
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN n1.cluster_id = n2.cluster_id THEN 1 ELSE 0 END) as internal,
                SUM(CASE WHEN n1.cluster_id != n2.cluster_id THEN 1 ELSE 0 END) as external
            FROM edges e
            JOIN nodes n1 ON e.source = n1.id
            JOIN nodes n2 ON e.target = n2.id
            WHERE n1.cluster_id IS NOT NULL AND n2.cluster_id IS NOT NULL
        """)
        row = cursor.fetchone()
        internal = row['internal'] or 0
        external = row['external'] or 0
        
        total = internal + external
        cohesion = internal / total if total > 0 else 0
        
        summary = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "num_clusters": num_clusters,
            "avg_cluster_size": round(total_nodes / num_clusters, 1) if num_clusters > 0 else 0,
            "internal_edges": internal,
            "external_edges": external,
            "cohesion": round(cohesion, 4)
        }
        
        return json.dumps(summary, indent=2)
    
    def get_cluster_list(self) -> str:
        """
        Get list of all clusters with basic stats.
        Token efficient - no node lists!
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                cluster_id,
                COUNT(*) as node_count
            FROM nodes 
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id
            ORDER BY node_count DESC
        """)
        
        clusters = []
        for row in cursor.fetchall():
            clusters.append({
                "id": row['cluster_id'],
                "size": row['node_count']
            })
        
        return json.dumps(clusters, indent=2)
    
    def get_cluster_info(self, cluster_id: str) -> str:
        """
        Get detailed info about a specific cluster.
        Includes sample nodes but not full list.
        """
        cursor = self.conn.cursor()
        
        # Get nodes in cluster
        cursor.execute(
            "SELECT id FROM nodes WHERE cluster_id = ? LIMIT 10",
            (cluster_id,)
        )
        sample_nodes = [row['id'] for row in cursor.fetchall()]
        
        # Get total count
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM nodes WHERE cluster_id = ?",
            (cluster_id,)
        )
        node_count = cursor.fetchone()['cnt']
        
        # Get internal edges
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM edges e
            JOIN nodes n1 ON e.source = n1.id
            JOIN nodes n2 ON e.target = n2.id
            WHERE n1.cluster_id = ? AND n2.cluster_id = ?
        """, (cluster_id, cluster_id))
        internal = cursor.fetchone()['cnt']
        
        # Get external edges (outgoing)
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM edges e
            JOIN nodes n1 ON e.source = n1.id
            JOIN nodes n2 ON e.target = n2.id
            WHERE n1.cluster_id = ? AND n2.cluster_id != ?
        """, (cluster_id, cluster_id))
        external = cursor.fetchone()['cnt']
        
        total = internal + external
        
        info = {
            "cluster_id": cluster_id,
            "node_count": node_count,
            "sample_nodes": sample_nodes,
            "internal_edges": internal,
            "external_edges": external,
            "cohesion": round(internal / total, 4) if total > 0 else 0
        }
        
        return json.dumps(info, indent=2)
    
    def get_node_info(self, node_id: str) -> str:
        """Get info about a specific node"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        
        if not row:
            return json.dumps({"error": f"Node '{node_id}' not found"})
        
        # Get connections
        cursor.execute("""
            SELECT target, 
                   (SELECT cluster_id FROM nodes WHERE id = target) as target_cluster
            FROM edges WHERE source = ?
        """, (node_id,))
        outgoing = [{"node": r['target'], "cluster": r['target_cluster']} for r in cursor.fetchall()]
        
        cursor.execute("""
            SELECT source,
                   (SELECT cluster_id FROM nodes WHERE id = source) as source_cluster
            FROM edges WHERE target = ?
        """, (node_id,))
        incoming = [{"node": r['source'], "cluster": r['source_cluster']} for r in cursor.fetchall()]
        
        info = {
            "node_id": node_id,
            "cluster": row['cluster_id'],
            "in_degree": row['in_degree'],
            "out_degree": row['out_degree'],
            "outgoing_connections": outgoing[:10],  # Limit for token efficiency
            "incoming_connections": incoming[:10]
        }
        
        return json.dumps(info, indent=2)
    
    def get_cross_cluster_edges(self, cluster_a: str, cluster_b: str) -> str:
        """Get edge count between two clusters"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM edges e
            JOIN nodes n1 ON e.source = n1.id
            JOIN nodes n2 ON e.target = n2.id
            WHERE (n1.cluster_id = ? AND n2.cluster_id = ?)
               OR (n1.cluster_id = ? AND n2.cluster_id = ?)
        """, (cluster_a, cluster_b, cluster_b, cluster_a))
        
        count = cursor.fetchone()['cnt']
        
        return json.dumps({
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "edge_count": count
        })
    
    def get_problematic_nodes(self, limit: int = 10) -> str:
        """
        Find nodes that might be misplaced.
        A node is "problematic" if it has more external edges than internal.
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            WITH node_edges AS (
                SELECT 
                    n.id,
                    n.cluster_id,
                    SUM(CASE WHEN n2.cluster_id = n.cluster_id THEN 1 ELSE 0 END) as internal,
                    SUM(CASE WHEN n2.cluster_id != n.cluster_id THEN 1 ELSE 0 END) as external
                FROM nodes n
                LEFT JOIN edges e ON n.id = e.source
                LEFT JOIN nodes n2 ON e.target = n2.id
                WHERE n.cluster_id IS NOT NULL
                GROUP BY n.id, n.cluster_id
            )
            SELECT 
                id,
                cluster_id,
                internal,
                external,
                CASE WHEN (internal + external) > 0 
                     THEN CAST(external AS REAL) / (internal + external) 
                     ELSE 0 END as external_ratio
            FROM node_edges
            WHERE external > internal
            ORDER BY external_ratio DESC
            LIMIT ?
        """, (limit,))
        
        problematic = []
        for row in cursor.fetchall():
            problematic.append({
                "node": row['id'],
                "cluster": row['cluster_id'],
                "internal_edges": row['internal'],
                "external_edges": row['external'],
                "external_ratio": round(row['external_ratio'], 2)
            })
        
        return json.dumps(problematic, indent=2)
    
    def get_cluster_connections(self) -> str:
        """
        Get edge counts between all pairs of clusters.
        Token efficient - just counts, no node lists!
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                n1.cluster_id as source_cluster,
                n2.cluster_id as target_cluster,
                COUNT(*) as edge_count
            FROM edges e
            JOIN nodes n1 ON e.source = n1.id
            JOIN nodes n2 ON e.target = n2.id
            WHERE n1.cluster_id IS NOT NULL 
              AND n2.cluster_id IS NOT NULL
              AND n1.cluster_id != n2.cluster_id
            GROUP BY n1.cluster_id, n2.cluster_id
            ORDER BY edge_count DESC
            LIMIT 20
        """)
        
        connections = []
        for row in cursor.fetchall():
            connections.append({
                "from": row['source_cluster'],
                "to": row['target_cluster'],
                "edges": row['edge_count']
            })
        
        return json.dumps(connections, indent=2)
    
    def get_suggested_moves(self, limit: int = 5) -> str:
        """
        Suggest node moves that would improve clustering.
        Finds nodes with strong connections to other clusters.
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            WITH node_cluster_edges AS (
                SELECT 
                    n1.id as node_id,
                    n1.cluster_id as current_cluster,
                    n2.cluster_id as connected_cluster,
                    COUNT(*) as edge_count
                FROM nodes n1
                JOIN edges e ON n1.id = e.source
                JOIN nodes n2 ON e.target = n2.id
                WHERE n1.cluster_id IS NOT NULL 
                  AND n2.cluster_id IS NOT NULL
                  AND n1.cluster_id != n2.cluster_id
                GROUP BY n1.id, n1.cluster_id, n2.cluster_id
            ),
            best_alternative AS (
                SELECT 
                    node_id,
                    current_cluster,
                    connected_cluster as suggested_cluster,
                    edge_count as edges_to_suggested,
                    ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY edge_count DESC) as rn
                FROM node_cluster_edges
            )
            SELECT 
                ba.node_id,
                ba.current_cluster,
                ba.suggested_cluster,
                ba.edges_to_suggested,
                COALESCE(internal.cnt, 0) as edges_to_current
            FROM best_alternative ba
            LEFT JOIN (
                SELECT n1.id, COUNT(*) as cnt
                FROM nodes n1
                JOIN edges e ON n1.id = e.source
                JOIN nodes n2 ON e.target = n2.id
                WHERE n1.cluster_id = n2.cluster_id
                GROUP BY n1.id
            ) internal ON ba.node_id = internal.id
            WHERE ba.rn = 1
              AND ba.edges_to_suggested > COALESCE(internal.cnt, 0)
            ORDER BY (ba.edges_to_suggested - COALESCE(internal.cnt, 0)) DESC
            LIMIT ?
        """, (limit,))
        
        suggestions = []
        for row in cursor.fetchall():
            suggestions.append({
                "node": row['node_id'],
                "from_cluster": row['current_cluster'],
                "to_cluster": row['suggested_cluster'],
                "edges_to_suggested": row['edges_to_suggested'],
                "edges_to_current": row['edges_to_current'],
                "improvement": row['edges_to_suggested'] - row['edges_to_current']
            })
        
        return json.dumps(suggestions, indent=2)
    
    # =========================================================================
    # Statistics and Metrics
    # =========================================================================
    
    def _update_cluster_stats(self, cluster_ids: Optional[List[str]] = None):
        """Update cached cluster statistics"""
        cursor = self.conn.cursor()
        
        if cluster_ids is None:
            # Update all clusters
            cursor.execute("DELETE FROM clusters")
            cursor.execute("""
                INSERT INTO clusters (id, node_count, internal_edges, external_edges, cohesion)
                SELECT 
                    n.cluster_id,
                    COUNT(DISTINCT n.id),
                    0, 0, 0
                FROM nodes n
                WHERE n.cluster_id IS NOT NULL
                GROUP BY n.cluster_id
            """)
        
        self.conn.commit()
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate clustering metrics (cohesion, coupling)"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN n1.cluster_id = n2.cluster_id THEN 1 ELSE 0 END) as internal,
                SUM(CASE WHEN n1.cluster_id != n2.cluster_id THEN 1 ELSE 0 END) as external
            FROM edges e
            JOIN nodes n1 ON e.source = n1.id
            JOIN nodes n2 ON e.target = n2.id
            WHERE n1.cluster_id IS NOT NULL AND n2.cluster_id IS NOT NULL
        """)
        
        row = cursor.fetchone()
        internal = row['internal'] or 0
        external = row['external'] or 0
        total = internal + external
        
        return {
            "internal_edges": internal,
            "external_edges": external,
            "cohesion": internal / total if total > 0 else 0,
            "coupling": external / total if total > 0 else 0
        }
    
    # =========================================================================
    # History and Logging
    # =========================================================================
    
    def _log_action(self, action: str, details: Dict):
        """Log an action to history"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO history (iteration, action, details, timestamp)
            VALUES (?, ?, ?, ?)
        """, (self._iteration, action, json.dumps(details), datetime.now().isoformat()))
        self.conn.commit()
    
    def start_iteration(self):
        """Mark start of new iteration"""
        self._iteration += 1
    
    def get_history(self) -> List[Dict]:
        """Get action history"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM history ORDER BY id")
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "id": row['id'],
                "iteration": row['iteration'],
                "action": row['action'],
                "details": json.loads(row['details']) if row['details'] else None,
                "timestamp": row['timestamp']
            })
        
        return history
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_clustering_to_rsf(self, output_path: str):
        """Export current clustering to RSF format"""
        clustering = self.get_current_clustering()
        
        with open(output_path, 'w') as f:
            for cluster_id, nodes in clustering.items():
                for node in nodes:
                    f.write(f"contain {cluster_id} {node}\n")
        
        print(f"✓ Exported clustering to {output_path}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """Demo the database functionality"""
    import tempfile
    
    # Create test graph
    G = nx.DiGraph()
    
    # Cluster 1: nodes 0-4
    for i in range(5):
        for j in range(5):
            if i != j:
                G.add_edge(f"node_{i}", f"node_{j}")
    
    # Cluster 2: nodes 5-9
    for i in range(5, 10):
        for j in range(5, 10):
            if i != j:
                G.add_edge(f"node_{i}", f"node_{j}")
    
    # Cross-cluster edges
    G.add_edge("node_2", "node_7")
    G.add_edge("node_4", "node_5")
    
    print("=" * 60)
    print("Database Demo")
    print("=" * 60)
    
    # Create database
    db = ClusteringDatabase(":memory:")
    
    # Load graph
    initial_clustering = {
        "cluster_1": [f"node_{i}" for i in range(5)],
        "cluster_2": [f"node_{i}" for i in range(5, 10)]
    }
    db.load_from_networkx(G, initial_clustering)
    
    # Test queries
    print("\n📊 Graph Summary:")
    print(db.get_graph_summary())
    
    print("\n📋 Cluster List:")
    print(db.get_cluster_list())
    
    print("\n🔍 Cluster Info (cluster_1):")
    print(db.get_cluster_info("cluster_1"))
    
    print("\n🔗 Cross-Cluster Edges:")
    print(db.get_cross_cluster_edges("cluster_1", "cluster_2"))
    
    print("\n⚠️ Problematic Nodes:")
    print(db.get_problematic_nodes())
    
    print("\n💡 Suggested Moves:")
    print(db.get_suggested_moves())
    
    # Test move
    print("\n🔄 Moving node_2 to cluster_2...")
    db.move_node("node_2", "cluster_2")
    
    print("\n📊 Updated Graph Summary:")
    print(db.get_graph_summary())
    
    print("\n📜 History:")
    print(json.dumps(db.get_history(), indent=2))
    
    db.close()


if __name__ == "__main__":
    demo()
