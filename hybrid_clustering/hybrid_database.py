#!/usr/bin/env python3
"""
Enhanced Clustering Database with Baseline Support

This module provides:
1. SQLite database for storing graph and clustering state
2. Baseline algorithm results storage (ACDC, MGMC, Louvain, etc.)
3. Consensus analysis - find where algorithms agree/disagree
4. Token-efficient queries for LLM agents
"""

import sqlite3
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import pickle

import networkx as nx


@dataclass
class BaselineResult:
    """Result from a baseline algorithm"""
    algorithm: str
    clustering: Dict[str, List[str]]
    turbomq: Optional[float]
    mojo_fm: Optional[float]
    cohesion: float
    num_clusters: int


class HybridClusteringDatabase:
    """
    SQLite-backed database for hybrid baseline-guided clustering.
    
    Key features:
    - Store multiple baseline results (ACDC, MGMC, Louvain)
    - Track node consensus across algorithms
    - Identify controversial nodes
    - Provide token-efficient queries
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
        
        # Baselines table - stores results from different algorithms
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                id TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                clustering TEXT NOT NULL,
                turbomq REAL,
                mojo_fm REAL,
                cohesion REAL,
                coupling REAL,
                num_clusters INTEGER,
                created_at TEXT
            )
        """)
        
        # Node placements by each baseline
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS baseline_placements (
                node_id TEXT,
                baseline_id TEXT,
                cluster_id TEXT,
                PRIMARY KEY (node_id, baseline_id)
            )
        """)
        
        # Node consensus - where algorithms agree/disagree
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS node_consensus (
                node_id TEXT PRIMARY KEY,
                agreement_score REAL,
                is_controversial BOOLEAN,
                placement_summary TEXT
            )
        """)
        
        # Current working clustering
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_clustering (
                node_id TEXT PRIMARY KEY,
                cluster_id TEXT,
                source TEXT  -- 'baseline:acdc', 'llm', 'manual'
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_consensus_controversial ON node_consensus(is_controversial)")
        
        self.conn.commit()
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_graph_from_pickle(self, pickle_path: str) -> nx.DiGraph:
        """Load graph from pickle file"""
        with open(pickle_path, 'rb') as f:
            graph = pickle.load(f)
        
        self.load_graph(graph)
        return graph
    
    def load_graph(self, graph: nx.DiGraph):
        """Load graph into database"""
        cursor = self.conn.cursor()
        
        # Clear existing graph data
        cursor.execute("DELETE FROM nodes")
        cursor.execute("DELETE FROM edges")
        
        # Insert nodes
        for node in graph.nodes():
            in_deg = graph.in_degree(node)
            out_deg = graph.out_degree(node)
            cursor.execute(
                "INSERT INTO nodes (id, in_degree, out_degree) VALUES (?, ?, ?)",
                (str(node), in_deg, out_deg)
            )
        
        # Insert edges
        for source, target in graph.edges():
            weight = graph[source][target].get('weight', 1.0)
            cursor.execute(
                "INSERT INTO edges (source, target, weight) VALUES (?, ?, ?)",
                (str(source), str(target), weight)
            )
        
        self.conn.commit()
        print(f"✓ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # =========================================================================
    # Baseline Management
    # =========================================================================
    
    def add_baseline(self, baseline_id: str, algorithm: str, 
                     clustering: Dict[str, List[str]],
                     turbomq: Optional[float] = None,
                     mojo_fm: Optional[float] = None):
        """
        Add a baseline algorithm result.
        
        Args:
            baseline_id: Unique ID (e.g., 'acdc', 'mgmc', 'louvain')
            algorithm: Algorithm name
            clustering: {cluster_id: [node_ids]}
            turbomq: TurboMQ score (optional)
            mojo_fm: MoJo-FM score (optional)
        """
        cursor = self.conn.cursor()
        
        # Calculate metrics
        cohesion, coupling = self._calculate_clustering_metrics(clustering)
        
        # Store baseline
        cursor.execute("""
            INSERT OR REPLACE INTO baselines 
            (id, algorithm, clustering, turbomq, mojo_fm, cohesion, coupling, num_clusters, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            baseline_id,
            algorithm,
            json.dumps(clustering),
            turbomq,
            mojo_fm,
            cohesion,
            coupling,
            len(clustering),
            datetime.now().isoformat()
        ))
        
        # Store individual node placements
        cursor.execute("DELETE FROM baseline_placements WHERE baseline_id = ?", (baseline_id,))
        for cluster_id, nodes in clustering.items():
            for node_id in nodes:
                cursor.execute("""
                    INSERT INTO baseline_placements (node_id, baseline_id, cluster_id)
                    VALUES (?, ?, ?)
                """, (str(node_id), baseline_id, cluster_id))
        
        self.conn.commit()
        
        # Update consensus after adding baseline
        self._update_consensus()
        
        print(f"✓ Added baseline '{baseline_id}': {len(clustering)} clusters, "
              f"cohesion={cohesion:.4f}, TurboMQ={turbomq or 'N/A'}")
    
    def load_baseline_from_rsf(self, rsf_path: str, baseline_id: str, algorithm: str,
                                turbomq: Optional[float] = None,
                                mojo_fm: Optional[float] = None):
        """Load baseline clustering from RSF file"""
        clustering = {}
        
        with open(rsf_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] == 'contain':
                    cluster_id = parts[1]
                    node_id = parts[2]
                    
                    if cluster_id not in clustering:
                        clustering[cluster_id] = []
                    clustering[cluster_id].append(node_id)
        
        self.add_baseline(baseline_id, algorithm, clustering, turbomq, mojo_fm)
    
    def _calculate_clustering_metrics(self, clustering: Dict[str, List[str]]) -> Tuple[float, float]:
        """Calculate cohesion and coupling for a clustering"""
        cursor = self.conn.cursor()
        
        # Build node to cluster mapping
        node_to_cluster = {}
        for cluster_id, nodes in clustering.items():
            for node in nodes:
                node_to_cluster[str(node)] = cluster_id
        
        # Count internal vs external edges
        internal = 0
        external = 0
        
        cursor.execute("SELECT source, target FROM edges")
        for row in cursor.fetchall():
            src_cluster = node_to_cluster.get(row['source'])
            tgt_cluster = node_to_cluster.get(row['target'])
            
            if src_cluster and tgt_cluster:
                if src_cluster == tgt_cluster:
                    internal += 1
                else:
                    external += 1
        
        total = internal + external
        cohesion = internal / total if total > 0 else 0
        coupling = external / total if total > 0 else 0
        
        return cohesion, coupling
    
    def _update_consensus(self):
        """Update node consensus based on all baselines"""
        cursor = self.conn.cursor()
        
        # Get all baselines
        cursor.execute("SELECT id FROM baselines")
        baseline_ids = [row['id'] for row in cursor.fetchall()]
        
        if len(baseline_ids) < 2:
            return  # Need at least 2 baselines for consensus
        
        # Get all nodes
        cursor.execute("SELECT id FROM nodes")
        node_ids = [row['id'] for row in cursor.fetchall()]
        
        # Clear existing consensus
        cursor.execute("DELETE FROM node_consensus")
        
        # Calculate consensus for each node
        for node_id in node_ids:
            cursor.execute("""
                SELECT baseline_id, cluster_id 
                FROM baseline_placements 
                WHERE node_id = ?
            """, (node_id,))
            
            placements = {row['baseline_id']: row['cluster_id'] for row in cursor.fetchall()}
            
            if not placements:
                continue
            
            # Count how many baselines agree
            cluster_counts = defaultdict(list)
            for baseline_id, cluster_id in placements.items():
                cluster_counts[cluster_id].append(baseline_id)
            
            # Agreement score = fraction of baselines that agree with majority
            max_agreement = max(len(v) for v in cluster_counts.values())
            agreement_score = max_agreement / len(baseline_ids)
            
            # Node is controversial if less than 2/3 agreement
            is_controversial = agreement_score < 0.67
            
            # Summary
            placement_summary = json.dumps(placements)
            
            cursor.execute("""
                INSERT INTO node_consensus (node_id, agreement_score, is_controversial, placement_summary)
                VALUES (?, ?, ?, ?)
            """, (node_id, agreement_score, is_controversial, placement_summary))
        
        self.conn.commit()
    
    # =========================================================================
    # Current Clustering Management
    # =========================================================================
    
    def start_from_baseline(self, baseline_id: str):
        """Initialize current clustering from a baseline"""
        cursor = self.conn.cursor()
        
        # Get baseline clustering
        cursor.execute("SELECT clustering FROM baselines WHERE id = ?", (baseline_id,))
        row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"Baseline '{baseline_id}' not found")
        
        clustering = json.loads(row['clustering'])
        
        # Clear current clustering
        cursor.execute("DELETE FROM current_clustering")
        
        # Copy from baseline
        for cluster_id, nodes in clustering.items():
            for node_id in nodes:
                cursor.execute("""
                    INSERT INTO current_clustering (node_id, cluster_id, source)
                    VALUES (?, ?, ?)
                """, (str(node_id), cluster_id, f"baseline:{baseline_id}"))
        
        self.conn.commit()
        
        print(f"✓ Started from baseline '{baseline_id}'")
    
    def get_current_clustering(self) -> Dict[str, List[str]]:
        """Get current clustering state"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT node_id, cluster_id FROM current_clustering")
        
        clustering = defaultdict(list)
        for row in cursor.fetchall():
            clustering[row['cluster_id']].append(row['node_id'])
        
        return dict(clustering)
    
    def move_node(self, node_id: str, to_cluster: str) -> bool:
        """Move a node to a different cluster"""
        cursor = self.conn.cursor()
        
        # Get current cluster
        cursor.execute("SELECT cluster_id FROM current_clustering WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        
        if not row:
            return False
        
        from_cluster = row['cluster_id']
        
        # Update
        cursor.execute("""
            UPDATE current_clustering 
            SET cluster_id = ?, source = 'llm'
            WHERE node_id = ?
        """, (to_cluster, node_id))
        
        self.conn.commit()
        
        # Log action
        self._log_action('move', {
            'node': node_id,
            'from': from_cluster,
            'to': to_cluster
        })
        
        return True
    
    def merge_clusters(self, cluster_a: str, cluster_b: str) -> str:
        """Merge two clusters"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE current_clustering 
            SET cluster_id = ?, source = 'llm'
            WHERE cluster_id = ?
        """, (cluster_a, cluster_b))
        
        self.conn.commit()
        
        self._log_action('merge', {'from': cluster_b, 'to': cluster_a})
        
        return cluster_a
    
    # =========================================================================
    # Query Methods - Token Efficient!
    # =========================================================================
    
    def get_graph_summary(self) -> str:
        """Get high-level graph summary"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as cnt FROM nodes")
        total_nodes = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(*) as cnt FROM edges")
        total_edges = cursor.fetchone()['cnt']
        
        # Current clustering stats
        cursor.execute("SELECT COUNT(DISTINCT cluster_id) as cnt FROM current_clustering")
        num_clusters = cursor.fetchone()['cnt']
        
        # Calculate metrics
        metrics = self._get_current_metrics()
        
        summary = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "num_clusters": num_clusters,
            "cohesion": round(metrics['cohesion'], 4),
            "coupling": round(metrics['coupling'], 4),
            "internal_edges": metrics['internal'],
            "external_edges": metrics['external']
        }
        
        return json.dumps(summary, indent=2)
    
    def get_baseline_comparison(self) -> str:
        """Compare all baseline algorithms - key for Analyst Agent"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, algorithm, turbomq, mojo_fm, cohesion, coupling, num_clusters
            FROM baselines
            ORDER BY COALESCE(turbomq, -1) DESC, cohesion DESC
        """)
        
        baselines = []
        for row in cursor.fetchall():
            baselines.append({
                "id": row['id'],
                "algorithm": row['algorithm'],
                "turbomq": row['turbomq'],
                "mojo_fm": row['mojo_fm'],
                "cohesion": round(row['cohesion'], 4) if row['cohesion'] else None,
                "coupling": round(row['coupling'], 4) if row['coupling'] else None,
                "num_clusters": row['num_clusters']
            })
        
        return json.dumps(baselines, indent=2)
    
    def get_best_baseline(self) -> str:
        """Get the baseline with best TurboMQ (falls back to cohesion if no TurboMQ)"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, algorithm, cohesion, turbomq, num_clusters
            FROM baselines
            ORDER BY COALESCE(turbomq, -1) DESC, cohesion DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        if not row:
            return json.dumps({"error": "No baselines found"})
        
        metric_name = "TurboMQ" if row['turbomq'] is not None else "cohesion"
        metric_val = row['turbomq'] if row['turbomq'] is not None else round(row['cohesion'], 4)
        
        return json.dumps({
            "best_baseline": row['id'],
            "algorithm": row['algorithm'],
            "cohesion": round(row['cohesion'], 4),
            "turbomq": row['turbomq'],
            "num_clusters": row['num_clusters'],
            "recommendation": f"Start from '{row['id']}' as it has the highest {metric_name} ({metric_val})"
        }, indent=2)
    
    def get_controversial_nodes(self, limit: int = 20) -> str:
        """Find nodes where baselines disagree - key insight!"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT node_id, agreement_score, placement_summary
            FROM node_consensus
            WHERE is_controversial = 1
            ORDER BY agreement_score ASC
            LIMIT ?
        """, (limit,))
        
        controversial = []
        for row in cursor.fetchall():
            placements = json.loads(row['placement_summary'])
            controversial.append({
                "node": row['node_id'],
                "agreement": round(row['agreement_score'], 2),
                "placements": placements,
                "conflict": f"Baselines disagree on where to place '{row['node_id']}'"
            })
        
        return json.dumps(controversial, indent=2)
    
    def get_baseline_placement(self, node_id: str) -> str:
        """See where each baseline placed a specific node"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT bp.baseline_id, bp.cluster_id, b.algorithm
            FROM baseline_placements bp
            JOIN baselines b ON bp.baseline_id = b.id
            WHERE bp.node_id = ?
        """, (node_id,))
        
        placements = {}
        for row in cursor.fetchall():
            placements[row['baseline_id']] = {
                "cluster": row['cluster_id'],
                "algorithm": row['algorithm']
            }
        
        # Get current placement
        cursor.execute("SELECT cluster_id FROM current_clustering WHERE node_id = ?", (node_id,))
        current = cursor.fetchone()
        
        return json.dumps({
            "node": node_id,
            "baseline_placements": placements,
            "current_placement": current['cluster_id'] if current else None
        }, indent=2)
    
    def get_cluster_info(self, cluster_id: str) -> str:
        """Get info about a cluster"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT node_id FROM current_clustering 
            WHERE cluster_id = ?
        """, (cluster_id,))
        
        nodes = [row['node_id'] for row in cursor.fetchall()]
        
        # Count internal/external edges
        internal = 0
        external = 0
        external_clusters = defaultdict(int)
        
        for node in nodes:
            cursor.execute("""
                SELECT target FROM edges WHERE source = ?
            """, (node,))
            
            for row in cursor.fetchall():
                cursor.execute("""
                    SELECT cluster_id FROM current_clustering WHERE node_id = ?
                """, (row['target'],))
                target_cluster = cursor.fetchone()
                
                if target_cluster:
                    if target_cluster['cluster_id'] == cluster_id:
                        internal += 1
                    else:
                        external += 1
                        external_clusters[target_cluster['cluster_id']] += 1
        
        return json.dumps({
            "cluster_id": cluster_id,
            "node_count": len(nodes),
            "sample_nodes": nodes[:10],
            "internal_edges": internal,
            "external_edges": external,
            "cohesion": round(internal / (internal + external), 4) if (internal + external) > 0 else 0,
            "connected_to": dict(external_clusters)
        }, indent=2)
    
    def get_problematic_nodes(self, limit: int = 10) -> str:
        """Find nodes that might be misplaced in current clustering"""
        cursor = self.conn.cursor()
        
        # Find nodes with more external than internal edges
        cursor.execute("""
            WITH node_edges AS (
                SELECT 
                    cc.node_id,
                    cc.cluster_id,
                    SUM(CASE WHEN cc2.cluster_id = cc.cluster_id THEN 1 ELSE 0 END) as internal,
                    SUM(CASE WHEN cc2.cluster_id != cc.cluster_id THEN 1 ELSE 0 END) as external
                FROM current_clustering cc
                JOIN edges e ON cc.node_id = e.source
                JOIN current_clustering cc2 ON e.target = cc2.node_id
                GROUP BY cc.node_id, cc.cluster_id
            )
            SELECT 
                node_id,
                cluster_id,
                internal,
                external,
                CAST(external AS REAL) / (internal + external) as external_ratio
            FROM node_edges
            WHERE external > internal
            ORDER BY external_ratio DESC
            LIMIT ?
        """, (limit,))
        
        problematic = []
        for row in cursor.fetchall():
            problematic.append({
                "node": row['node_id'],
                "cluster": row['cluster_id'],
                "internal_edges": row['internal'],
                "external_edges": row['external'],
                "external_ratio": round(row['external_ratio'], 2),
                "issue": "More connections outside cluster than inside"
            })
        
        return json.dumps(problematic, indent=2)
    
    def get_suggested_moves(self, limit: int = 5) -> str:
        """Get suggested node moves based on edge analysis"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            WITH node_cluster_edges AS (
                SELECT 
                    cc1.node_id,
                    cc1.cluster_id as current_cluster,
                    cc2.cluster_id as target_cluster,
                    COUNT(*) as edge_count
                FROM current_clustering cc1
                JOIN edges e ON cc1.node_id = e.source
                JOIN current_clustering cc2 ON e.target = cc2.node_id
                WHERE cc1.cluster_id != cc2.cluster_id
                GROUP BY cc1.node_id, cc1.cluster_id, cc2.cluster_id
            ),
            current_internal AS (
                SELECT cc1.node_id, COUNT(*) as internal_count
                FROM current_clustering cc1
                JOIN edges e ON cc1.node_id = e.source
                JOIN current_clustering cc2 ON e.target = cc2.node_id
                WHERE cc1.cluster_id = cc2.cluster_id
                GROUP BY cc1.node_id
            )
            SELECT 
                nce.node_id,
                nce.current_cluster,
                nce.target_cluster,
                nce.edge_count as edges_to_target,
                COALESCE(ci.internal_count, 0) as edges_to_current
            FROM node_cluster_edges nce
            LEFT JOIN current_internal ci ON nce.node_id = ci.node_id
            WHERE nce.edge_count > COALESCE(ci.internal_count, 0)
            ORDER BY (nce.edge_count - COALESCE(ci.internal_count, 0)) DESC
            LIMIT ?
        """, (limit,))
        
        suggestions = []
        for row in cursor.fetchall():
            suggestions.append({
                "action": f"Move '{row['node_id']}' to '{row['target_cluster']}'",
                "node": row['node_id'],
                "from_cluster": row['current_cluster'],
                "to_cluster": row['target_cluster'],
                "edges_to_target": row['edges_to_target'],
                "edges_to_current": row['edges_to_current'],
                "expected_improvement": row['edges_to_target'] - row['edges_to_current']
            })
        
        return json.dumps(suggestions, indent=2)
    
    def compare_with_baselines(self) -> str:
        """Compare current clustering with all baselines (TurboMQ + cohesion)"""
        cursor = self.conn.cursor()
        
        # Get current metrics
        current_metrics = self._get_current_metrics()
        
        # Get baseline metrics
        cursor.execute("SELECT id, algorithm, cohesion, turbomq, mojo_fm FROM baselines")
        
        comparisons = []
        for row in cursor.fetchall():
            cohesion_diff = current_metrics['cohesion'] - (row['cohesion'] or 0)
            entry = {
                "baseline": row['id'],
                "algorithm": row['algorithm'],
                "baseline_turbomq": row['turbomq'],
                "baseline_cohesion": round(row['cohesion'], 4) if row['cohesion'] else None,
                "baseline_mojo_fm": row['mojo_fm'],
                "current_cohesion": round(current_metrics['cohesion'], 4),
                "cohesion_diff": round(cohesion_diff, 4),
                "status": "better" if cohesion_diff > 0 else "worse" if cohesion_diff < 0 else "same"
            }
            comparisons.append(entry)
        
        return json.dumps({
            "current_metrics": {
                "cohesion": round(current_metrics['cohesion'], 4),
                "internal_edges": current_metrics['internal'],
                "external_edges": current_metrics['external']
            },
            "vs_baselines": comparisons,
            "note": "TurboMQ for current clustering is computed externally via Java jar"
        }, indent=2)
    
    def get_cluster_list(self) -> str:
        """Get list of all current clusters"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT cluster_id, COUNT(*) as node_count
            FROM current_clustering
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
    
    # =========================================================================
    # Metrics and Statistics
    # =========================================================================
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for current clustering"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN cc1.cluster_id = cc2.cluster_id THEN 1 ELSE 0 END) as internal,
                SUM(CASE WHEN cc1.cluster_id != cc2.cluster_id THEN 1 ELSE 0 END) as external
            FROM edges e
            JOIN current_clustering cc1 ON e.source = cc1.node_id
            JOIN current_clustering cc2 ON e.target = cc2.node_id
        """)
        
        row = cursor.fetchone()
        internal = row['internal'] or 0
        external = row['external'] or 0
        total = internal + external
        
        return {
            "internal": internal,
            "external": external,
            "cohesion": internal / total if total > 0 else 0,
            "coupling": external / total if total > 0 else 0
        }
    
    def calculate_current_metrics(self) -> Dict[str, float]:
        """Public method to get current metrics"""
        return self._get_current_metrics()
    
    # =========================================================================
    # History
    # =========================================================================
    
    def _log_action(self, action: str, details: Dict):
        """Log an action"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO history (iteration, action, details, timestamp)
            VALUES (?, ?, ?, ?)
        """, (self._iteration, action, json.dumps(details), datetime.now().isoformat()))
        self.conn.commit()
    
    def start_iteration(self):
        """Start new iteration"""
        self._iteration += 1
    
    def get_history(self) -> List[Dict]:
        """Get action history"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM history ORDER BY id")
        
        return [{
            "id": row['id'],
            "iteration": row['iteration'],
            "action": row['action'],
            "details": json.loads(row['details']) if row['details'] else None,
            "timestamp": row['timestamp']
        } for row in cursor.fetchall()]
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_to_rsf(self, output_path: str):
        """Export current clustering to RSF"""
        clustering = self.get_current_clustering()
        
        with open(output_path, 'w') as f:
            for cluster_id, nodes in clustering.items():
                for node in nodes:
                    f.write(f"contain {cluster_id} {node}\n")
        
        print(f"✓ Exported to {output_path}")
    
    def close(self):
        """Close database"""
        self.conn.close()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the hybrid database"""
    print("=" * 60)
    print("Hybrid Clustering Database Demo")
    print("=" * 60)
    
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
    
    # Initialize database
    db = HybridClusteringDatabase(":memory:")
    db.load_graph(G)
    
    # Add baselines (simulating ACDC, MGMC, Louvain results)
    baseline_acdc = {
        "cluster_A": ["node_0", "node_1", "node_2", "node_3", "node_4"],
        "cluster_B": ["node_5", "node_6", "node_7", "node_8", "node_9"]
    }
    
    baseline_mgmc = {
        "cluster_1": ["node_0", "node_1", "node_2", "node_7"],  # node_7 misplaced
        "cluster_2": ["node_3", "node_4", "node_5", "node_6", "node_8", "node_9"]
    }
    
    baseline_louvain = {
        "comm_0": ["node_0", "node_1", "node_2", "node_3"],
        "comm_1": ["node_4", "node_5"],  # node_4 and node_5 grouped (edge between them)
        "comm_2": ["node_6", "node_7", "node_8", "node_9"]
    }
    
    db.add_baseline("acdc", "ACDC", baseline_acdc, turbomq=0.72)
    db.add_baseline("mgmc", "MGMC", baseline_mgmc, turbomq=0.65)
    db.add_baseline("louvain", "Louvain", baseline_louvain, turbomq=0.68)
    
    print("\n" + "=" * 60)
    print("Baseline Comparison:")
    print(db.get_baseline_comparison())
    
    print("\n" + "=" * 60)
    print("Best Baseline:")
    print(db.get_best_baseline())
    
    print("\n" + "=" * 60)
    print("Controversial Nodes (where baselines disagree):")
    print(db.get_controversial_nodes())
    
    print("\n" + "=" * 60)
    print("Starting from best baseline...")
    db.start_from_baseline("acdc")
    
    print("\nCurrent Clustering Summary:")
    print(db.get_graph_summary())
    
    print("\n" + "=" * 60)
    print("Comparison with Baselines:")
    print(db.compare_with_baselines())
    
    db.close()
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo()
