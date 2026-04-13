"""
Hybrid Baseline-Guided Clustering System

A multi-agent system that:
1. Compares baseline algorithms (ACDC, MGMC, Louvain)
2. Starts from the best baseline
3. Uses LLM agents to improve clustering
4. Focuses on controversial nodes where baselines disagree
"""

from .hybrid_database import HybridClusteringDatabase
from .hybrid_analyzer import HybridBaselineAnalyzer

__all__ = ['HybridClusteringDatabase', 'HybridBaselineAnalyzer']
