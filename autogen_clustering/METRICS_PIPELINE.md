# Interactive Analyzer with Metrics - Usage Guide

## Overview

Enhanced pipeline that includes:
1. **Clustering Agent** - Creates microservice clusters
2. **Metrics Agent** - Calculates TurboMQ and MoJo-FM scores
3. **Improvement Loop** - Analyzes scores and improves clustering

## Pipeline Flow

```
Graph → Cluster → Evaluate Metrics → Analyze → Improve → Done
         ↓              ↓                ↓         ↓
    Clustering    TurboMQ + MoJo    Analysis   Adjusted
    Agent         Calculator        Agent      Clusters
```

## Quick Start

### Basic Usage (with metrics)

```bash
python interactive_analyzer_with_metrics.py graph.pkl \
  --dependency-rsf ../dataset/bash/bash-dependency.rsf \
  --model gpt-oss:120b-cloud \
  --clusters 10
```

### With Improvement Iterations

```bash
python autogen_clustering/interactive_analyzer_with_metrics.py bash-g.pkl \
  --dependency-rsf dataset/bash/bash-dependency.rsf \
  --model gpt-oss:120b-cloud \
  --clusters 10 \
  --iterations 2
```

**Note:** Make sure to use the correct path to your dependency RSF file. The file must exist and be in the correct format (`depends source target` per line).

### With Reference Clustering (for MoJo-FM)

```bash
python interactive_analyzer_with_metrics.py graph.pkl \
  --dependency-rsf ../dataset/bash/bash-dependency.rsf \
  --reference-rsf ../dataset/bash/bash-mgmc-clustered.rsf \
  --model gpt-oss:120b-cloud \
  --clusters 10 \
  --iterations 2
```

## Command Line Arguments

```
Required:
  graph_file              Path to pickled graph file (.pkl)

Optional:
  --dependency-rsf FILE   Path to dependency RSF file (required for metrics)
  --reference-rsf FILE    Path to reference clustering RSF (for MoJo-FM)
  --model MODEL          Ollama model (default: gpt-oss:120b-cloud)
  --clusters N           Number of clusters (default: 10)
  --iterations N         Max improvement iterations (default: 1)
  --output FILE          Output JSON file (default: clusters_with_metrics.json)
  --host URL             Ollama host (default: http://localhost:11434)
```

## What It Does

### Step 1: Clustering
- Clustering agent queries graph via functions
- Creates initial clusters
- Outputs JSON with cluster assignments

### Step 2: Metrics Evaluation
- Converts clusters to RSF format
- Runs `java -jar turbomq.jar` to calculate TurboMQ
- Runs `java -jar mojo.jar` to calculate MoJo-FM (if reference provided)
- Returns scores

### Step 3: Analysis
- Metrics agent analyzes the scores
- Identifies issues (low cohesion, high coupling, etc.)
- Provides recommendations

### Step 4: Improvement (if iterations > 1)
- Clustering agent receives feedback
- Adjusts clusters based on metrics
- Re-evaluates metrics
- Repeats until max iterations reached

## Metrics Explained

### TurboMQ (Turbo Modularity Quality)
- **Range**: 0 to 1
- **Higher is better**
- Measures cohesion within clusters
- Formula: Average of `u / (u + 0.5 * exdep)` for each cluster
  - `u` = intra-cluster dependencies
  - `exdep` = inter-cluster dependencies
- **Good score**: > 0.7
- **Poor score**: < 0.5

### MoJo-FM (Move-and-Join with F-Measure)
- **Range**: 0 to 1 (or distance measure)
- **Lower is better**
- Measures distance from reference clustering
- Requires a reference clustering RSF file
- **Good score**: Close to 0 (matches reference)
- **Poor score**: High (doesn't match reference)

## Example Output

```json
{
  "model": "gpt-oss:120b-cloud",
  "approach": "interactive_with_metrics",
  "clusters": {
    "cluster_1": ["node1", "node2", ...],
    ...
  },
  "statistics": {
    "num_clusters": 10,
    "cohesion_score": 0.82,
    "coupling_score": 0.18
  },
  "metrics": {
    "turbomq": 0.756,
    "mojo_fm": 0.234
  },
  "metadata": {
    "iterations": 2
  }
}
```

## Requirements

1. **Graph file** (.pkl) - From graph_builder.py
2. **Dependency RSF** - Original dependency file in RSF format
3. **Java** - For running TurboMQ and MoJo-FM jars
4. **Jar files** - Must be in `experiments/` directory:
   - `turbomq.jar`
   - `mojo.jar` (optional, for MoJo-FM)

## Troubleshooting

### "TurboMQ jar not found"
```bash
# Check if jar exists
ls experiments/turbomq.jar

# If missing, download or copy from your experiments folder
```

### "Java not found"
```bash
# Check Java installation
java -version

# Install if needed (varies by OS)
```

### "MoJo-FM skipped"
- This is normal if `--reference-rsf` is not provided
- MoJo-FM requires a reference clustering to compare against

### Metrics return None
- Check that dependency RSF file exists and is readable
- Verify jar files are in `experiments/` directory
- Check Java can run the jars: `java -jar experiments/turbomq.jar --help`

## Comparison with Basic Analyzer

| Feature | Basic Analyzer | With Metrics |
|---------|---------------|--------------|
| Clustering | ✅ | ✅ |
| Graph Query | ✅ | ✅ |
| TurboMQ | ❌ | ✅ |
| MoJo-FM | ❌ | ✅ |
| Improvement Loop | ❌ | ✅ |
| Metrics Analysis | ❌ | ✅ |

## Tips

1. **Start with 1 iteration** to see initial scores
2. **Use 2-3 iterations** if scores need improvement
3. **Provide reference RSF** if you have a ground truth clustering
4. **Check TurboMQ first** - it's the most important metric
5. **MoJo-FM is optional** - only needed if comparing to reference

## Example Workflow

```bash
# 1. Build graph
python utils/graph_builder.py ../dataset/bash/bash-dependency_compact.rsf bash_graph

# 2. Run with metrics (1 iteration)
python interactive_analyzer_with_metrics.py bash_graph.pkl \
  --dependency-rsf ../dataset/bash/bash-dependency.rsf \
  --model gpt-oss:120b-cloud \
  --clusters 10 \
  --iterations 1 \
  --output initial_clusters.json

# 3. Check scores in output
cat initial_clusters.json | grep -A 5 "metrics"

# 4. If scores need improvement, run with more iterations
python interactive_analyzer_with_metrics.py bash_graph.pkl \
  --dependency-rsf ../dataset/bash/bash-dependency.rsf \
  --model gpt-oss:120b-cloud \
  --clusters 10 \
  --iterations 2 \
  --output improved_clusters.json
```

## Next Steps

After getting clusters with metrics:
1. Review TurboMQ score (aim for >0.7)
2. Check cluster sizes (should be balanced)
3. Use the RSF output for further analysis
4. Compare with other clustering methods

---

**The complete pipeline: Cluster → Evaluate → Analyze → Improve → Done!** 🚀

