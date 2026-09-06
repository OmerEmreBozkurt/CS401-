# CS401- — LLM-Guided Louvain Hierarchical Software Clustering

This repository contains the research code for **automatic software architecture recovery** using a two-stage pipeline:

1. **Louvain community detection** — compresses a large dependency graph into manageable *super-nodes*
2. **LLM clustering** — an LLM receives the super-node graph and decides the final cluster assignments

The LLM never sees thousands of raw nodes at once; it only reasons over the Louvain-compressed graph. This keeps prompts short while preserving global structure.

---

## Repository Layout

```
CS401-/
├── graph_builder.py              # Build a .pkl graph from a dependency RSF file
├── louvian/
│   ├── louvain_hierarchical.py           # Louvain algorithm (shared library)
│   ├── louvain_hierarchical_single.py    # Single-agent pipeline (recommended entry point)
│   └── hierarchical_analyzer.py         # Multi-agent pipeline (AutoGen, Ollama)
├── autogen_clustering/           # AutoGen-based LLM orchestration
│   ├── interactive_analyzer_improved.py
│   ├── interactive_analyzer_with_metrics.py
│   ├── prompts/prompts.py
│   └── config.json
├── dataset/                      # Input dependency graphs (RSF format)
│   ├── bash/
│   ├── archstudio/
│   ├── hadoop/
│   └── ... (12 systems total)
├── experiments/                  # Evaluation JAR tools
│   ├── turbomq.jar               # TurboMQ metric
│   ├── mojo.jar                  # MoJo-FM metric
│   └── acdc.jar                  # ACDC baseline
├── metrics/                      # Python metric helpers
├── test_results/                 # Raw JSON/RSF outputs
├── results/                      # Aggregated Excel summaries
└── GA/                           # Genetic algorithm baseline (HYGAR)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.10 | |
| Java | >= 11 | Required for TurboMQ / MoJo JAR tools |
| [Ollama](https://ollama.com) | latest | Local LLM server |
| networkx | >= 3.0 | `pip install networkx` |
| requests | any | `pip install requests` |

### Install Python dependencies

```bash
pip install networkx requests
```

> The single-agent pipeline (`louvain_hierarchical_single.py`) only needs `networkx` and `requests`. No AutoGen or heavy dependencies required.

### Install and start Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen3-vl:235b-cloud
ollama pull qwen3-coder:480b-cloud

# Start the server (runs on http://localhost:11434)
ollama serve
```

---

## Data Format

All dependency graphs are stored as **RSF** (Relational Structure Format) files:

```
# Dependency RSF — one edge per line
depends  source_module  target_module
```

```
# Clustering RSF — ground-truth or output clustering
contain  ClusterName  NodeName
```

The `dataset/` directory contains both `*-dependency.rsf` and `*-clustering.rsf` for each system.

---

## Step-by-Step: Running the Pipeline

### Step 1 — (Optional) Build a graph pickle from RSF

If you already have a `.pkl` file in `test_results/`, skip this step.

```bash
python graph_builder.py dataset/bash/bash-dependency.rsf bash_graph
```

This produces:
- `bash_graph.pkl` — NetworkX graph (input for the clustering pipeline)
- `bash_graph_edges.txt`, `bash_graph_adj.json`, `bash_graph.graphml` — auxiliary exports

---

### Step 2 — Run the Louvain + LLM Single-Agent Pipeline

`louvian/louvain_hierarchical_single.py` is the **recommended entry point**. It:
- Loads the graph (from `.pkl` or directly from `--dependency-rsf`)
- Applies hierarchical Louvain coarsening
- Sends the compressed super-node graph **once** to the LLM
- Expands clusters back to original nodes
- Computes TurboMQ and MoJo-FM (if JAR files and reference RSF are provided)

#### Minimal run (Louvain only, no LLM)

```bash
python louvian/louvain_hierarchical_single.py \
  --dependency-rsf dataset/bash/bash-dependency.rsf \
  --skip-llm \
  --output test_results/bash_louvain_only.json
```

#### Standard run with local LLM

```bash
python louvian/louvain_hierarchical_single.py \
  --dependency-rsf dataset/bash/bash-dependency.rsf \
  --model qwen3-coder:480b-cloud \
  --louvain-levels 1 \
  --louvain-resolution 1.0 \
  --output test_results/bash_result.json \
  --output-rsf test_results/bash_result.rsf
```

#### Full run with metrics (TurboMQ + MoJo-FM)

```bash
python louvian/louvain_hierarchical_single.py \
  test_results/bash-g.pkl \
  --dependency-rsf dataset/bash/bash-dependency.rsf \
  --reference-rsf  dataset/bash/bash-clustering.rsf \
  --model qwen3-coder:480b-cloud \
  --louvain-levels 1 \
  --louvain-resolution 1.0 \
  --output test_results/bash_result.json \
  --output-rsf test_results/bash_result.rsf \
  --turbomq-jar experiments/turbomq.jar
```

#### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `graph_file` | *(optional)* | Path to a `.pkl` NetworkX graph |
| `--dependency-rsf` | *(required)* | Dependency RSF file |
| `--reference-rsf` | — | Ground-truth RSF for MoJo-FM score |
| `--model` | `None` | Ollama model name (omit to use `--skip-llm`) |
| `--ollama-url` | `http://localhost:11434` | Ollama server address |
| `--louvain-levels` | `1` | Coarsening depth (more levels = fewer super-nodes) |
| `--louvain-resolution` | `1.0` | Higher = more, smaller communities |
| `--output` | *(required)* | JSON output path |
| `--output-rsf` | auto | RSF clustering output (defaults to same name as `--output`) |
| `--turbomq-jar` | — | Path to `turbomq.jar` for TurboMQ score |
| `--skip-llm` | `False` | Skip LLM, use Louvain clusters directly |
| `--random-seed` | `42` | Louvain random seed |

---

### Step 3 — Interpret the Output

The JSON output file contains:

```json
{
  "clusters": 8,
  "super_nodes_used": 12,
  "turbomq": 0.74,
  "mojofm": 61.3,
  "final_clustering": {
    "cluster_0": ["bash_main", "bash_execute"],
    "cluster_1": ["..."]
  },
  "llm_response": "..."
}
```

The RSF output is directly usable as input to other metric tools:

```
contain  cluster_0  bash_main
contain  cluster_0  bash_execute
```

---

## Alternative: Multi-Agent Pipeline (AutoGen + Ollama)

`louvian/hierarchical_analyzer.py` uses an **AutoGen multi-agent loop**: a *Clustering Agent* proposes clusters and an *Inspector Agent* iteratively refines them based on TurboMQ feedback.

```bash
python louvian/hierarchical_analyzer.py test_results/bash-g.pkl \
  --dependency-rsf dataset/bash/bash-dependency.rsf \
  --reference-rsf  dataset/bash/bash-clustering.rsf \
  --model qwen3-coder:480b-cloud \
  --louvain-levels 2 \
  --iterations 50 \
  --output test_results/bash_hierarchical_result.json
```

**Additional requirement:**

```bash
pip install pyautogen
```

> The multi-agent pipeline runs until TurboMQ converges or `--iterations` is reached. It is slower but can produce higher-quality results.

---

## Evaluated Systems

| System | Domain | Notes |
|---|---|---|
| bash | Shell interpreter | Small, good for quick tests |
| archstudio | IDE framework | Medium size |
| hadoop | Distributed computing | Large |
| chromium | Browser | Very large (>100 MB graphs, not in git) |
| itk | Medical imaging toolkit | Large |
| camel | Integration framework | Medium |
| cxf, lucene, nutch, openjpa, struts2, wicket | Various Java | Medium |

Pre-built `.pkl` graph files for bash, archstudio, and hadoop are in `test_results/`. Chromium graphs are very large and are excluded from git — rebuild with `graph_builder.py` if needed.

---

## Metrics

| Metric | Tool | Description |
|---|---|---|
| **TurboMQ** | `experiments/turbomq.jar` | Modularity of the clustering (higher is better) |
| **MoJo-FM** | `experiments/mojo.jar` | Edit-distance similarity to ground truth (0–100, higher is better) |

Both are computed automatically when you pass `--turbomq-jar` and `--reference-rsf`. Java must be on your `PATH`.

---

## Quick Experiment Checklist

```
[ ] 1. Install Python deps:   pip install networkx requests
[ ] 2. Install + start Ollama, pull a model (e.g. ollama pull qwen3-coder:480b-cloud)
[ ] 3. Verify Java is available:  java -version
[ ] 4. Run single-agent pipeline on bash (smallest system):
        python louvian/louvain_hierarchical_single.py \
          --dependency-rsf dataset/bash/bash-dependency.rsf \
          --model qwen3-coder:480b-cloud \
          --output test_results/bash_result.json
[ ] 5. Inspect test_results/bash_result.json for cluster assignments and scores
```

