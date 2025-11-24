# Interactive Analyzer - Function-Based Graph Querying

## What This Is

An AutoGen implementation where:
- ✅ Graph data is **NOT embedded** in prompts
- ✅ Graph is held by a `GraphDataProvider` component
- ✅ Agent **queries the graph** via function calls when it needs data
- ✅ Agent sees **whole graph** when it requests it (not partial views)
- ✅ More natural "agent explores graph" workflow

## How It Works

### Traditional Approach (simple_analyzer.py)
```
You → Send ENTIRE graph in prompt → Agent → Clusters
```
**Problem:** Large graphs = huge prompts

### Interactive Approach (interactive_analyzer.py)
```
You → Tell agent about task
      ↓
Agent → "Let me check the graph..." → Calls get_graph_info()
      ↓
GraphProvider → Returns graph stats
      ↓
Agent → "I need to see edges..." → Calls get_all_edges()
      ↓
GraphProvider → Returns ALL edges (whole graph!)
      ↓
Agent → Analyzes → Creates clusters
```

**Advantage:** 
- Agent only requests data it needs
- But when it requests edges, it gets the WHOLE graph
- No partial views unless agent chooses to query specific nodes
- Graph data stored separately from conversation

## Available Functions

The agent can call these functions to explore the graph:

| Function | What It Does | Returns |
|----------|-------------|---------|
| `get_graph_info()` | Basic statistics | nodes, edges, density, etc. |
| `get_all_nodes()` | List all nodes | Complete node list |
| `get_all_edges()` | **Get ALL edges** | **Entire graph dependencies** |
| `get_node_dependencies(node)` | What node depends on | Outgoing edges |
| `get_node_dependents(node)` | What depends on node | Incoming edges |
| `get_hub_nodes(n)` | Most depended-on nodes | Top N hubs |

**Key:** When agent calls `get_all_edges()`, it sees the **WHOLE graph**, not pieces!

## Quick Start

```bash
# Basic usage
python interactive_analyzer.py your_graph.pkl

# With cloud model
python interactive_analyzer.py your_graph.pkl \
  --model gpt-os:120b-cloud \
  --clusters 10
```

## Example: How Agent Works

### Conversation Flow

**You:**
```
"Please analyze this graph and create 10 clusters."
```

**Agent (thinking):**
```
"I need to understand the graph first."
→ Calls: get_graph_info()
```

**GraphProvider:**
```json
{
  "total_nodes": 856,
  "total_edges": 2341,
  "density": 0.0032,
  "avg_in_degree": 2.7
}
```

**Agent:**
```
"Okay, 856 nodes. Let me see the actual dependencies."
→ Calls: get_all_edges()
```

**GraphProvider:**
```json
[
  {"source": "NodeA", "target": "NodeB"},
  {"source": "NodeA", "target": "NodeC"},
  ... ALL 2341 edges ...
]
```

**Agent:**
```
"Now I can see the whole graph! Let me identify hubs."
→ Calls: get_hub_nodes(10)
```

**GraphProvider:**
```json
[
  {"node": "CoreService", "dependents": 234},
  {"node": "DatabaseLayer", "dependents": 189},
  ...
]
```

**Agent:**
```
"Based on the complete graph I've seen, here are the clusters..."
→ Returns: JSON with 10 clusters
```

## Editing the System Prompt

The system prompt is at the top of `interactive_analyzer.py`:

```python
# ============================================================================
# PROMPTS - EDIT THESE!
# ============================================================================

SYSTEM_PROMPT = """You are an expert in microservice architecture...

You have access to these functions:
- get_graph_info()
- get_all_nodes()
- get_all_edges()  ← Agent calls this to see WHOLE graph!
...

Your task: Analyze and cluster...
"""
```

Just edit this string to change how the agent behaves!

## Adding Custom Functions

Want to add more ways for agent to query the graph?

### Step 1: Add function to GraphDataProvider

```python
class GraphDataProvider:
    # ... existing functions ...
    
    def get_strongly_connected_components(self) -> List[List[str]]:
        """Get strongly connected components"""
        return [list(c) for c in nx.strongly_connected_components(self.graph)]
```

### Step 2: Register it in setup_agents

```python
def _register_functions(self):
    # ... existing registrations ...
    
    register_function(
        provider.get_strongly_connected_components,
        caller=self.assistant,
        executor=self.user_proxy,
        name="get_strongly_connected_components",
        description="Get strongly connected components in the graph"
    )
```

### Step 3: Update system prompt

```python
SYSTEM_PROMPT = """...
You have access to:
- get_graph_info()
- get_strongly_connected_components()  ← Add here
..."""
```

Done! Agent can now call this function.

## Advantages of This Approach

### vs Embedded Prompts (simple_analyzer.py)

| Aspect | Embedded | Interactive |
|--------|----------|-------------|
| Prompt size | Huge (all edges) | Small (task only) |
| Agent control | None | Explores as needed |
| Graph visibility | All at once | Whole graph when requested |
| Flexibility | Low | High |
| Token usage | High upfront | Only what's needed |

### vs Staged Approach (ollama_autogen_analyzer.py)

| Aspect | Staged | Interactive |
|--------|--------|-------------|
| Graph view | Summarized then detailed | Full when agent requests |
| Complexity | High (multi-stage) | Medium (function calling) |
| Agent autonomy | Low (predetermined stages) | High (agent decides) |
| Transparency | Complex logic | Clear function calls |

## When to Use This

**Use `interactive_analyzer.py` when:**
- ✅ You want agent to explore the graph naturally
- ✅ Graph is medium-large (500-5000 nodes)
- ✅ You want to see what data agent requests
- ✅ You want to add custom query functions
- ✅ You want cleaner separation of concerns

**Use `simple_analyzer.py` when:**
- Small graphs (< 500 nodes)
- Want simplest possible approach
- Don't need function calling

**Use `ollama_autogen_analyzer.py` when:**
- Very large graphs (> 5000 nodes)
- Need production-grade with fallbacks
- Want predetermined analysis stages

## Examples

### Example 1: Basic Run

```bash
python interactive_analyzer.py test_graph.pkl
```

Output shows what agent does:
```
📊 Graph loaded: 100 nodes, 250 edges
   Agent can now query this graph via functions

✓ Registered 6 graph query functions for agent

INTERACTIVE AUTOGEN ANALYSIS
Model: gpt-os:120b-cloud
Agent will query the graph via functions

🤖 Starting interactive analysis...

[Agent calls: get_graph_info()]
[Agent calls: get_all_edges()]
[Agent calls: get_hub_nodes()]
[Agent creates clusters]

✓ Analysis completed in 25.3 seconds
```

### Example 2: Different Models

```bash
# Local model
python interactive_analyzer.py graph.pkl --model llama3.2:3b

# Cloud model
python interactive_analyzer.py graph.pkl --model deepseek-v3.1:671b-cloud
```

### Example 3: Custom Clusters

```bash
python interactive_analyzer.py graph.pkl \
  --clusters 15 \
  --output my_15_clusters.json
```

## Understanding the Output

```json
{
  "model": "gpt-os:120b-cloud",
  "approach": "interactive_function_calling",
  "clusters": {
    "cluster_1": ["NodeA", "NodeB", ...],
    ...
  },
  "statistics": {
    "cohesion_score": 0.82,
    "coupling_score": 0.18
  },
  "metadata": {
    "graph_queried_via_functions": true
  }
}
```

The `graph_queried_via_functions: true` confirms agent accessed graph via function calls, not embedded prompts.

## Advanced: See What Agent Does

The conversation log shows all function calls:

```python
# After running analysis
for conv_id, messages in analyzer.assistant.chat_messages.items():
    for msg in messages:
        print(f"{msg['role']}: {msg.get('content', 'Function call')[:100]}")
```

This shows you exactly what data the agent requested!

## Troubleshooting

### Agent doesn't call functions

**Problem:** Some models don't support function calling well.

**Solution:** Try a better model:
```bash
python interactive_analyzer.py graph.pkl --model gpt-os:120b-cloud
```

### Agent calls functions but doesn't cluster

**Problem:** Agent gets stuck exploring.

**Solution:** Edit the system prompt to be more directive:
```python
SYSTEM_PROMPT = """...

IMPORTANT: After calling get_all_edges() to see the graph, 
immediately create the clusters. Don't overthink it.
"""
```

### Function calls timeout

**Problem:** Large graph, function returns too much data.

**Solution:** Modify functions to return subsets:
```python
def get_all_edges(self, limit: int = 5000) -> List[Dict]:
    """Get edges (limited to first N)"""
    edges = list(self.graph.edges())[:limit]
    return [{"source": u, "target": v} for u, v in edges]
```

## Summary

**Key Innovation:**
- Graph is **held separately** by `GraphDataProvider`
- Agent **queries it via functions** when needed
- When agent requests edges, it gets the **WHOLE graph**
- Not embedded in prompts = cleaner, more flexible
- Agent has autonomy to explore as it needs

**Perfect for:**
- Medium-large graphs
- When you want agent autonomy
- When you want to add custom query functions
- When you want clean separation of data and logic

**Try it now:**
```bash
cd autogen_clustering
python interactive_analyzer.py your_graph.pkl --model gpt-os:120b-cloud
```

The agent will explore your graph and cluster it naturally! 🚀

