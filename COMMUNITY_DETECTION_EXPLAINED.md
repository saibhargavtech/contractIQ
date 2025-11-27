# 🔍 Community Detection & Louvain Resolution - Explained

## What is Community Detection?

**Community Detection** (also called **graph clustering** or **modularity optimization**) is the process of finding groups of nodes in a graph that are more densely connected to each other than to nodes outside the group.

### Real-World Analogy
Think of a social network:
- **Nodes** = People
- **Edges** = Friendships
- **Communities** = Friend groups (people who know each other well)

In your contract knowledge graph:
- **Nodes** = Entities (e.g., "ASC 606", "Revenue Contract", "IBM")
- **Edges** = Relationships (e.g., "governed_by", "audited_by")
- **Communities** = Thematic groups (e.g., "Revenue Recognition", "Vendor Management", "Compliance")

### Visual Example

```
Before Community Detection:
┌─────────────────────────────────────┐
│  ASC 606 ──→ Revenue Contract       │
│  IFRS 15 ──→ Revenue Contract       │
│  IBM ──→ Service Agreement          │
│  Microsoft ──→ Service Agreement    │
│  Auditor A ──→ Financial Statement  │
│  Auditor B ──→ Financial Statement   │
└─────────────────────────────────────┘

After Community Detection:
┌─────────────────────────────────────┐
│  Cluster 0: Revenue Recognition     │
│    • ASC 606                        │
│    • IFRS 15                        │
│    • Revenue Contract               │
│                                     │
│  Cluster 1: Vendor Management       │
│    • IBM                            │
│    • Microsoft                      │
│    • Service Agreement              │
│                                     │
│  Cluster 2: Audit & Compliance      │
│    • Auditor A                      │
│    • Auditor B                      │
│    • Financial Statement            │
└─────────────────────────────────────┘
```

### Why Use Community Detection?

1. **Thematic Organization**: Groups related entities together
2. **Summarization**: Each cluster gets a summary (e.g., "Revenue Recognition Risk")
3. **Query Efficiency**: When answering questions, we can focus on relevant clusters
4. **Pattern Discovery**: Reveals hidden relationships and themes
5. **CFO Analytics**: Clusters become "risk themes" or "KPI categories"

---

## The Louvain Algorithm

The **Louvain algorithm** is a popular method for community detection that's:
- **Fast**: O(n log n) complexity for sparse graphs
- **Scalable**: Works on graphs with millions of nodes
- **Quality**: Produces high-quality communities (high modularity)
- **Deterministic**: With a fixed random seed, results are reproducible

### How Louvain Works (Simplified)

```
Step 1: Initialize
  - Each node starts in its own community

Step 2: Local Optimization
  - For each node, try moving it to neighboring communities
  - Keep the move that increases modularity the most
  - Repeat until no moves improve modularity

Step 3: Aggregation
  - Merge communities into "super-nodes"
  - Create a new graph where communities are nodes

Step 4: Repeat
  - Go back to Step 2 on the aggregated graph
  - Continue until no improvement

Step 5: Refinement
  - Unfold the communities back to original nodes
  - Final assignment: {node: community_id}
```

### Example Walkthrough

**Initial Graph:**
```
A ── B ── C
│    │    │
D ── E ── F
```

**Iteration 1:**
- Node A: Try moving to B's community → improves modularity ✓
- Node B: Already optimal
- Continue for all nodes...

**Result after Iteration 1:**
- Community 0: {A, B, C}
- Community 1: {D, E, F}

**Iteration 2 (Aggregated):**
- Super-node 0 (represents {A,B,C})
- Super-node 1 (represents {D,E,F})
- Check if merging improves modularity...

**Final Result:**
- Two communities (or one, depending on connections)

---

## Louvain Resolution Parameter

### What is Resolution?

The **resolution parameter** (γ, gamma) controls the **granularity** of community detection:
- **Lower resolution** → Fewer, larger communities (broader themes)
- **Higher resolution** → More, smaller communities (fine-grained themes)

### The Mathematical Formula

The Louvain algorithm optimizes **modularity**:

```
Q = (1/2m) × Σ_ij [A_ij - γ × (k_i × k_j)/(2m)] × δ(c_i, c_j)

Where:
- m = total edge weight
- A_ij = edge weight between nodes i and j
- k_i = degree (total connections) of node i
- k_j = degree of node j
- γ = resolution parameter (LOUVAIN_RESOLUTION)
- c_i = community of node i
- c_j = community of node j
- δ(c_i, c_j) = 1 if same community, else 0
```

### Understanding the Resolution Parameter

The resolution parameter (γ) modifies the **null model** - the expected number of edges between nodes if connections were random.

**Without resolution (γ = 1.0):**
- Expected edges = (k_i × k_j) / (2m)
- Standard modularity

**With resolution (γ ≠ 1.0):**
- Expected edges = γ × (k_i × k_j) / (2m)
- **γ > 1.0**: Higher expected edges → harder to form communities → more communities
- **γ < 1.0**: Lower expected edges → easier to form communities → fewer communities

### Resolution Impact Examples

#### Resolution = 0.5 (Low Resolution)
```
Result: 2-3 large communities

Cluster 0: All Revenue-Related Entities (15 nodes)
  - ASC 606, IFRS 15, Revenue Contract, etc.

Cluster 1: All Vendor-Related Entities (12 nodes)
  - IBM, Microsoft, Service Agreements, etc.

Cluster 2: All Compliance Entities (8 nodes)
  - Auditors, Standards, Regulations, etc.
```

#### Resolution = 1.25 (Default - Medium Resolution)
```
Result: 5-8 medium communities

Cluster 0: Revenue Recognition Standards (5 nodes)
  - ASC 606, IFRS 15, Revenue Contract

Cluster 1: Vendor Contracts (4 nodes)
  - IBM, Service Agreement, Payment Terms

Cluster 2: Audit Entities (3 nodes)
  - Auditor A, Financial Statement

Cluster 3: Compliance Frameworks (4 nodes)
  - GDPR, SOX, Regulatory Standards

... (more clusters)
```

#### Resolution = 2.0 (High Resolution)
```
Result: 10-15 small communities

Cluster 0: ASC 606 Entities (2 nodes)
  - ASC 606, Revenue Contract

Cluster 1: IFRS 15 Entities (2 nodes)
  - IFRS 15, Revenue Recognition

Cluster 2: IBM Contracts (2 nodes)
  - IBM, Service Agreement A

Cluster 3: Microsoft Contracts (2 nodes)
  - Microsoft, Service Agreement B

... (many small clusters)
```

### Visual Comparison

```
Resolution = 0.5          Resolution = 1.25         Resolution = 2.0
┌─────────────┐          ┌──────┬──────┐          ┌──┬──┬──┬──┐
│             │          │      │     │          │  │  │  │  │
│  Large      │          │ Med  │ Med │          │S │S │S │S │
│  Cluster    │          │      │     │          │m │m │m │m │
│             │          └──────┴──────┘          └──┴──┴──┴──┘
└─────────────┘          ┌──────┬──────┐          ┌──┬──┬──┬──┐
                         │      │     │          │S │S │S │S │
                         │ Med  │ Med │          │m │m │m │m │
                         └──────┴──────┘          └──┴──┴──┴──┘
```

---

## How Resolution Works in Your Project

### Default Setting
```python
LOUVAIN_RESOLUTION = 1.25  # Default value
```

### Why 1.25?

1. **Balanced Granularity**: Not too coarse (would miss themes) or too fine (would create noise)
2. **Financial Domain**: Financial documents have natural thematic groupings (revenue, compliance, vendors)
3. **CFO Analytics**: Medium-sized clusters map well to "risk themes" or "KPI categories"
4. **Empirical Testing**: Found to work well for contract analysis use cases

### Tuning Resolution for Different Use Cases

#### **High-Level Strategic Analysis** (Resolution = 0.5 - 0.8)
- **Use Case**: Executive summaries, board reports
- **Result**: 2-4 broad themes
- **Example Clusters**: "Revenue", "Compliance", "Operations", "Risk"

#### **Detailed Risk Analysis** (Resolution = 1.5 - 2.0)
- **Use Case**: Deep dive into specific risk areas
- **Result**: 10-15 specific risk clusters
- **Example Clusters**: "ASC 606 Compliance", "Vendor Concentration", "Payment Terms Risk"

#### **Balanced Analysis** (Resolution = 1.0 - 1.5) ⭐ **Recommended**
- **Use Case**: General contract analysis
- **Result**: 5-10 thematic clusters
- **Example Clusters**: "Revenue Recognition", "Vendor Management", "Audit & Compliance"

---

## Practical Example from Your Code

### Code Implementation
```python
def detect_communities(G, resolution=None, random_state=None, use_ontology_for_clustering=False):
    if G.number_of_nodes() == 0:
        return G, {}
    
    res = resolution if resolution is not None else LOUVAIN_RESOLUTION  # Default: 1.25
    rs  = random_state if random_state is not None else LOUVAIN_RANDOM_STATE  # Default: 202
    
    # Convert to undirected graph (Louvain works on undirected graphs)
    Gu = _to_weighted_undirected(G, use_ontology_for_clustering)
    
    # Run Louvain algorithm
    part = community_louvain.best_partition(
        Gu, 
        resolution=float(res),      # ← Resolution parameter here
        random_state=int(rs),       # ← Random seed for reproducibility
        weight="weight"             # ← Edge weights (ontology-boosted if enabled)
    )
    
    # Assign community IDs to nodes
    nx.set_node_attributes(G, part, 'community')
    return G, part
```

### What Happens After Community Detection?

1. **Each node gets a community ID**:
   ```python
   partition = {
       "ASC 606": 0,
       "IFRS 15": 0,
       "Revenue Contract": 0,
       "IBM": 1,
       "Microsoft": 1,
       "Service Agreement": 1,
       ...
   }
   ```

2. **Clusters are summarized**:
   ```python
   CLUSTER_SUMMARIES = {
       0: {
           "KPI/Risk Label": "Revenue Recognition Risk",
           "Cluster Summary": "Entities related to revenue recognition standards..."
       },
       1: {
           "KPI/Risk Label": "Vendor Management",
           "Cluster Summary": "Vendor contracts and service agreements..."
       }
   }
   ```

3. **Used in Q&A**:
   - When you ask "What are the revenue recognition risks?"
   - System finds Cluster 0 (Revenue Recognition)
   - Includes cluster summary + relevant edges in LLM context
   - Provides comprehensive answer

---

## Key Takeaways for Interviews

### What to Say About Community Detection:

**"Community detection groups related entities in our knowledge graph into thematic clusters. We use the Louvain algorithm because it's fast, scalable, and produces high-quality communities. Each cluster represents a theme like 'Revenue Recognition' or 'Vendor Management', which we then summarize using LLMs to create CFO-ready insights."**

### What to Say About Resolution:

**"The resolution parameter controls how granular the clustering is. Lower values create fewer, broader communities - useful for high-level analysis. Higher values create more, smaller communities - useful for detailed risk analysis. We default to 1.25, which provides a balanced granularity that works well for financial contract analysis, creating 5-10 thematic clusters that map naturally to risk themes and KPI categories."**

### Technical Details to Mention:

1. **Algorithm**: Louvain (modularity optimization)
2. **Complexity**: O(n log n) - efficient for large graphs
3. **Reproducibility**: Random state seed ensures deterministic results
4. **Weighted Edges**: We can boost financial relations (e.g., "recognised_under" gets 1.5x weight)
5. **Undirected Conversion**: Louvain works on undirected graphs, but we preserve the directed graph for queries

---

## Common Interview Questions & Answers

### Q: Why not use K-means or hierarchical clustering?
**A:** "K-means requires knowing the number of clusters upfront, which we don't. Hierarchical clustering is O(n²) and doesn't scale. Louvain automatically finds the optimal number of communities and scales to millions of nodes."

### Q: How do you choose the resolution parameter?
**A:** "We start with 1.25 as a balanced default. For strategic analysis, we lower it to 0.5-0.8 for broader themes. For detailed risk analysis, we raise it to 1.5-2.0 for fine-grained clusters. The parameter is exposed in the UI so domain experts can tune it based on their needs."

### Q: What if resolution doesn't give good results?
**A:** "We can enable ontology-weighted clustering, which boosts financial relations (like 'recognised_under' gets 1.5x weight). This helps financial relations form stronger communities. We also use the random state parameter to ensure reproducibility, and we can try different seeds if needed."

### Q: How does community detection help with Q&A?
**A:** "When a user asks a question, we identify which clusters are relevant. Instead of searching the entire graph, we focus on those clusters, including their summaries and key entities. This provides richer context to the LLM and improves answer quality while reducing token usage."

---

*This document provides a clear, interview-ready explanation of community detection and Louvain resolution. Use it to confidently explain these concepts to interviewers!*


