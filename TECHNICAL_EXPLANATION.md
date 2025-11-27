# 📊 Contract GraphRAG Studio - Comprehensive Technical Explanation

## 🎯 Project Overview

**Contract GraphRAG Studio** is a sophisticated **Knowledge Graph-based Retrieval-Augmented Generation (GraphRAG)** system designed for financial contract analysis and CFO-level business intelligence. The system extracts structured knowledge from unstructured financial documents (PDFs, Word docs, text files) and builds a semantic knowledge graph that enables intelligent querying, risk analysis, and strategic insights.

### Core Value Proposition
- **Automated Knowledge Extraction**: Transforms unstructured contract documents into structured knowledge graphs
- **Intelligent Querying**: Natural language Q&A over contract knowledge with graph-based reasoning
- **CFO-Ready Analytics**: Generates 30+ strategic questions across 6 financial dimensions
- **Hybrid Search**: Combines graph-based reasoning with vector semantic search
- **RDF/Ontology Support**: Exports to semantic web standards for interoperability

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Gradio UI  │  │ Streamlit    │  │  RDF Export  │      │
│  │   (Primary)  │  │  Dashboard   │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Processing Pipeline                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Document    │→ │  Entity/Rel  │→ │  Graph       │      │
│  │  Parser      │  │  Extraction  │  │  Builder     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                ↓                    ↓               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  PDF/DOCX    │  │  LLM + Rule   │  │  NetworkX    │      │
│  │  Text Parser │  │  Based        │  │  DiGraph     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Knowledge Graph Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Community   │  │  Cluster     │  │  Graph       │      │
│  │  Detection   │  │  Summaries   │  │  Context     │      │
│  │  (Louvain)   │  │  (LLM)       │  │  Memory      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Query & Retrieval Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Graph-based │  │  Vector      │  │  Hybrid      │      │
│  │  Q&A         │  │  Search      │  │  Fusion      │      │
│  │  (LLM)       │  │  (LlamaIndex)│  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Analytics & Export Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CFO         │  │  Metrics     │  │  RDF/CSV     │      │
│  │  Insights    │  │  Computation │  │  Export      │      │
│  │  (JSONL)     │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Graph Library** | NetworkX 3.0+ | Directed graph representation, graph algorithms |
| **LLM** | OpenAI GPT-4o | Entity extraction, relation extraction, summarization, Q&A |
| **Community Detection** | python-louvain | Modularity-based clustering (Louvain algorithm) |
| **Vector Search** | LlamaIndex | Semantic search with OpenAI embeddings |
| **UI Framework** | Gradio 4.0+ | Interactive web interface |
| **RDF/Ontology** | rdflib 6.0+ | Semantic web standards (Turtle, JSON-LD) |
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Visualization** | Matplotlib | Graph visualization and metrics |

### Optional Dependencies
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document parsing
- **rdflib**: RDF/OWL ontology support
- **llama-index**: Vector search capabilities

---

## 📐 Graph Tuning Parameters (Detailed)

### 1. **Extraction Parameters**

#### `EXTRACT_TEMP` (Default: 0.4)
- **Type**: Float (0.0 - 1.0)
- **Purpose**: Controls LLM creativity during entity/relation extraction
- **Impact**:
  - **Low (0.0-0.3)**: More deterministic, consistent extractions, fewer hallucinations
  - **Medium (0.4-0.6)**: Balanced creativity and accuracy
  - **High (0.7-1.0)**: More diverse extractions, higher risk of noise
- **Use Case**: Lower for financial documents requiring precision, higher for exploratory analysis

#### `STRICT_PROMPT` (Default: True)
- **Type**: Boolean
- **Purpose**: Enforces strict extraction rules in LLM prompts
- **Behavior**:
  - **True**: Rejects edges without clear relation verbs, avoids co-occurrence-only edges
  - **False**: More permissive, allows weaker connections
- **Impact**: Higher precision vs. higher recall trade-off

#### `REQUIRE_EVIDENCE` (Default: True)
- **Type**: Boolean
- **Purpose**: Requires evidence text for each extracted relation
- **Impact**: 
  - **True**: Only edges with source sentences are kept (higher quality)
  - **False**: Allows edges without provenance (lower quality but more coverage)

#### `CHUNK_MAX_CHARS` (Default: 3500)
- **Type**: Integer (2000-5000)
- **Purpose**: Maximum characters per text chunk for LLM processing
- **Trade-offs**:
  - **Smaller chunks**: More API calls, better context focus, higher cost
  - **Larger chunks**: Fewer calls, more context per call, risk of truncation
- **Optimal Range**: 3000-4000 for GPT-4o (context window: 128k tokens)

### 2. **Fallback & Co-occurrence Parameters**

#### `ENABLE_FALLBACK_COOCC` (Default: False)
- **Type**: Boolean
- **Purpose**: Enables co-occurrence-based relation extraction when LLM fails
- **Algorithm**: Extracts capitalized entities from same sentence, creates "CO_OCCURS_IN_SENTENCE" relations
- **Use Case**: Backup when LLM extraction yields no results

#### `MAX_COOCC_PAIRS` (Default: 5)
- **Type**: Integer (0-10)
- **Purpose**: Maximum co-occurrence pairs extracted per sentence
- **Impact**: Controls graph density from fallback extraction
- **Formula**: `C(n,2)` combinations limited to this value

### 3. **Relation Filtering Parameters**

#### `RELATION_POLICY` (Default: "Off")
- **Type**: Enum ("Off" | "Standard" | "Strict")
- **Purpose**: Whitelist-based relation filtering
- **Policies**:
  - **Off**: Accepts all relation types
  - **Standard**: Allows `{governed_by, audited_by, recognised_under, modifies, involves_counterparty, has_obligation, linked_to, evidenced_by}`
  - **Strict**: Only `{governed_by, audited_by, recognised_under, has_obligation, evidenced_by}`
- **Impact**: Reduces noise, focuses on financial/regulatory relations

#### `ENABLE_ALIAS_NORMALIZATION` (Default: True)
- **Type**: Boolean
- **Purpose**: Normalizes entity aliases to canonical forms
- **Examples**:
  - "asc 606" → "ASC 606"
  - "ifrs 15" → "IFRS 15"
  - "acme holdings limited" → "Acme Holdings Ltd."
- **Impact**: Reduces duplicate nodes, improves graph quality

#### `SKIP_EDGES_NO_EVID` (Default: True)
- **Type**: Boolean
- **Purpose**: Filters out edges without evidence text
- **Impact**: Improves graph quality but may reduce coverage

### 4. **Community Detection Parameters (Louvain Algorithm)**

#### `LOUVAIN_RESOLUTION` (Default: 1.25)
- **Type**: Float (0.2 - 2.0)
- **Purpose**: Controls granularity of community detection
- **Impact**:
  - **Low (0.2-0.8)**: Fewer, larger communities (broader themes)
  - **Medium (1.0-1.5)**: Balanced community sizes
  - **High (1.5-2.0)**: More, smaller communities (fine-grained themes)
- **Mathematical Basis**: Resolution parameter in modularity optimization
  - Modularity: `Q = (1/2m) * Σ[A_ij - (k_i*k_j)/(2m)] * δ(c_i, c_j)`
  - Resolution modifies the null model: `Q = (1/2m) * Σ[A_ij - γ*(k_i*k_j)/(2m)] * δ(c_i, c_j)`
  - Higher γ (resolution) → more communities

#### `LOUVAIN_RANDOM_STATE` (Default: 202)
- **Type**: Integer (1-999)
- **Purpose**: Random seed for reproducible community detection
- **Impact**: Ensures deterministic clustering results across runs
- **Note**: Louvain is a greedy algorithm with random tie-breaking

#### `use_ontology_for_clustering` (Default: False)
- **Type**: Boolean
- **Purpose**: Uses ontology-based edge weighting for clustering
- **Weight Boosts**:
  - `recognised_under`: 1.50x
  - `audited_by`: 1.40x
  - `governed_by`: 1.30x
  - `has_obligation`: 1.20x
  - `involves_counterparty`: 1.10x
- **Impact**: Financial/regulatory relations get higher weight in clustering

### 5. **Q&A & Prompt Tuning Parameters**

#### `MAX_EDGES_IN_PROMPT` (Default: 48)
- **Type**: Integer
- **Purpose**: Maximum edges included in LLM context for Q&A
- **Impact**: 
  - Too low: Missing relevant context
  - Too high: Token limit, noise, higher cost
- **Optimization**: Selected via relevance scoring (see `_edge_score_for_question`)

#### `MAX_EVIDENCE_CHARS` (Default: 220)
- **Type**: Integer
- **Purpose**: Maximum characters per evidence snippet in prompts
- **Impact**: Balances context richness vs. token usage

#### Edge Scoring Weights (for Q&A relevance)
- `EDGE_NAME_WEIGHT` (Default: 3): Weight for entity name matches
- `EDGE_REL_WEIGHT` (Default: 2): Weight for relation type matches
- `EDGE_EVID_WEIGHT` (Default: 1): Weight for evidence text matches
- **Scoring Formula**: 
  ```
  score = (name_hits × 3) + (rel_hits × 2) + (ev_hits × 1)
  ```

### 6. **Vector Search Parameters**

#### `ENABLE_VECTOR` (Default: False)
- **Type**: Boolean
- **Purpose**: Enables LlamaIndex-based vector search
- **Impact**: Adds semantic search capability alongside graph-based retrieval

#### `VECTOR_TOPK` (Default: 8)
- **Type**: Integer (2-16)
- **Purpose**: Number of top-k vector search results
- **Trade-off**: More results = better coverage but higher token usage

#### Vector Index Configuration
- **Embedding Model**: `text-embedding-3-small` (OpenAI)
- **Chunk Size**: 900 characters
- **Chunk Overlap**: 120 characters
- **LLM**: GPT-4o for query processing

---

## 🔄 Data Flow & Processing Pipeline

### Phase 1: Document Ingestion
```
Input: PDF/DOCX/TXT files
  ↓
File Type Detection
  ↓
Text Extraction (PyPDF2, python-docx, or direct read)
  ↓
Chunking (paragraph-based, max CHUNK_MAX_CHARS)
  ↓
Output: List of (source_name, text_chunk) pairs
```

### Phase 2: Knowledge Extraction
```
For each chunk:
  ↓
LLM Extraction (GPT-4o, temperature=EXTRACT_TEMP)
  ↓
Parse JSON triples: (head, relation, tail, evidence)
  ↓
Fallback: Rule-based extraction (if LLM fails)
  ↓
Fallback: Co-occurrence extraction (if enabled)
  ↓
Normalization:
  - Entity alias normalization
  - Relation normalization (lowercase, underscore)
  - Relation filtering (RELATION_POLICY)
  - Evidence requirement check
  ↓
Output: Cleaned triples list
```

### Phase 3: Graph Construction
```
Initialize: nx.DiGraph()
  ↓
For each triple (head, relation, tail, evidence):
  - Add edge: head → tail
  - Edge attributes: {label: relation, evidence_text: evidence, source: filename}
  ↓
Deduplication: Skip if edge already exists
  ↓
Output: Knowledge Graph G (NetworkX DiGraph)
```

### Phase 4: Community Detection
```
Convert to undirected graph (for clustering)
  ↓
Apply ontology weights (if enabled):
  - recognised_under: 1.50x
  - audited_by: 1.40x
  - etc.
  ↓
Louvain Algorithm:
  - Resolution: LOUVAIN_RESOLUTION
  - Random state: LOUVAIN_RANDOM_STATE
  - Weight: edge weights (ontology-boosted)
  ↓
Assign community IDs to nodes
  ↓
Output: Partition dictionary {node: community_id}
```

### Phase 5: Cluster Summarization
```
For each community:
  ↓
Extract subgraph (nodes in community)
  ↓
LLM Summarization:
  - Input: Entity list + context text
  - Output: Cluster summary + KPI/Risk label
  ↓
Heuristic Fallback (if LLM fails):
  - Top nodes by degree
  - Generic theme description
  ↓
Output: CLUSTER_SUMMARIES {cluster_id: {summary, label}}
```

### Phase 6: Graph Context Memory
```
For each cluster:
  - Cluster label & summary
  - Top entities (by degree)
  - Sample edges (max MAX_EDGES_PER_CLUSTER)
  ↓
Concatenate into structured text
  ↓
Output: GRAPH_CONTEXT_MEMORY (string)
```

### Phase 7: Query Processing
```
User Question
  ↓
Tokenize question
  ↓
Score all edges (relevance to question)
  ↓
Select top-k edges (MAX_EDGES_IN_PROMPT)
  ↓
Identify touched clusters
  ↓
Build LLM context:
  - Cluster summaries
  - Relevant edges with evidence
  - Vector search snippets (if enabled)
  ↓
LLM Q&A (GPT-4o, temperature=0.2)
  ↓
Return answer
```

---

## 🧮 Key Algorithms & Mathematical Foundations

### 1. **Louvain Community Detection**

**Algorithm**: Modularity optimization with greedy local search

**Modularity Formula**:
```
Q = (1/2m) * Σ_ij [A_ij - γ * (k_i * k_j)/(2m)] * δ(c_i, c_j)

Where:
- m = total edge weight
- A_ij = adjacency matrix (edge weight between i and j)
- k_i = degree of node i
- γ = resolution parameter (LOUVAIN_RESOLUTION)
- c_i = community of node i
- δ(c_i, c_j) = 1 if same community, else 0
```

**Optimization Process**:
1. Initialize: Each node in its own community
2. Iterate: Move nodes to maximize local modularity gain
3. Aggregate: Merge communities into super-nodes
4. Repeat until no improvement

**Time Complexity**: O(n log n) for sparse graphs

### 2. **Graph Metrics Computation**

#### **Density** (Directed Graph)
```
D = E / (V * (V - 1))

Where:
- E = number of edges
- V = number of nodes
```

#### **Degree Entropy**
```
H = -Σ p(d) * log₂(p(d))

Where:
- p(d) = probability of degree d
- Measures distribution uniformity
```

#### **Gini Coefficient** (Degree Inequality)
```
G = (2 * Σ(i * x_i)) / (n * Σ x_i) - (n + 1) / n

Where:
- x_i = sorted degree values
- n = number of nodes
- Range: [0, 1] (0 = perfect equality, 1 = maximum inequality)
```

#### **Assortativity** (Degree Correlation)
```
r = (Σ_ij (A_ij - k_i*k_j/2m) * k_i * k_j) / (Σ_ij (k_i * δ_ij - k_i*k_j/2m) * k_i * k_j)

Measures: Do high-degree nodes connect to high-degree nodes?
```

#### **Transitivity** (Global Clustering)
```
T = 3 * (number of triangles) / (number of connected triples)

Measures: Proportion of closed triangles
```

#### **Modularity** (Community Quality)
```
Q = (1/2m) * Σ_ij [A_ij - (k_i*k_j)/(2m)] * δ(c_i, c_j)

Measures: How well-defined are communities?
```

### 3. **Edge Relevance Scoring** (for Q&A)

```
score(u, v, d, q_tokens) = 
  (EDGE_NAME_WEIGHT × name_hits) +
  (EDGE_REL_WEIGHT × rel_hits) +
  (EDGE_EVID_WEIGHT × ev_hits) +
  domain_boosts

Where:
- name_hits = count of query tokens in entity names
- rel_hits = count of query tokens in relation label
- ev_hits = count of query tokens in evidence text
- domain_boosts = special boosts (e.g., "recogn" + "revenue" = +2)
```

---

## 📊 Performance Considerations & Optimizations

### 1. **LLM Call Optimization**
- **Chunking Strategy**: Paragraph-based chunking reduces redundant context
- **Timeout Handling**: 10s timeout per chunk prevents hanging
- **Fallback Chain**: LLM → Rule-based → Co-occurrence (graceful degradation)
- **Batch Processing**: Sequential chunk processing (can be parallelized)

### 2. **Graph Construction Efficiency**
- **Deduplication**: Skip existing edges (O(1) check with NetworkX)
- **Incremental Building**: Add edges one-by-one (no batch rebuild needed)
- **Memory**: NetworkX DiGraph is memory-efficient for sparse graphs

### 3. **Community Detection Scalability**
- **Louvain Algorithm**: O(n log n) for sparse graphs (efficient)
- **Undirected Conversion**: Only for clustering (preserves directed graph for queries)
- **Weighted Edges**: Ontology boosts don't significantly impact performance

### 4. **Query Performance**
- **Edge Scoring**: O(E) where E = edges (linear scan)
- **Top-k Selection**: O(E log k) with heap (efficient)
- **Context Building**: O(k) where k = selected edges (fast)

### 5. **Vector Search Integration**
- **Index Building**: One-time cost (O(n) where n = chunks)
- **Query Time**: O(log n) with approximate nearest neighbor search
- **Hybrid Fusion**: Combines graph + vector results (additive cost)

---

## 🎯 Interview Talking Points

### **1. Why GraphRAG over Traditional RAG?**
- **Structured Reasoning**: Graph enables multi-hop reasoning (e.g., "What contracts are governed by standards that affect revenue recognition?")
- **Relationship Awareness**: Captures explicit relationships (audited_by, governed_by) that vectors miss
- **Community Insights**: Clustering reveals thematic patterns (risk clusters, vendor dependencies)
- **Provenance**: Every edge has evidence text (explainable AI)

### **2. Technical Challenges Solved**
- **Noise Reduction**: Multi-layer filtering (relation policy, evidence requirement, alias normalization)
- **Scalability**: Efficient algorithms (Louvain O(n log n), edge scoring O(E))
- **Robustness**: Fallback chain (LLM → rules → co-occurrence) ensures extraction always works
- **Reproducibility**: Random state seeds for deterministic clustering

### **3. Graph Tuning Philosophy**
- **Precision vs. Recall Trade-off**: Strict parameters (STRICT_PROMPT, REQUIRE_EVIDENCE) favor precision
- **Domain-Specific Optimization**: Financial relations get higher weights in clustering
- **Adaptive Extraction**: Temperature and chunk size tuned for financial document characteristics
- **User Control**: All parameters exposed in UI for domain experts to tune

### **4. Hybrid Search Architecture**
- **Complementary Strengths**: Graph for structured queries, vectors for semantic similarity
- **Fusion Strategy**: Graph results prioritized (structured), vector snippets supplement (unstructured)
- **Context Building**: Smart edge selection (relevance scoring) + vector top-k fusion

### **5. CFO Analytics Integration**
- **30 Strategic Questions**: Pre-defined across 6 dimensions (Financial Exposure, Revenue, Risk, Operations, Cash Flow, Strategy)
- **KPI Extraction**: Structured JSONL output with KPIs, risks, opportunities, evidence
- **Contract Data Fusion**: Merges graph knowledge with structured contract data (CSV)
- **Dashboard Integration**: JSONL format enables external Streamlit dashboard consumption

### **6. Production Readiness Features**
- **Error Handling**: Try-catch blocks, timeout handling, graceful degradation
- **Export Formats**: CSV (tables), JSONL (analytics), RDF (semantic web), Turtle (ontology)
- **Incremental Processing**: Contract data persists across sessions
- **Monitoring**: Comprehensive logging at each pipeline stage

---

## 🔬 Advanced Technical Details

### **RDF/Ontology Export**
- **OWL Classes**: Entity, Standard, CFOInsight, KPI, Risk, Opportunity
- **Object Properties**: auditedBy, governedBy, recognisedUnder, hasObligation, etc.
- **Namespaces**: Custom ontology namespace + DCTERMS, SKOS, RDFS, OWL
- **Serialization**: Turtle (TTL) and JSON-LD formats

### **Contract Extraction Integration**
- **Multi-stage Pipeline**: Document upload → Contract extraction → Graph building → Analytics
- **Data Merging**: Dummy contracts (foundation) + uploaded contracts + new contracts
- **Context Enrichment**: Contract portfolio statistics fed into graph context

### **Graph Context Memory Structure**
```
GRAPH CONTEXT MEMORY
[Cluster 0] Label: Revenue Recognition Risk
[Cluster 0] Summary: Entities related to IFRS-15 compliance...
[Cluster 0] Key entities: ASC 606, IFRS 15, Revenue Contract
[Cluster 0] Rel: ASC 606 —[governed_by]→ Revenue Contract; Ev: "Revenue is recognized under ASC 606..."
...
```

### **CFO Analytics Schema** (JSONL)
```json
{
  "dimension": "Financial Exposure & Obligations",
  "question": "What is the total value of active contracts?",
  "insight": "Total contract value is $X with Y contracts...",
  "kpis": {"total_value": 1000000, "contract_count": 50},
  "risks": ["Concentration risk with Vendor A"],
  "opportunities": ["Renegotiate high-value contracts"],
  "evidence": [{"snippet": "...", "source": "contract.pdf"}],
  "data_gaps": [],
  "confidence": 0.85,
  "_meta": {"generated_at": "...", "model": "gpt-4o"}
}
```

---

## 📈 Metrics & Evaluation

### **Graph Quality Metrics**
- **Node Count**: Number of unique entities
- **Edge Count**: Number of relations
- **Density**: Graph connectivity (higher = more connected)
- **Modularity**: Community quality (higher = better-defined clusters)
- **Provenance Completeness**: % of edges with evidence (quality indicator)

### **Extraction Quality Indicators**
- **Triples Extracted**: Total relations found
- **Unique Relation Types**: Diversity of relations
- **Evidence Coverage**: % of edges with evidence text
- **Cluster Count**: Number of communities (thematic groups)

---

## 🚀 Future Enhancements (Discussion Points)

1. **Parallel Processing**: Multi-threaded chunk processing for faster extraction
2. **Graph Embeddings**: Node2Vec or GraphSAGE for similarity search
3. **Incremental Graph Updates**: Add new documents without full rebuild
4. **Confidence Scoring**: ML-based confidence for extracted relations
5. **Multi-modal Support**: Extract from images, tables, diagrams
6. **Real-time Updates**: WebSocket-based live graph updates
7. **Advanced Analytics**: Temporal analysis, anomaly detection, trend prediction

---

## 📝 Summary for Interview

**"This is a production-grade GraphRAG system that transforms unstructured financial contracts into a queryable knowledge graph. The system uses a hybrid architecture combining graph-based reasoning with vector semantic search, enabling CFO-level strategic insights. Key technical innovations include:**

1. **Multi-layer extraction pipeline** with LLM, rule-based, and co-occurrence fallbacks
2. **Tunable graph parameters** (15+ parameters) for domain-specific optimization
3. **Louvain community detection** with ontology-weighted clustering
4. **Hybrid search** fusing graph traversal with vector similarity
5. **CFO analytics** generating 30 strategic questions with structured JSONL output
6. **RDF/OWL export** for semantic web interoperability

**The system is designed for financial domain experts, with all parameters exposed in the UI for fine-tuning. It handles errors gracefully, supports incremental processing, and exports to multiple formats for downstream analytics."**

---

*This document provides comprehensive technical coverage for interview discussions. Focus on the sections most relevant to your interviewer's background (ML, graph algorithms, software engineering, etc.).*


