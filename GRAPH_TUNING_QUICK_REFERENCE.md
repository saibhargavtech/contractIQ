# 🎛️ Graph Tuning Parameters - Quick Reference

## 📋 Parameter Cheat Sheet

### **Extraction Control**

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `EXTRACT_TEMP` | 0.4 | 0.0-1.0 | **Lower** = more deterministic, fewer hallucinations<br>**Higher** = more creative, risk of noise |
| `STRICT_PROMPT` | True | Boolean | **True** = rejects weak edges, higher precision<br>**False** = more permissive, higher recall |
| `REQUIRE_EVIDENCE` | True | Boolean | **True** = only edges with source sentences<br>**False** = allows edges without provenance |
| `CHUNK_MAX_CHARS` | 3500 | 2000-5000 | **Smaller** = more API calls, better focus<br>**Larger** = fewer calls, risk of truncation |

### **Fallback & Noise Control**

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `ENABLE_FALLBACK_COOCC` | False | Boolean | **True** = enables co-occurrence backup extraction |
| `MAX_COOCC_PAIRS` | 5 | 0-10 | Controls graph density from fallback extraction |
| `SKIP_EDGES_NO_EVID` | True | Boolean | **True** = filters edges without evidence (quality) |

### **Relation Filtering**

| Parameter | Default | Options | Impact |
|-----------|---------|---------|--------|
| `RELATION_POLICY` | "Off" | Off/Standard/Strict | **Off** = all relations<br>**Standard** = 8 financial relations<br>**Strict** = 5 core relations |
| `ENABLE_ALIAS_NORMALIZATION` | True | Boolean | Normalizes entity names (e.g., "asc 606" → "ASC 606") |

### **Community Detection (Louvain)**

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `LOUVAIN_RESOLUTION` | 1.25 | 0.2-2.0 | **Lower** = fewer, larger communities<br>**Higher** = more, smaller communities |
| `LOUVAIN_RANDOM_STATE` | 202 | 1-999 | Ensures reproducible clustering |
| `use_ontology_for_clustering` | False | Boolean | **True** = financial relations get 1.1-1.5x weight boost |

### **Q&A Tuning**

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `MAX_EDGES_IN_PROMPT` | 48 | Integer | Maximum edges in LLM context (token limit) |
| `MAX_EVIDENCE_CHARS` | 220 | Integer | Max characters per evidence snippet |
| `EDGE_NAME_WEIGHT` | 3 | Integer | Relevance weight for entity name matches |
| `EDGE_REL_WEIGHT` | 2 | Integer | Relevance weight for relation matches |
| `EDGE_EVID_WEIGHT` | 1 | Integer | Relevance weight for evidence matches |

### **Vector Search**

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `ENABLE_VECTOR` | False | Boolean | Enables LlamaIndex semantic search |
| `VECTOR_TOPK` | 8 | 2-16 | Number of vector search results |

---

## 🎯 Recommended Settings by Use Case

### **High Precision (Financial Auditing)**
```
EXTRACT_TEMP = 0.2
STRICT_PROMPT = True
REQUIRE_EVIDENCE = True
RELATION_POLICY = "Strict"
SKIP_EDGES_NO_EVID = True
ENABLE_FALLBACK_COOCC = False
```

### **High Recall (Exploratory Analysis)**
```
EXTRACT_TEMP = 0.6
STRICT_PROMPT = False
REQUIRE_EVIDENCE = False
RELATION_POLICY = "Off"
SKIP_EDGES_NO_EVID = False
ENABLE_FALLBACK_COOCC = True
MAX_COOCC_PAIRS = 10
```

### **Balanced (General Use)**
```
EXTRACT_TEMP = 0.4
STRICT_PROMPT = True
REQUIRE_EVIDENCE = True
RELATION_POLICY = "Standard"
LOUVAIN_RESOLUTION = 1.25
```

### **Fine-Grained Clustering**
```
LOUVAIN_RESOLUTION = 1.8
use_ontology_for_clustering = True
```

### **Broad Themes**
```
LOUVAIN_RESOLUTION = 0.8
use_ontology_for_clustering = False
```

---

## 🔢 Mathematical Relationships

### **Louvain Resolution Impact**
- **Resolution = 0.5**: ~2-3 large communities
- **Resolution = 1.0**: ~5-8 medium communities
- **Resolution = 1.5**: ~10-15 small communities
- **Resolution = 2.0**: ~20+ very small communities

### **Chunk Size vs. Cost**
- **Chunk = 2000**: ~2x API calls, lower cost per call
- **Chunk = 3500**: Balanced (default)
- **Chunk = 5000**: ~0.7x API calls, higher cost per call

### **Edge Scoring Formula**
```
relevance_score = 
  (name_matches × 3) + 
  (relation_matches × 2) + 
  (evidence_matches × 1) + 
  domain_boosts
```

---

## 💡 Pro Tips

1. **Start with defaults** and tune based on output quality
2. **Monitor extraction logs** to see which fallback is used
3. **Check graph metrics** (modularity, density) after tuning
4. **Use ontology weighting** for financial documents
5. **Lower temperature** for regulatory/compliance documents
6. **Higher resolution** for detailed risk analysis
7. **Enable vector search** for semantic queries (not just entity-based)

---

## 🐛 Troubleshooting

| Problem | Likely Fix |
|---------|-----------|
| Too few edges extracted | Lower `STRICT_PROMPT`, enable `ENABLE_FALLBACK_COOCC` |
| Too many noisy edges | Increase `STRICT_PROMPT`, set `RELATION_POLICY="Strict"` |
| All nodes in one cluster | Lower `LOUVAIN_RESOLUTION` |
| Too many tiny clusters | Raise `LOUVAIN_RESOLUTION` |
| LLM timeouts | Reduce `CHUNK_MAX_CHARS`, check API key |
| Poor Q&A answers | Increase `MAX_EDGES_IN_PROMPT`, enable `ENABLE_VECTOR` |

---

*Quick reference for interview discussions. See `TECHNICAL_EXPLANATION.md` for detailed explanations.*


