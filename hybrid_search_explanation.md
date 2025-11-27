# 🕸️ Hybrid Search: GraphRAG + Vector Search Integration

## 📊 How The Hybrid System Works

When you search in the Gradio interface, **both GraphRAG and Vector Search work together** to provide comprehensive answers:

### 🔄 Step 1: Document Processing
```
Uploaded Documents
        ↓
┌─────────────────┐    ┌─────────────────┐
│   GraphRAG      │    │   Vector        │
│   Processing    │    │   Processing    │
└─────────────────┘    └─────────────────┘
        ↓                        ↓
┌──────────────┐         ┌──────────────┐
│ Knowledge    │         │ Text Chunks  │
│ Graph        │         │ Embeddings   │
│ • Entities   │         │ • Semantic   │
│ • Relations  │         │   Search     │
│ • Clusters   │         │ • Similarity │
└──────────────┘         └──────────────┘
```

### 🔍 Step 2: During Search

When you ask: "What are the payment terms for our IBM contracts?"

**GraphRAG Path:**
1. 🔍 **Entity Recognition**: Finds "IBM" in graph nodes
2. 🔗 **Relation Traversal**: Follows edges connected to IBM
3. 📊 **Cluster Context**: Loads cluster summaries for IBM-related communities
4. 📝 **Gragh-based Answer**: Uses relationship evidence

**Vector Search Path:**
1. 📝 **Semantic Understanding**: "payment terms" → embeddings
2. 🔍 **Similarity Search**: Finds chunks about "payment", "IBM", "terms"
3. 💰 **Financial Relevance**: Prioritizes high-relevance financial chunks
4. 📋 **Content-based Answer**: Uses document snippets

### 🎯 Step 3: Hybrid Fusion

```python
def hybrid_search(query, graph_context, vector_snippets):
    context_parts = []
    
    # 1. GraphRAG Context (Structured Relations)
    if graph_context:
        context_parts.append("Cluster context:\n" + graph_context)
    
    # 2. Vector Search Snippets (Semantic Similarity)  
    if vector_snippets:
        context_parts.append("Additional snippets:\n" + snippets)
    
    # 3. Combined Evidence for GPT-4O
    combined_context = "\n\n".join(context_parts)
    
    return gpt4o_answer(query, combined_context)
```

## 💡 Example: "What are our compliance requirements?"

### GraphRAG Contribution:
```
📊 Cluster Context:
Corporate Compliance Framework
• Entities: GDPR, SOC2, HIPAA, Company X
• Relations: Company X → governed_by → GDPR
• Evidence: "compliance with data protection regulations"

🔗 Relevant Relations:
[Cluster 3] Rel: IBM → [compliance] → GDPR
[Cluster 3] Rel: Contract C-1000 → [subject_to] → SOC2
```

### Vector Search Contribution:
```
💰 [uploaded] **compliance** requirements include SOC2, PCI DSS...
📊 [demo] All contracts shall **comply** with applicable data protection laws...
📋 [uploaded] **Governance** framework includes quarterly compliance audits...
```

### 🎯 Combined Answer:
"The compliance framework includes:
• **SOC2** coverage for IBM contracts (via GraphRAG relations)
• **Data Protection** requirements from document snippets (via Vector)
• **Quarterly audits** governance structure (via Vector)"
```

## 🚀 Why Both Together Are Powerful

**GraphRAG Strengths:**
- ✅ **Relationship clarity**: Entity A → relates_to → Entity B
- ✅ **Evidence-based**: "Company X governs IBM contract"
- ✅ **Structured reasoning**: Graph paths for complex queries
- ✅ **Multi-hop**: Follow chains of relationships

**Vector Search Strengths:**
- ✅ **Semantic understanding**: "payment terms" finds "billing schedule"
- ✅ **Full-text search**: Catches things not in graph relations
- ✅ **Financial relevance**: Prioritizes money/payment content
- ✅ **Flexible matching**: Finds related concepts GPT understands

## 🔧 Configuration

**Enable Hybrid Search:**
```
enable_vector=True    # Activate vector search
vector_topk=8         # Number of vector snippets to retrieve
```

**Processing Flow:**
```
Upload Documents → GraphRAG Build → Vector Index Build → Ready for Hybrid Search
```

## 🎯 Result: Best of Both Worlds

When you search, you get:
1. **Structured relations** from GraphRAG (who/what/how connections)
2. **Semantic context** from Vector Search (full-text similarity)  
3. **Combined intelligence** fed to GPT-4O for comprehensive answers

This creates **richer, more contextual responses** than either approach alone!
