"""
Enhanced Vector Search Module for CFO Analytics
LlamaIndex integration for semantic search over contract documents
"""

from typing import List, Tuple, Dict
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

import config

# ==================== Vector Index Building ====================

def build_vector_index_from_docs(doc_pairs: List[Tuple[str, str]]):
    """
    Build multi-document vector index with CFO-optimized chunking
    """
    config.VECTOR_INDEX = None
    
    if not config.LLAMA_AVAILABLE or not doc_pairs:
        return None

    try:
        # Configure LlamaIndex components
        llm = LlamaOpenAI(model="gpt-4", temperature=0.0)
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.llm = llm
        Settings.embed_model = embed_model
    except Exception as e:
        print(f"[Vector] Failed to init OpenAI providers: {e}")
        return None

    # CFO-optimized chunking for financial contract analysis
    splitter = SentenceSplitter(
        chunk_size=900,  # Smaller chunks for precision
        chunk_overlap=120,  # Adequate overlap for financial context
        paragraph_separator="\n\n"  # Contract paragraphs are critical
    )
    
    documents = []
    for src, txt in doc_pairs:
        if not txt.strip():
            continue
            
        # Create document with source metadata
        doc = Document(text=txt, doc_id=src, metadata={"source": src})
        
        # Split into nodes optimized for financial analysis
        nodes = splitter.get_nodes_from_documents([doc])
        
        for node in nodes:
            node_text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
            if node_text and node_text.strip():
                # Enhanced metadata for CFO context
                enhanced_metadata = {
                    "source": src,
                    "document_type": "contract",
                    "financial_relevance": _assess_financial_relevance(node_text)
                }
                documents.append(Document(text=node_text, metadata=enhanced_metadata))

    if not documents:
        return None

    try:
        # Build index with CFO-optimized settings
        config.VECTOR_INDEX = VectorStoreIndex.from_documents(documents)
        print(f"[Vector] Built index with {len(documents)} document chunks")
        return config.VECTOR_INDEX
    except Exception as e:
        print(f"[Vector] Index build failed: {e}")
        config.VECTOR_INDEX = None
        return None

def _assess_financial_relevance(text: str) -> str:
    """Assess financial relevance of text chunk"""
    text_lower = text.lower()
    
    high_relevance_keywords = [
        'payment', 'revenue', 'cost', 'price', 'fee', 'amount', 'dollar', 'financial', 
        'budget', 'profit', 'loss', 'expeреnditure', 'obligation', 'liability',
        'contract value', 'total cost', 'annual fee'
    ]
    
    medium_relevance_keywords = [
        'terms', 'duration', 'period', 'schedule', 'timeline', 'milestone', 
        'deliverable', 'performance', 'requirement'
    ]
    
    high_count = sum(1 for keyword in high_relevance_keywords if keyword in text_lower)
    medium_count = sum(1 for keyword in medium_relevance_keywords if keyword in text_lower)
    
    if high_count >= 2:
        return "high"
    elif medium_count >= 2 or high_count >= 1:
        return "medium"
    else:
        return "low"

# ==================== Enhanced Semantic Search ====================

def vector_snippets(query: str, top_k: int = None) -> List[str]:
    """Enhanced semantic search with CFO context awareness"""
    if not config.LLAMA_AVAILABLE or config.VECTOR_INDEX is None or not query:
        return []
    
    top_k = top_k or config.VECTOR_TOPK
    
    try:
        retriever = config.VECTOR_INDEX.as_retriever(similarity_top_k=int(top_k))
        results = retriever.retrieve(query)
        
        snippets = []
        for r in results:
            try:
                txt = node.get_content()
            except Exception:
                txt = getattr(r, "text", "")
            
            meta = r.node.metadata if hasattr(r.node, "metadata") else {}
            src = meta.get("source", "uploaded")
            relevance = meta.get("financial_relevance", "unknown")
            
            if txt:
                # Enhanced snippet formatting for CFO context
                cleaned = _prepare_cfo_snippet(txt, query)
                if cleaned:
                    relevance_indicator = "💰" if relevance == "high" else "📊" if relevance == "medium" else "📋"
                    snippet_text = f"{relevance_indicator} [{src}] {cleaned}"
                    snippets.append(snippet_text)
        
        return snippets
    except Exception as e:
        print(f"[Vector] Retrieval failed: {e}")
        return []

def _prepare_cfo_snippet(text: str, query: str) -> str:
    """Prepare text snippet optimized for CFO analysis"""
    # Clean and highlight relevant sections
    cleaned = re.sub(r"\s+", " ", text).strip()
    
    # Highlight query terms
    query_terms = query.lower().split()
    for term in query_terms:
        if len(term) > 3:  # Only highlight meaningful terms
            cleaned = re.sub(f"\\b({term})\\b", r"**\1**", cleaned, flags=re.IGNORECASE)
    
    # Truncate intelligently at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    if len(sentences) == 1:
        # Single sentence - truncate at reasonable point
        if len(cleaned) > 400:
            cleaned = cleaned[:397] + "..."
    else:
        # Multiple sentences - try to find relevant ones
        relevant_sentences = []
        current_length = 0
        query_lower = query.lower()
        
        for sentence in sentences:
            if current_length + len(sentence) > 350:
                break
            # Prioritize sentences with query terms
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_lower.split()):
                relevant_sentences.append(sentence)
                current_length += len(sentence)
        
        if relevant_sentences:
            cleaned = " ".join(relevant_sentences)
        else:
            cleaned = " ".join(sentences[:3])  # Fallback to first 3 sentences
    
    return cleaned

# ==================== CFO-Specific Vector Queries ====================

def search_financial_clauses(query: str, top_k: int = 5) -> List[Dict]:
    """Search for specific financial clauses and terms"""
    if not config.VECTOR_INDEX:
        return []
    
    # Enhance query for financial focus
    enhanced_query = f"{query} financial payment revenue cost obligation"
    
    try:
        retriever = config.VECTOR_INDEX.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(enhanced_query)
        
        financial_results = []
        for result in results:
            try:
                text = result.node.get_content()
                metadata = result.node.metadata or {}
                
                financial_score = _calculate_financial_score(text, query)
                
                if financial_score > 0.3:  # Filter for relevant results
                    financial_results.append({
                        "text": text[:300] + "..." if len(text) > 300 else text,
                        "source": metadata.get("source", "unknown"),
                        "relevance": metadata.get("financial_relevance", "unknown"),
                        "score": financial_score,
                        "query_match": query
                    })
            except Exception:
                continue
        
        # Sort by financial relevance score
        return sorted(financial_results, key=lambda x: x["score"], reverse=True)
    
    except Exception as e:
        print(f"[Vector] Financial search failed: {e}")
        return []

def _calculate_financial_score(text: str, query: str) -> float:
    """Calculate financial relevance score for text"""
    text_lower = text.lower()
    query_lower = query.lower()
    
    # Base relevance (query term matches)
    query_terms = query_lower.split()
    base_score = sum(0.1 for term in query_terms if term in text_lower)
    
    # Financial term bonus
    financial_terms = [
        'payment', 'revenue', 'cost', 'price', 'fee', 'amount', 'financial', 
        'obligation', 'liability', 'dollar', 'contract value', 'annual', 'quarterly'
    ]
    financial_bonus = sum(0.05 for term in financial_terms if term in text_lower)
    
    # Currency/number bonus (likely financial context)
    currency_bonus = 0.1 if re.search(r'\$[\d,]+|[\d,]+\s*(?:USD|dollars?)', text_lower) else 0.0
    
    return base_score + financial_bonus + currency_bonus

def search_compliance_requirements(query: str, standard: str = None) -> List[Dict]:
    """Search for compliance-related content"""
    if not config.VECTOR_INDEX:
        return []
    
    # Build compliance-focused query
    compliance_query = f"{query} compliance requirement regulation policy"
    if standard:
        compliance_query += f" {standard}"
    
    try:
        retriever = config.VECTOR_INDEX.as_retriever(similarity_top_k=8)
        results = retriever.retrieve(compliance_query)
        
        compliance_results = []
        for result in results:
            try:
                text = result.node.get_content()
                metadata = result.node.metadata or {}
                
                # Check if text contains compliance-related terms
                compliance_score = _calculate_compliance_score(text, standard)
                
                if compliance_score > 0.4:
                    compliance_results.append({
                        "text": text[:400] + "..." if len(text) > 400 else text,
                        "source": metadata.get("source", "unknown"),
                        "compliance_score": compliance_score,
                        "standard": standard or "general",
                        "context": query
                    })
            except Exception:
                continue
        
        return sorted(compliance_results, key=lambda x: x["compliance_score"], reverse=True)
    
    except Exception as e:
        print(f"[Vector] Compliance search failed: {e}")
        return []

def _calculate_compliance_score(text: str, standard: str = None) -> float:
    """Calculate compliance relevance score"""
    text_lower = text.lower()
    
    # Base compliance terms
    compliance_terms = [
        'compliance', 'requirement', 'regulation', 'policy', 'standard', 'framework',
        'audit', 'certification', 'governance', 'control', 'oversight'
    ]
    
    base_score = sum(0.1 for term in compliance_terms if term in text_lower)
    
    # Standard-specific bonus
    if standard:
        standard_score = 0.2 if standard.lower() in text_lower else 0.0
        base_score += standard_score
    
    # Key compliance frameworks
    frameworks = ['gdpr', 'sox', 'ifrs', 'gaap', 'iso', 'pci', 'soc2', 'hipaa']
    framework_bonus = sum(0.15 for framework in frameworks if framework in text_lower)
    
    return base_score + framework_bonus

# ==================== Vector Search Integration ====================

def integrate_vector_search_with_graph(G, query: str) -> str:
    """Integrate vector search results with graph context"""
    vector_results = vector_snippets(query, top_k=6)
    
    if not vector_results:
        return ""
    
    # Format vector results for graph context
    vector_context = "Additional Document Context (Vector Search):\n"
    for i, result in enumerate(vector_results, 1):
        vector_context += f"- {result}\n"
    
    return vector_context
