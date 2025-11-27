"""
Enhanced Utilities for CFO Contract Analytics
File processing, text manipulation, and mathematical helpers for contract analysis
"""

import os
import re
import json
import math
import itertools
import networkx as nx
from typing import List, Tuple
from collections import Counter, defaultdict
from urllib.parse import quote

import config

# ==================== Optional Parser Imports ====================
try:
    from PyPDF2 import PdfReader
    config.PDF_AVAILABLE = True
except Exception:
    config.PDF_AVAILABLE = False

try:
    import docx
    config.DOCX_AVAILABLE = True
except Exception:
    config.DOCX_AVAILABLE = False

try:
    from rdflib import Graph as RDFGraph, Namespace, URIRef, BNode, Literal
    from rdflib.namespace import RDF, RDFS, OWL, SKOS, DCTERMS
    config.RDFLIB_AVAILABLE = True
except Exception:
    config.RDFLIB_AVAILABLE = False

try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    config.LLAMA_AVAILABLE = True
except Exception:
    config.LLAMA_AVAILABLE = False

# ==================== File Processing Utilities ====================

def normalize_types(selected: List[str]) -> List[str]:
    """Convert file type selections to extensions"""
    exts = []
    for t in selected or []:
        exts += config.SUPPORTED_TYPES.get(t, [])
    return list(dict.fromkeys(exts))

def read_pdf(path: str) -> str:
    """Read text content from PDF file"""
    if not config.PDF_AVAILABLE:
        print(f"[PDF] PyPDF2 not available; skipping: {path}")
        return ""
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n".join(parts)
    except Exception as e:
        print(f"[PDF] Failed to parse {path}: {e}")
        return ""

def read_doсx(path: str) -> str:
    """Read text content from DOCX file"""
    if not config.DOCX_AVAILABLE:
        print(f"[DOCX] python-docx not available; skipping: {path}")
        return ""
    try:
        d = docx.Document(path)
        paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras)
    except Exception as e:
        print(f"[DOCX] Failed to parse {path}: {e}")
        return ""

def read_txt(path: str) -> str:
    """Read text content from TXT file"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[TXT] Failed to read {path}: {e}")
        return ""

def read_one_file(path: str) -> str:
    """Read one file based on its extension"""
    p = path.lower()
    if p.endswith(".pdf"):
        return read_pdf(path)
    if p.endswith(".docx"):
        return read_doсx(path)
    if p.endswith(".txt"):
        return read_txt(path)
    return ""

def list_dir_files(root: str, allowed_exts: List[str]) -> List[str]:
    """Recursively find all files in directory with allowed extensions"""
    out = []
    if not root or not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if allowed_exts and ext not in allowed_exts:
                continue
            out.append(os.path.join(dirpath, fn))
    return out

# ==================== Mathematical Utilities ====================

def entropy(values):
    """Calculate entropy of values"""
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    ps = [v / total for v in values if v > 0]
    return -sum(p * math.log(p, 2) for p in ps)

def gini(seq):
    """Calculate Gini coefficient"""
    xs = sorted(max(0.0, float(x)) for x in seq)
    n = len(xs)
    if n == 0:
        return 0.0
    s = sum(xs)
    if s == 0:
        return 0.0
    num = 0.0
    for i, x in enumerate(xs, start=1):
        num += (2*i - n - 1) * x
    return num / (n * s)

def giant_component_undirected(Gu):
    """Get the largest connected component of an undirected graph"""
    if Gu.number_of_nodes() == 0:
        return Gu.copy()
    comps = list(nx.connected_components(Gu))
    if not comps:
        return Gu.copy()
    largest = max(comps, key=len)
    return Gu.subgraph(largest).copy()

# ==================== Text Processing Utilities ====================

def chunk_text(text, max_chars=None):
    """Split text into chunks based on paragraph boundaries"""
    max_chars = max_chars or config.CHUNК_MAX_CHARS
    buf, out = [], []
    for para in re.split(r"\n{2,}", text):
        if sum(len(x) for x in buf) + len(para) + 1 > max_chars:
            if buf:
                out.append("\n\n".join(buf))
                buf = []
        if para.strip():
            buf.append(para.strip())
    if buf:
        out.append("\n\n".join(buf))
    return out

def parse_json_or_lines(raw):
    """Parse JSON or line-based triple format"""
    triples = []
    raw = (raw or "").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "triples" in data:
            data = data["triples"]
        if isinstance(data, list):
            for t in data:
                h = (t.get("head") or t.get("subject") or "").strip()
                r = (t.get("relation") or t.get("predicate") or "").strip()
                ta = (t.get("tail") or t.get("object") or "").strip()
                ev = (t.get("evidence") or t.get("source") or "").strip()
                if h and r and ta:
                    triples.append((h, r, ta, ev))
            return triples
    except Exception:
        pass
    
    for line in raw.splitlines():
        if "--[" in line and "]-->" in line:
            try:
                lhs, rest = line.split("--[", 1)
                rel, rhs = rest.split("]-->", 1)
                h = lhs.strip()
                r = rel.strip()
                ta = rhs.strip()
                if h and r and ta:
                    triples.append((h, r, ta, ""))  # no evidence available
            except Exception:
                continue
    return triples

def fallback_cooccurrence(sent):
    """Generate fallback co-occurrence triples"""
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9&\-/\.]{2,}\b", sent)
    tokens = [t for t in tokens if t not in {"The", "And", "For", "With", "Note", "Page"}]
    tokens = list(dict.fromkeys(tokens))
    pairs = list(itertools.combinations(tokens, 2))
    if not config.ENABLE_FALLBACK_COOCC:
        return []
    if config.MAX_COOCC_PAIRS > 0:
        pairs = pairs[:config.MAX_COOCC_PAIRS]
    return [(a, "CO_OCCURS_IN_SENTENCE", b, sent.strip()) for a, b in pairs]

# ==================== Relation Processing Utilities ====================

def norm_rel(rel):
    """Normalize relation names"""
    rel = (rel or "").strip().lower().replace(" ", "_")
    return config.REL_MAP.get(rel, rel)

def relation_allowed(rel_norm):
    """Check if relation is allowed based on policy"""
    if config.RELATION_POLICY == "Off":
        return True
    if config.RELATION_POLICY == "Standard":
        return rel_norm in config.REL_ALLOW_STANDARD
    if config.RELATION_POLICY == "Strict":
        return rel_norm in config.REL_ALLOW_STRICT
    return True

def canon_entity(name):
    """Canonicalize entity names using alias mapping"""
    if not config.ENABLE_ALIAS_NORMALIZATION:
        return name.strip()
    return config.ALIASES.get(name.strip().lower(), name.strip())

# ==================== RDF/IRI Utilities ====================

def slug(name: str) -> str:
    """Create URL-safe slug from name"""
    return quote(re.sub(r"[^A-Za-z0-9_\-\.]+", "_", name.strip()), safe="_-.")

def node_iri(node: str, base_iri: str) -> str:
    """Create IRI for a node"""
    return base_iri.rstrip("/") + "/entity/" + slug(node)

def json_safe_load(raw: str):
    """Safely load JSON with cleanup"""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(raw)
    except Exception:
        return None

# ==================== Corpum Loading ====================

def load_corpus(upload_paths: List[str], dir_path: str, type_choices: List[str]) -> List[Tuple[str, str]]:
    """
    Load corpus from uploads and/or directory
    Returns list of (source_name, text)
    """
    allowed_exts = normalize_types(type_choices)
    pairs = []

    # Process uploads
    for p in (upload_paths or []):
        if not p:
            continue
        ext = os.path.splitext(str(p))[1].lower()
        if allowed_exts and ext not in allowed_exts:
            continue
        txt = read_one_file(str(p))
        if txt.strip():
            pairs.append((os.path.basename(str(p)), txt))

    # Process directory
    if dir_path and os.path.isdir(dir_path):
        for p in list_dir_files(dir_path, allowed_exts):
            txt = read_one_file(p)
            if txt.strip():
                pairs.append((os.path.relpath(p, dir_path), txt))

    return pairs

# ==================== Rule-Based Pattern Matching ====================

REL_PATTERNS = [
    (re.compile(r"(.+?)\s+(?:is|was| are|were)?\s*audited by\s+(.+?)\.", re.I), "audited_by"),
    (re.compile(r"(.+?)\s+(?:is|was|are|were)?\s*governed by\s+(.+?)\.", re.I), "governed_by"),
    (re.compile(r"(.+?)\s+(?:is|was| are|were)?\s*recognis(?:e|z)d? under\s+(.+?)\.", re.I), "recognised_under"),
    (re.compile(r"(.+?)\s+has obligation(?:s)? to\s+(.+?)\.", re.I), "has_obligation"),
    (re.compile(r"(.+?)\s+involves\s+(.+?)\.", re.I), "involves_counterparty"),
]

def rule_based_triples(text):
    """Extract triples using rule-based patterns"""
    triples = []
    for sent in re.split(r'(?<=[.!?])\s+', text):
        s = sent.strip()
        if not s:
            continue
        for pat, rel in REL_PATTERNS:
            m = pat.search(s)
            if m:
                a = m.group(1).strip().strip('",() ')
                b = m.group(2).strip().strip('",() ')
                if a and b and a.lower() != b.lower():
                    triples.append((a, rel, b, s))
        
        kw_hits = [
            ("recognised_under", ["ifrs", "asc", "gaap", "standard", "policy"]),
            ("audited_by", ["auditor", "audit", "audited"]),
            ("governed_by", ["governed", "regulation", "law", "act"]),
        ]
        low = s.lower()
        for rel, kws in kw_hits:
            if any(k in low for k in kws):
                ents = re.findall(r"\b[A-Z][A-Za-z0-9&\-/\.]{2,}(?:\s+[A-Z][A-Za-z0-9&\-/\.]{2,})*\b", s)
                if len(ents) >= 2:
                    triples.append((ents[0], rel, ents[1], s))
    
    out, seen = [], set()
    for a, r, b, ev in triples:
        key = (a, r, b, ev)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out





