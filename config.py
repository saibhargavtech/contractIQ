"""
Enhanced Configuration for CFO Contract Analytics Studio
Centralized settings for the comprehensive contract analysis system
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============ API Configuration ============
# Priority: Streamlit secrets > environment variable > .env file
OPENAI_API_KEY = None

# Try Streamlit secrets first (for Streamlit Cloud deployment)
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
except (AttributeError, KeyError, FileNotFoundError, ImportError):
    # Streamlit secrets not available, try environment variable
    pass

# Fallback to environment variable
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fallback to .env file (for local development)
if not OPENAI_API_KEY:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
USE_AZURE_OPENAI = bool(AZURE_OPENAI_ENDPOINT)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Azure OpenAI configuration
if USE_AZURE_OPENAI:
    # Extract deployment name from endpoint if not provided
    if "/deployments/" in AZURE_OPENAI_ENDPOINT:
        parts = AZURE_OPENAI_ENDPOINT.split("/deployments/")
        AZURE_BASE_URL = parts[0]
        deployment_part = parts[1].split("/")[0]  # Get just the deployment name, before any /chat/completions
        if "?" in deployment_part:
            deployment_part = deployment_part.split("?")[0]
        AZURE_DEPLOYMENT_NAME = deployment_part
    else:
        AZURE_BASE_URL = AZURE_OPENAI_ENDPOINT
        AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1")
    
    # Remove /openai from base URL if present (Azure OpenAI base URL should not include /openai)
    if AZURE_BASE_URL.endswith("/openai"):
        AZURE_BASE_URL = AZURE_BASE_URL[:-7]
else:
    AZURE_DEPLOYMENT_NAME = None
    AZURE_BASE_URL = None

# ============ Optional Dependencies ============
PDF_AVAILABLE = True
DOCX_AVAILABLE = True
RDFLIB_AVAILABLE = True
LLAMA_AVAILABLE = True

# ============ Global State Variables ============
G = None
PARTITION = {}
VECTOR_INDEX = None
ENABLE_VECTOR = False
VECTOR_TOPK = 8

SEARCH_OPTIONS = []
CLUSTER_SUMMARIES = {}
GRAPH_CONTEXT_MEMORY = ""
RAW_COMBINED_TEXT = ""

# ============ CFO Analytics Configuration ============
CFO_DIMENSIONS_AND_QUESTIONS = [
    ("Financial Exposure & Obligations", "What is the total value of active contracts by business unit and geography?"),
    ("Financial Exposure & Obligations", "How much committed spend (future obligations) exists across all contracts?"),
    ("Financial Exposure & Obligations", "Which vendors/customers represent the top 10% of our contractual spend or revenue?"),
    ("Risk & Compliance", "Are there clauses exposing the company to penalties, liquidated damages, or termination risks?"),
    ("Cash Flow & Working Capital", "What is the average payment term (days payable/receivable) across contracts?")
]

# ============ Extraction Configuration ============
EXTRACT_TEMP = 0.1
STRICT_PROMPT = False
REQUIRE_EVIDENCE = True
ENABLE_FALLBACK_COOCC = False
MAX_COOCC_PAIRS = 5
CHUNK_MAX_CHARS = 2500

# ============ Relation Policy Configuration ============
RELATION_POLICY = "Off"  # Off | Standard | Strict
ENABLE_ALIAS_NORMALIZATION = True
SKIP_EDGES_NO_EVID = True

REL_ALLOW_STANDARD = {
    "governed_by", "audited_by", "recognised_under", "modifies",
    "involves_counterparty", "has_obligation", "linked_to", "evidenced_by"
}

REL_ALLOW_STRICT = {
    "governed_by", "audited_by", "recognised_under", "has_obligation", "evidenced_by"
}

REL_MAP = {
    "recognized_under": "recognised_under", 
    "recognized by": "recognised_under", 
    "governed by": "governed_by"
}

ALIASES = {
    "asc 606": "ASC 606", "ifrs 15": "IFRS 15", "asc 842": "ASC 842", "ifrs 16": "IFRS 16",
    "right-of-use asset": "ROU asset", "acme holdings limited": "Acme Holdings Ltd."
}

# ============ Clustering Configuration ============
LOUVAIN_RESOLUTION = 1.25
LOUVAIN_RANDOM_STATE = 202

# ============ Ontologies for Clustering ============
ONTO_REL_BOOST = {
    "recognised_under": 1.50,
    "audited_by": 1.40,
    "governed_by": 1.30,
    "has_obligation": 1.20,
    "involves_counterparty": 1.10,
}

# ============ Q&A Configuration ============
MAX_EDGES_IN_PROMPT = 48
MAX_EVIDENCE_CHARS = 220
EDGE_NAME_WEIGHT = 3
EDGE_REL_WEIGHT = 2
EDGE_EVID_WEIGHT = 1

# ============ File Processing Configuration ============
SUPPORTED_TYPES = {
    "PDF": [".pdf"],
    "Word": [".docx"],
    "Text": [".txt"]
}

# ============ RDF/Ontology Configuration ============
DEFAULT_BASE_IRI = "http://example.org/kg/"
DEFAULT_ONTO_IRI = "http://example.org/onto/"
DEFAULT_CONTEXT_IRI = "http://example.org/context/"
DEFAULT_ONTOLOGY_FILENAME = "business_ontology.ttl"
DEFAULT_RDF_TTL_FILENAME = "graph_export.ttl"
DEFAULT_JSONLD_FILENAME = "graph_export.jsonld"

REL_TO_PROP = {
    "audited_by": "auditedBy",
    "governed_by": "governedBy",
    "recognised_under": "recognisedUnder",
    "has_obligation": "hasObligation",
    "involves_counterparty": "involvesCounterparty",
    "maps_to_standard": "mapsToStandard",
    "linked_to": "linkedTo",
    "evidenced_by": "evidencedBy",
    "co_occurs_in_sentence": "coOccursInSentence",
}

# ============ CFO Export Configuration ============
DEFAULT_JSONL_PATH = "cfo_contract_insights.jsonl"

# Initialize global state
def initialize_globals():
    """Initialize global state variables"""
    global G, PARTITION, SEARCH_OPTIONS, CLUSTER_SUMMARIES, GRAPH_CONTEXT_MEMORY, RAW_COMBINED_TEXT
    
    # Try to load existing GraphRAG state
    try:
        from graphrag_persistence import (
            load_existing_graph_graphrag, 
            load_existing_partition, 
            load_existing_context_memory,
            load_existing_raw_text
        )
        
        G = load_existing_graph_graphrag()
        PARTITION = load_existing_partition()
        GRAPH_CONTEXT_MEMORY = load_existing_context_memory()
        RAW_COMBINED_TEXT = load_existing_raw_text()
        
        print(f"[Config] Loaded persistent GraphRAG state - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
    except Exception as e:
        print(f"[Config] No existing GraphRAG state found: {e}")
        G = None
        PARTITION = {}
        GRAPH_CONTEXT_MEMORY = ""
        RAW_COMBINED_TEXT = ""
    
    # Initialize other global state
    SEARCH_OPTIONS = []
    CLUSTER_SUMMARIES = {}
