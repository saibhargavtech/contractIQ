import os
import re
import json
import math
import itertools
from typing import List, Tuple
from collections import Counter, defaultdict
from urllib.parse import quote

import openai
import streamlit as st
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from io import BytesIO
import traceback

# Import everything from promptrefreshed.py (except the Gradio UI)
import sys
import importlib.util

# Create a mock Gradio module to prevent import errors
class MockGradio:
    class update:
        @staticmethod
        def __call__(*args, **kwargs):
            return None
    
    @staticmethod
    def update(*args, **kwargs):
        return None

# Execute promptrefreshed.py but stop before the UI section
spec = importlib.util.spec_from_file_location("graphrag_functions", "promptrefreshed.py")
graphrag_module = importlib.util.module_from_spec(spec)

# Add mock gradio to module namespace before execution
graphrag_module.__dict__['gr'] = MockGradio()

# Read and execute the file up to the UI section
with open("promptrefreshed.py", "r", encoding="utf-8") as f:
    content = f.read()
    # Stop before the Gradio UI section
    ui_start = content.find("# ============ 10) UI ============")
    if ui_start > 0:
        content = content[:ui_start]
    
    # Remove Gradio import to avoid dependency issues in Streamlit
    # Replace gradio import with a mock to prevent errors
    content = content.replace("import gradio as gr", "# import gradio as gr  # Skipped for Streamlit - using mock")
    content = content.replace("from gradio", "# from gradio  # Skipped for Streamlit - using mock")
    
    # Execute in module context
    exec(content, graphrag_module.__dict__)

# Import all functions and variables we need
for name in dir(graphrag_module):
    if not name.startswith('_') or name in ['_chat_strict', '_normalize_types', '_read_pdf', '_read_docx', 
                                             '_read_txt', '_read_one_file', '_list_dir_files', '_chunk',
                                             '_parse_json_or_lines', '_fallback_cooccurrence', '_entropy',
                                             '_gini', '_giant_component_undirected', '_norm_rel', 
                                             '_relation_allowed', '_canon_entity', '_rule_based_triples',
                                             '_to_weighted_undirected', '_cluster_card', '_build_llm_context',
                                             '_tokenize_question', '_edge_score_for_question', '_select_relevant_edges',
                                             '_cluster_id_from_text', '_summarize_cluster_by_id', '_describe_entity_freeform',
                                             '_heuristic_cluster_label_and_summary', '_slug', '_node_iri',
                                             '_build_search_options', '_compute_entity_table', '_compute_relationship_table',
                                             '_compute_cluster_table', '_chat_bridge', '_build_cfo_prompt_from_context',
                                             '_cfo_json_schema_instruction', '_json_safe_load', '_normalize_records']:
        try:
            globals()[name] = getattr(graphrag_module, name)
        except:
            pass

# Override global variables with session state
def init_session_state():
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.DiGraph()
    if 'partition' not in st.session_state:
        st.session_state.partition = {}
    if 'search_options' not in st.session_state:
        st.session_state.search_options = []
    if 'cluster_summaries' not in st.session_state:
        st.session_state.cluster_summaries = {}
    if 'graph_context_memory' not in st.session_state:
        st.session_state.graph_context_memory = ""
    if 'raw_combined_text' not in st.session_state:
        st.session_state.raw_combined_text = ""
    if 'vector_index' not in st.session_state:
        st.session_state.vector_index = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'contract_status' not in st.session_state:
        st.session_state.contract_status = ""
    
    # Sync with global variables
    global G, partition, SEARCH_OPTIONS, CLUSTER_SUMMARIES, GRAPH_CONTEXT_MEMORY, RAW_COMBINED_TEXT, VECTOR_INDEX, ENABLE_VECTOR, VECTOR_TOPK
    G = st.session_state.graph
    partition = st.session_state.partition
    SEARCH_OPTIONS = st.session_state.search_options
    CLUSTER_SUMMARIES = st.session_state.cluster_summaries
    GRAPH_CONTEXT_MEMORY = st.session_state.graph_context_memory
    RAW_COMBINED_TEXT = st.session_state.raw_combined_text
    VECTOR_INDEX = st.session_state.vector_index
    ENABLE_VECTOR = False  # Can be set from UI
    VECTOR_TOPK = 8  # Can be set from UI

init_session_state()

# Set up Streamlit page
st.set_page_config(
    page_title="Contract GraphRAG Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main UI
st.title("📊 Contract GraphRAG Studio")

# File Upload Section (same as Gradio)
col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader(
        "Upload documents (.pdf, .docx, .txt)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
with col2:
    dir_path = st.text_input(
        "OR: Directory path (will recurse)",
        placeholder="e.g., C:/Users/Me/Documents/contracts"
    )

type_choices = st.multiselect(
    "Include file types",
    ["PDF", "Word", "Text"],
    default=["PDF", "Word", "Text"]
)

# Two column layout for settings (like Gradio)
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### Vector Search")
    enable_vector = st.checkbox("Enable Vector Search (LlamaIndex)", value=False)
    vector_topk = st.slider("Vector top-k", 2, 16, 8)
    
    st.markdown("### RDF / Ontology (optional)")
    base_iri = st.text_input("Base IRI (entities)", value="http://example.org/kg/")
    onto_iri = st.text_input("Ontology IRI", value="http://example.org/onto/")
    use_onto_cluster = st.checkbox("Use ontology-weighted clustering", value=False)
    
    st.markdown("**Parsers:** PDF, Word, Text | **RDF:** Available")

with col_right:
    st.markdown("### Graph Tuning")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        extract_temp = st.slider("Extraction temperature", 0.0, 1.0, 0.4, 0.05)
        strict_prompt = st.checkbox("Strict prompt", value=True)
        require_evidence = st.checkbox("Require evidence", value=True)
    with col_t2:
        enable_fallback = st.checkbox("Enable fallback", value=False)
        max_coocc_pairs = st.slider("Max co-occurrence", 0, 10, 5)
        chunk_size = st.slider("Chunk size (chars)", 2000, 5000, 3500, 100)
    with col_t3:
        relation_policy = st.selectbox("Relation whitelist", ["Off", "Standard", "Strict"], index=0)
        alias_norm = st.checkbox("Alias normalization", value=True)
        skip_no_evidence = st.checkbox("Skip no evidence", value=True)
    
    col_t4, col_t5 = st.columns(2)
    with col_t4:
        louvain_resolution = st.slider("Louvain resolution", 0.2, 2.0, 1.25, 0.05)
    with col_t5:
        louvain_random_state = st.slider("Random state", 1, 999, 202)

# Run button
if st.button("🚀 Run: Build Graph, Clusters & Metrics", type="primary", use_container_width=True):
    if not uploaded_files and not dir_path:
        st.error("Please upload files or provide a directory path")
    else:
        with st.spinner("Building knowledge graph..."):
            try:
                # Convert uploaded files to paths
                upload_paths = []
                if uploaded_files:
                    for file in uploaded_files:
                        # Save temporarily
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        upload_paths.append(temp_path)
                
                # Call the processing function
                result = process_corpus_with_contract_extraction(
                    upload_paths if upload_paths else None,
                    dir_path if dir_path else "",
                    type_choices,
                    enable_vector,
                    vector_topk,
                    extract_temp,
                    strict_prompt,
                    require_evidence,
                    enable_fallback,
                    max_coocc_pairs,
                    chunk_size,
                    relation_policy,
                    alias_norm,
                    skip_no_evidence,
                    louvain_resolution,
                    louvain_random_state,
                    use_onto_cluster
                )
                
                # Update session state
                st.session_state.graph_img = result[0] if len(result) > 0 else None
                st.session_state.summary_df = result[1] if len(result) > 1 else None
                st.session_state.metrics_df = result[2] if len(result) > 2 else None
                if len(result) > 3:
                    st.session_state.search_options = result[3] if result[3] else []
                st.session_state.contract_status = result[8] if len(result) > 8 else "Unknown"
                
                st.success("✅ Graph built successfully!")
                
            except Exception as e:
                st.error(f"Error building graph: {str(e)}")
                st.error(traceback.format_exc())

# Show status
if 'contract_status' in st.session_state:
    st.info(st.session_state.contract_status)

# Display outputs in 3 columns (like Gradio)
col_output1, col_output2, col_output3 = st.columns(3)

with col_output1:
    st.markdown("### 🕸️ Knowledge Graph")
    if 'graph_img' in st.session_state and st.session_state.graph_img:
        st.image(st.session_state.graph_img)
    else:
        st.info("No graph available. Upload files and click 'Run' to build a graph.")

with col_output2:
    st.markdown("### 🧾 Cluster Summaries")
    if 'summary_df' in st.session_state and st.session_state.summary_df is not None:
        st.dataframe(st.session_state.summary_df, use_container_width=True, height=300)
    else:
        st.info("No cluster summaries available.")

with col_output3:
    st.markdown("### 📈 Graph Metrics")
    if 'metrics_df' in st.session_state and st.session_state.metrics_df is not None:
        st.dataframe(st.session_state.metrics_df, use_container_width=True, height=300)
    else:
        st.info("No metrics available.")

# Search section
st.markdown("### 🔎 Search Entity or Cluster")
search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    if 'search_options' in st.session_state and st.session_state.search_options:
        search_term = st.selectbox(
            "Type to filter…",
            options=[""] + st.session_state.search_options,
            label_visibility="collapsed"
        )
    else:
        search_term = ""
        st.info("No search options available. Build a graph first.")

search_md = st.empty()

if search_term and st.button("Show Details", key="search_btn"):
    try:
        description = describe_selection(search_term)
        search_md.markdown(description)
    except Exception as e:
        search_md.error(f"Error: {str(e)}")

# Export section
st.markdown("### 📤 Export for CFO Dashboard")
export_col1, export_col2, export_col3, export_col4 = st.columns(4)

with export_col1:
    if st.button("Export CFO CSV/TXT"):
        try:
            paths = export_cfo_csv_and_txt()
            if paths:
                st.success("✅ Files exported!")
                for name, path in [("entities.csv", paths[0]), ("relationships.csv", paths[1]), 
                                   ("clusters.csv", paths[2]), ("cfo_dashboard_export.txt", paths[3])]:
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button(f"Download {name}", f.read(), file_name=name, key=f"dl_{name}")
        except Exception as e:
            st.error(f"Export error: {str(e)}")

with export_col2:
    if st.button("Build Ontology"):
        try:
            path = build_business_ontology(onto_iri=onto_iri, base_iri=base_iri)
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button("Download Ontology", f.read(), file_name="business_ontology.ttl", key="dl_onto")
        except Exception as e:
            st.error(f"Ontology build error: {str(e)}")

with export_col3:
    if st.button("Export RDF"):
        try:
            ttl_path, jsonld_path = export_graph_as_rdf(base_iri=base_iri, onto_iri=onto_iri)
            if ttl_path and os.path.exists(ttl_path):
                with open(ttl_path, "rb") as f:
                    st.download_button("Download TTL", f.read(), file_name="graph_export.ttl", key="dl_ttl")
        except Exception as e:
            st.error(f"RDF export error: {str(e)}")

with export_col4:
    if st.button("Export CFO JSONL"):
        try:
            path = export_cfo_jsonl_from_context("cfo_contract_insights.jsonl")
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button("Download JSONL", f.read(), file_name="cfo_contract_insights.jsonl", key="dl_jsonl")
        except Exception as e:
            st.error(f"JSONL export error: {str(e)}")

# Chat section
st.markdown("### 💬 Ask Questions")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the graph..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    try:
        response = graph_chat(prompt, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.error(error_msg)

# Footer
st.divider()
st.markdown("**Parsers:** PDF, Word, Text | **RDF:** Available | **Vector Search:** LlamaIndex")

