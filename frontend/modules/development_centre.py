"""
Development Centre - Graph Tuning Panel
Upload contracts and configure graph extraction parameters
"""

import streamlit as st
import os
import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import graph tuning functions from graphrag_streamlit
try:
    # Import the graph processing functions
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
    
    # Get the correct path to promptrefreshed.py (parent directory)
    current_dir = Path(__file__).parent.parent.parent
    promptrefreshed_path = current_dir / "promptrefreshed.py"
    
    # Execute promptrefreshed.py but stop before the UI section
    spec = importlib.util.spec_from_file_location("graphrag_functions", str(promptrefreshed_path))
    graphrag_module = importlib.util.module_from_spec(spec)
    graphrag_module.__dict__['gr'] = MockGradio()
    
    # Read and execute the file up to the UI section
    with open(str(promptrefreshed_path), "r", encoding="utf-8") as f:
        content = f.read()
        # Stop before the Gradio UI section
        ui_start = content.find("# ============ 10) UI ============")
        if ui_start > 0:
            content = content[:ui_start]
        
        # Remove Gradio import to avoid dependency issues in Streamlit
        content = content.replace("import gradio as gr", "# import gradio as gr  # Skipped for Streamlit - using mock")
        content = content.replace("from gradio", "# from gradio  # Skipped for Streamlit - using mock")
        
        # Execute in module context
        exec(content, graphrag_module.__dict__)
    
    # Import all functions we need
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
    
    # Also import public functions like describe_selection
    if hasattr(graphrag_module, 'describe_selection'):
        describe_selection = graphrag_module.describe_selection
    else:
        describe_selection = None
    
    # Import the processing function - use the public function name
    if hasattr(graphrag_module, 'process_corpus_with_contract_extraction'):
        process_corpus_with_contract_extraction = graphrag_module.process_corpus_with_contract_extraction
    else:
        process_corpus_with_contract_extraction = None
    
except Exception as e:
    st.error(f"Error loading graph processing functions: {str(e)}")
    st.error(traceback.format_exc())
    process_corpus_with_contract_extraction = None

def create_page(analytics_data):
    """Create Development Centre page with graph tuning panel"""
    
    st.markdown("# 🔧 Development Centre")
    st.markdown("*Upload contracts and configure graph extraction parameters*")
    st.markdown("---")
    
    # File Upload Section
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "Upload documents (.pdf, .docx, .txt)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload contract documents to process"
        )
    with col2:
        dir_path = st.text_input(
            "OR: Directory path (will recurse)",
            placeholder="e.g., C:/Users/Me/Documents/contracts",
            help="Provide a directory path to process all contracts in that folder"
        )
    
    type_choices = st.multiselect(
        "Include file types",
        ["PDF", "Word", "Text"],
        default=["PDF", "Word", "Text"]
    )
    
    # Two column layout for settings
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
    if st.button("🚀 Upload & Process Contracts", type="primary", use_container_width=True):
        if not uploaded_files and not dir_path:
            st.error("Please upload files or provide a directory path")
        else:
            if process_corpus_with_contract_extraction is None:
                st.error("Graph processing functions not available. Please check the error messages above.")
            else:
                with st.spinner("Processing contracts and building knowledge graph..."):
                    try:
                        # Convert uploaded files to paths
                        upload_paths = []
                        if uploaded_files:
                            # Create temp directory
                            temp_dir = "temp_uploads"
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            for file in uploaded_files:
                                # Save temporarily
                                temp_path = os.path.join(temp_dir, file.name)
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
                        
                        # Check if processing was successful
                        if result and len(result) > 0:
                            # Store results in session state
                            st.session_state.graph_img = result[0] if len(result) > 0 else None
                            st.session_state.summary_df = result[1] if len(result) > 1 else None
                            st.session_state.metrics_df = result[2] if len(result) > 2 else None
                            if len(result) > 3:
                                st.session_state.search_options = result[3] if result[3] else []
                            st.session_state.contract_status = result[8] if len(result) > 8 else "Processing completed"
                            
                            # Save graph context memory to file for chatbot
                            try:
                                # Get graph context memory from the graphrag module
                                if hasattr(graphrag_module, 'GRAPH_CONTEXT_MEMORY'):
                                    graph_context = graphrag_module.GRAPH_CONTEXT_MEMORY
                                    if graph_context and graph_context.strip() and graph_context != "(no graph memory)":
                                        # Save to root directory (where chatbot looks for it)
                                        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                                        context_file = os.path.join(root_dir, "graph_context_memory.txt")
                                        with open(context_file, 'w', encoding='utf-8') as f:
                                            f.write(graph_context)
                                        print(f"[✅ SAVED] Graph context memory saved to {context_file} ({len(graph_context)} chars)")
                                        st.session_state.graph_context = graph_context
                                        st.session_state.graph_ready = True
                                        st.info(f"✅ Graph context saved! Chatbot can now use uploaded contracts.")
                                    else:
                                        print(f"[⚠️ WARNING] Graph context is empty or invalid")
                                else:
                                    print(f"[⚠️ WARNING] GRAPH_CONTEXT_MEMORY not found in graphrag module")
                            except Exception as e:
                                print(f"[⚠️ WARNING] Could not save graph context: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Save cluster summaries to file for chatbot
                            try:
                                import json
                                # Get cluster summaries from the graphrag module
                                if hasattr(graphrag_module, 'CLUSTER_SUMMARIES'):
                                    cluster_summaries = graphrag_module.CLUSTER_SUMMARIES
                                    if cluster_summaries:
                                        # Save to root directory (where chatbot looks for it)
                                        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                                        cluster_file = os.path.join(root_dir, "cluster_summaries.json")
                                        with open(cluster_file, 'w', encoding='utf-8') as f:
                                            json.dump(cluster_summaries, f, indent=2, ensure_ascii=False)
                                        print(f"[✅ SAVED] Cluster summaries saved to {cluster_file} ({len(cluster_summaries)} clusters)")
                                        # Also update config if available
                                        try:
                                            import config
                                            config.CLUSTER_SUMMARIES = cluster_summaries
                                        except:
                                            pass
                                    else:
                                        print(f"[⚠️ WARNING] Cluster summaries are empty")
                                else:
                                    print(f"[⚠️ WARNING] CLUSTER_SUMMARIES not found in graphrag module")
                            except Exception as e:
                                print(f"[⚠️ WARNING] Could not save cluster summaries: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Show success message
                            st.success("✅ Contracts processed successfully!")
                            if st.session_state.contract_status:
                                st.info(st.session_state.contract_status)
                            
                            # Clear cache to force dashboard refresh
                            try:
                                # Import from frontend/utils.py (ensure frontend directory is in path first)
                                import sys
                                frontend_dir = os.path.dirname(os.path.dirname(__file__))
                                if frontend_dir not in sys.path:
                                    sys.path.insert(0, frontend_dir)
                                from utils import load_cfo_analytics_data, load_unified_contracts
                                load_cfo_analytics_data.clear()
                                load_unified_contracts.clear()
                            except:
                                pass
                            
                            # Set flag for main dashboard to refresh
                            st.session_state.upload_success = True
                            
                            # Show next steps
                            st.markdown("### ✅ Processing Complete!")
                            st.markdown("""
                            Your contracts have been:
                            - ✅ Extracted and processed
                            - ✅ Added to the dashboard
                            - ✅ Available for analysis
                            
                            **Next Steps:**
                            1. Go back to the main dashboard to see your updated contract portfolio
                            2. Use "Talk to Your Contracts" to ask questions about your contracts
                            3. View analytics and insights in the other dashboard pages
                            """)
                            
                        else:
                            st.warning("Processing completed but no contract data was extracted. Please check your documents.")
                            
                    except Exception as e:
                        st.error(f"Error processing contracts: {str(e)}")
                        st.error(traceback.format_exc())
                    finally:
                        # Clean up temp files
                        if uploaded_files:
                            try:
                                import shutil
                                if os.path.exists("temp_uploads"):
                                    shutil.rmtree("temp_uploads")
                            except:
                                pass
    
    # Show status if available
    if 'contract_status' in st.session_state and st.session_state.get('upload_success'):
        st.info(f"📋 Status: {st.session_state.contract_status}")
    
    # Display graph visualization, clusters, and metrics
    st.markdown("---")
    st.markdown("## 📊 Graph Analysis Results")
    
    # Display outputs in 3 columns (like the original graphrag_streamlit)
    col_output1, col_output2, col_output3 = st.columns(3)
    
    with col_output1:
        st.markdown("### 🕸️ Knowledge Graph")
        if 'graph_img' in st.session_state and st.session_state.graph_img is not None:
            # graph_img could be a PIL Image, numpy array, or file path
            try:
                if isinstance(st.session_state.graph_img, str):
                    # It's a file path
                    if os.path.exists(st.session_state.graph_img):
                        st.image(st.session_state.graph_img, use_container_width=True)
                    else:
                        st.info("Graph image file not found.")
                else:
                    # It's a PIL Image or numpy array
                    st.image(st.session_state.graph_img, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display graph: {str(e)}")
                # Try to find graph image file
                graph_img_path = "graph_visualization.png"
                if os.path.exists(graph_img_path):
                    st.image(graph_img_path, use_container_width=True)
                else:
                    st.info("No graph available. Upload files and click 'Upload & Process Contracts' to build a graph.")
        else:
            # Try to find graph image file
            graph_img_path = "graph_visualization.png"
            if os.path.exists(graph_img_path):
                st.image(graph_img_path, use_container_width=True)
            else:
                st.info("No graph available. Upload files and click 'Upload & Process Contracts' to build a graph.")
    
    with col_output2:
        st.markdown("### 🧾 Cluster Summaries")
        if 'summary_df' in st.session_state and st.session_state.summary_df is not None:
            try:
                st.dataframe(st.session_state.summary_df, use_container_width=True, height=400)
            except Exception as e:
                st.warning(f"Could not display cluster summaries: {str(e)}")
                st.info("No cluster summaries available.")
        else:
            st.info("No cluster summaries available. Process contracts to see cluster analysis.")
    
    with col_output3:
        st.markdown("### 📈 Graph Metrics")
        if 'metrics_df' in st.session_state and st.session_state.metrics_df is not None:
            try:
                st.dataframe(st.session_state.metrics_df, use_container_width=True, height=400)
            except Exception as e:
                st.warning(f"Could not display metrics: {str(e)}")
                st.info("No metrics available.")
        else:
            st.info("No metrics available. Process contracts to see graph metrics.")
    
    # Search section (if search options are available)
    if 'search_options' in st.session_state and st.session_state.search_options:
        st.markdown("---")
        st.markdown("### 🔎 Search Entity or Cluster")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_term = st.selectbox(
                "Type to filter…",
                options=[""] + st.session_state.search_options,
                label_visibility="collapsed"
            )
        
        if search_term:
            try:
                # Try to get description of the selected entity/cluster
                if describe_selection:
                    description = describe_selection(search_term)
                    st.markdown(description)
                else:
                    st.info(f"Selected: {search_term}")
            except Exception as e:
                st.info(f"Selected: {search_term}")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About Graph Tuning Parameters"):
        st.markdown("""
        **Extraction Temperature (0.0-1.0):**
        - Lower values = more deterministic, fewer hallucinations
        - Higher values = more creative, risk of noise
        
        **Strict Prompt:**
        - Rejects weak edges for higher precision
        - More permissive when disabled for higher recall
        
        **Require Evidence:**
        - Only creates edges with source sentences
        - Ensures traceability and accuracy
        
        **Chunk Size:**
        - Smaller = more API calls, better focus
        - Larger = fewer calls, risk of truncation
        
        **Relation Whitelist:**
        - Off = all relations allowed
        - Standard = 8 financial relations
        - Strict = 5 core relations
        
        **Louvain Resolution:**
        - Lower = fewer, larger communities
        - Higher = more, smaller communities
        """)

