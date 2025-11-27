"""
Contract Chatbot Page
Q&A interface using GraphRAG technology, CSV analysis, and OpenAI API
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, List
import sys
import openai

# Add parent directory to path for imports (works for both local and Streamlit Cloud)
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import modules, create fallbacks if not available
try:
    import config
except ImportError:
    class Config:
        CFO_DIMENSIONS_AND_QUESTIONS = [
            ("Financial", "What is the total value of all contracts?"),
            ("Financial", "What are the payment terms across all contracts?"),
            ("Vendor", "Which vendors have the highest contract values?"),
            ("Risk", "What are the compliance requirements across all contracts?"),
            ("Performance", "What are the SLA requirements for all contracts?")
        ]
    config = Config()

try:
    import graph
except ImportError:
    class Graph:
        @staticmethod
        def detect_communities(G):
            return G, {}
    graph = Graph()

try:
    import clustering
except ImportError:
    class Clustering:
        @staticmethod
        def summarize_clusters(G, partition, data):
            return {}
        
        @staticmethod
        def build_graph_context_memory(G, partition, summaries):
            return ""
    clustering = Clustering()

try:
    import cfo_analytics
except ImportError:
    class CFOAnalytics:
        @staticmethod
        def generate_cfo_insights_from_context(context):
            return []
    cfo_analytics = CFOAnalytics()

# Set OpenAI API key from Streamlit secrets or environment variable
from dotenv import load_dotenv

# Priority: Streamlit secrets > environment variable > .env file
api_key = None

# Try Streamlit secrets first (for Streamlit Cloud deployment)
try:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
except (AttributeError, KeyError, FileNotFoundError):
    # Streamlit secrets not available, try environment variable
    pass

# Fallback to environment variable
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

# Fallback to .env file (for local development)
if not api_key:
    load_dotenv()  # Root directory
    if not os.getenv("OPENAI_API_KEY"):
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))  # Frontend directory
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found. Please set it in Streamlit secrets (for cloud) or .env file (for local).")
    st.stop()

openai.api_key = api_key

def create_page(analytics_data: Dict):
    """Create Contract Chatbot page with GraphRAG integration"""
    
    st.markdown("## 💬 Talk to Your Contracts")
    st.markdown("*Ask questions about your contract portfolio using AI-powered insights*")
    
    # Initialize session state for chat
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'graph_ready' not in st.session_state:
        st.session_state.graph_ready = False
    
    if 'graph_context' not in st.session_state:
        st.session_state.graph_context = ""
    
    # Auto-load graph context if it exists (check multiple locations)
    # Also reload if file has been updated (check file modification time)
    context_files = [
        "../graph_context_memory.txt",  # Parent directory
        "graph_context_memory.txt",      # Current directory
        os.path.join(os.path.dirname(__file__), "..", "..", "graph_context_memory.txt")  # Root directory
    ]
    
    # Check if we need to reload (file exists and either not loaded or file was modified)
    should_reload = False
    context_file_path = None
    for context_file in context_files:
        if os.path.exists(context_file):
            context_file_path = context_file
            # Check if file was modified since last load
            file_mtime = os.path.getmtime(context_file)
            last_load_time = st.session_state.get('graph_context_mtime', 0)
            if not st.session_state.graph_ready or file_mtime > last_load_time:
                should_reload = True
            break
    
    if should_reload and context_file_path:
        try:
            with open(context_file_path, 'r', encoding='utf-8') as f:
                st.session_state.graph_context = f.read()
            st.session_state.graph_ready = True
            st.session_state.graph_context_mtime = os.path.getmtime(context_file_path)
            print(f"[Chatbot] Reloaded graph context from {context_file_path} ({len(st.session_state.graph_context)} chars)")
        except Exception as e:
            print(f"[Chatbot] Error loading graph context: {e}")
    elif not st.session_state.graph_ready:
        # Try to load for the first time
        for context_file in context_files:
            if os.path.exists(context_file):
                try:
                    with open(context_file, 'r', encoding='utf-8') as f:
                        st.session_state.graph_context = f.read()
                    st.session_state.graph_ready = True
                    st.session_state.graph_context_mtime = os.path.getmtime(context_file)
                    break
                except Exception:
                    continue
    
    # Check if graph is already processed
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🤖 Contract Intelligence Assistant")
        
        # Check if graph context memory exists (check multiple locations)
        context_files = [
            "../graph_context_memory.txt",
            "graph_context_memory.txt",
            os.path.join(os.path.dirname(__file__), "..", "..", "graph_context_memory.txt")
        ]
        
        context_found = any(os.path.exists(f) for f in context_files)
        
        if context_found and st.session_state.graph_ready:
            st.success("✅ Knowledge graph ready! You can now ask questions about your contracts.")
        elif context_found:
            st.info("🔄 Loading knowledge graph context...")
            st.rerun()
        else:
            st.info("ℹ️ Using contract data analysis. Graph context will enhance answers when available.")
            if st.button("🔄 Check for Graph Context", type="secondary"):
                st.rerun()
    
    with col2:
        st.markdown("### 📊 System Status")
        if st.session_state.graph_ready:
            st.success("🟢 Graph Ready")
            st.metric("Contracts Processed", len(analytics_data["contract_csv"]) if analytics_data["contract_csv"] is not None else 0)
        else:
            st.warning("🟡 Graph Not Ready")
            st.info("Click 'Build Knowledge Graph' to start")
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write(f"Graph Context Length: {len(st.session_state.get('graph_context', ''))}")
            st.write(f"Contract Data Available: {analytics_data.get('contract_csv') is not None}")
            if analytics_data.get('contract_csv') is not None:
                st.write(f"Contract Count: {len(analytics_data['contract_csv'])}")
            st.write(f"OpenAI API Key Set: {'Yes' if openai.api_key else 'No'}")
    
    # Chat interface (always available)
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About Your Contracts")
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your contracts (e.g., 'What are the payment terms for IBM contracts?')"):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your contracts..."):
                try:
                    response = generate_ai_response(prompt, analytics_data)
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}. Please try rephrasing your question."
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()
    
    # Sample questions
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    
    sample_questions = [
        "What are the top vendors by contract value?",
        "Which contracts are expiring in the next 6 months?",
        "What is the timeline of IBM contracts?",
        "What contracts have unfavorable payment terms?",
        "What is our penalty exposure for SLA breaches?",
        "What contracts are expiring in the next 2 years?",
        "Which vendors have the highest concentration risk?",
        "What are the compliance requirements across all contracts?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"💬 {question}", key=f"sample_{i}", use_container_width=True):
                # Add sample question to chat (always works, even without graph)
                st.session_state.chat_messages.append({"role": "user", "content": question})
                st.rerun()

def build_knowledge_graph(analytics_data: Dict) -> bool:
    """Load existing knowledge graph from processed context memory"""
    try:
        # Check multiple locations for graph context
        context_files = [
            "../graph_context_memory.txt",  # Parent directory
            "graph_context_memory.txt",      # Current directory
            os.path.join(os.path.dirname(__file__), "..", "..", "graph_context_memory.txt")  # Root directory
        ]
        
        for context_file in context_files:
            if os.path.exists(context_file):
                try:
                    with open(context_file, 'r', encoding='utf-8') as f:
                        context_memory = f.read()
                    
                    if context_memory and context_memory.strip():
                        st.session_state.graph_context = context_memory
                        st.session_state.graph_ready = True
                        return True
                except Exception:
                    continue
        
        # If no context found, that's okay - chatbot can still work with CSV data
        st.info("ℹ️ Graph context not found. Chatbot will use contract data analysis.")
        return False
        
    except Exception as e:
        st.warning(f"Note: Could not load graph context: {str(e)}. Chatbot will use contract data.")
        return False

def generate_ai_response(prompt: str, analytics_data: Dict) -> str:
    """Generate AI response using Graph Context + Cluster Summaries + CSV Analysis + OpenAI API"""
    try:
        # Get all available data sources in priority order
        graph_context = st.session_state.get('graph_context', '')
        
        # Get cluster summaries from file or config (priority: file > config)
        cluster_summaries = {}
        try:
            # Try to load from file first (most up-to-date)
            cluster_files = [
                "../cluster_summaries.json",
                "cluster_summaries.json",
                os.path.join(os.path.dirname(__file__), "..", "..", "cluster_summaries.json")
            ]
            for cluster_file in cluster_files:
                if os.path.exists(cluster_file):
                    try:
                        with open(cluster_file, 'r', encoding='utf-8') as f:
                            cluster_summaries = json.load(f)
                        break
                    except Exception:
                        continue
            
            # Fallback to config if file not found
            if not cluster_summaries:
                if hasattr(config, 'CLUSTER_SUMMARIES') and config.CLUSTER_SUMMARIES:
                    cluster_summaries = config.CLUSTER_SUMMARIES
        except Exception as e:
            print(f"[Chatbot] Error loading cluster summaries: {e}")
            pass
        
        df = analytics_data.get("contract_csv", pd.DataFrame())
        
        if df.empty:
            return "❌ No contract data available. Please upload contracts in the Development Centre first."
        
        # Generate comprehensive CSV analysis
        csv_analysis = generate_comprehensive_csv_analysis(df, prompt)
        
        if not csv_analysis or not csv_analysis.strip():
            return "❌ Unable to analyze contract data. Please check your data sources."
        
        # Combine all context for OpenAI (priority: Graph Context → Cluster Summaries → CSV)
        combined_context = build_combined_context(graph_context, cluster_summaries, csv_analysis, prompt)
        
        # Use OpenAI API for intelligent response
        try:
            response = generate_openai_response(prompt, combined_context)
            if response and response.strip():
                return response
            else:
                # If OpenAI returns empty, fall back to CSV analysis
                st.warning("⚠️ OpenAI returned empty response, using CSV analysis")
                return generate_simple_csv_response(prompt, df)
        except Exception as openai_error:
            # If OpenAI fails, use CSV analysis as fallback
            error_msg = str(openai_error)
            st.warning(f"⚠️ OpenAI API error: {error_msg[:100]}... Using CSV analysis as fallback.")
            print(f"[Chatbot] OpenAI error: {error_msg}")
            return generate_simple_csv_response(prompt, df)
        
    except Exception as e:
        # Final fallback to simple CSV analysis
        try:
            df = analytics_data.get("contract_csv", pd.DataFrame())
            if not df.empty:
                return generate_simple_csv_response(prompt, df)
            else:
                return f"❌ I encountered an error: {str(e)}. Please try rephrasing your question or upload contracts first."
        except Exception as e2:
            return f"❌ I encountered an error: {str(e)}. Please try rephrasing your question."

def generate_comprehensive_csv_analysis(df: pd.DataFrame, prompt: str) -> str:
    """Generate comprehensive CFO-level CSV analysis based on the question"""
    try:
        analysis = []
        prompt_lower = prompt.lower()
        
        # Remove duplicates first to get accurate data
        df_clean = df.drop_duplicates(subset=['contract_id'], keep='first')
        
        # Always include portfolio overview
        total_contracts = len(df_clean)
        total_value = df_clean['total_value_usd'].sum()
        total_annual = df_clean['annual_commitment_usd'].sum()
        active_contracts = len(df_clean[df_clean['status'].str.contains('Active', na=False)])
        
        analysis.append(f"=== PORTFOLIO OVERVIEW ===")
        analysis.append(f"• Total Contracts: {total_contracts}")
        analysis.append(f"• Total Portfolio Value: ${total_value:,.0f}")
        analysis.append(f"• Total Annual Commitments: ${total_annual:,.0f}")
        analysis.append(f"• Active Contracts: {active_contracts}")
        analysis.append(f"• Average Contract Value: ${total_value/total_contracts:,.0f}")
        
        # Top Vendors Analysis
        if any(word in prompt_lower for word in ['vendor', 'supplier', 'counterparty', 'top', 'concentration']):
            vendor_analysis = df_clean.groupby('counterparty').agg({
                'total_value_usd': 'sum',
                'annual_commitment_usd': 'sum',
                'contract_id': 'count'
            }).sort_values('total_value_usd', ascending=False)
            
            analysis.append(f"\n=== TOP VENDORS ANALYSIS ===")
            for i, (vendor, data) in enumerate(vendor_analysis.head(5).iterrows()):
                pct = (data['total_value_usd'] / total_value) * 100
                analysis.append(f"• {i+1}. {vendor}: ${data['total_value_usd']:,.0f} ({pct:.1f}% of portfolio) - {data['contract_id']} contracts")
            
            # Concentration risk
            top_vendor_pct = (vendor_analysis.iloc[0]['total_value_usd'] / total_value) * 100
            if top_vendor_pct > 50:
                analysis.append(f"⚠️ HIGH CONCENTRATION RISK: Top vendor represents {top_vendor_pct:.1f}% of portfolio")
            elif top_vendor_pct > 30:
                analysis.append(f"⚠️ MODERATE CONCENTRATION RISK: Top vendor represents {top_vendor_pct:.1f}% of portfolio")
        
        # Contract Expiry Analysis
        if any(word in prompt_lower for word in ['expiring', 'renewal', 'expiry', 'timeline', '6 months', '2 years']):
            df_temp = df_clean.copy()
            df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
            df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
            
            # 6 months
            expiring_6m = df_temp[df_temp['months_to_expiry'] <= 6]
            # 2 years
            expiring_2y = df_temp[df_temp['months_to_expiry'] <= 24]
            
            analysis.append(f"\n=== CONTRACT EXPIRY ANALYSIS ===")
            analysis.append(f"• Expiring in 6 months: {len(expiring_6m)} contracts (${expiring_6m['annual_commitment_usd'].sum():,.0f} at risk)")
            analysis.append(f"• Expiring in 2 years: {len(expiring_2y)} contracts (${expiring_2y['annual_commitment_usd'].sum():,.0f} at risk)")
            
            if not expiring_6m.empty:
                analysis.append(f"• Critical Renewals Needed: {', '.join(expiring_6m['counterparty'].unique())}")
                analysis.append(f"• Contract IDs Expiring: {', '.join(expiring_6m['contract_id'].astype(str).unique())}")
        
        # Payment Terms Analysis
        if any(word in prompt_lower for word in ['payment', 'terms', 'unfavorable']):
            payment_terms = df_clean['payment_terms'].value_counts()
            unfavorable = df_clean[df_clean['payment_terms'].isin(['Net 60', 'Net 90', 'Milestone-based'])]
            
            analysis.append(f"\n=== PAYMENT TERMS ANALYSIS ===")
            for term, count in payment_terms.items():
                analysis.append(f"• {term}: {count} contracts")
            
            if not unfavorable.empty:
                analysis.append(f"⚠️ UNFAVORABLE TERMS RISK: {len(unfavorable)} contracts (${unfavorable['annual_commitment_usd'].sum():,.0f})")
                analysis.append(f"• Vendors with Unfavorable Terms: {', '.join(unfavorable['counterparty'].unique())}")
        
        # Risk Analysis
        if any(word in prompt_lower for word in ['penalty', 'sla', 'escalation', 'risk']):
            penalty_contracts = df_clean[df_clean['sla_penalty'].notna() & (df_clean['sla_penalty'] != '')]
            escalation_contracts = df_clean[df_clean['escalation'].notna() & (df_clean['escalation'] != 'None')]
            
            analysis.append(f"\n=== RISK ANALYSIS ===")
            analysis.append(f"• SLA Penalty Risk: {len(penalty_contracts)} contracts (${penalty_contracts['total_value_usd'].sum():,.0f})")
            analysis.append(f"• Escalation Risk: {len(escalation_contracts)} contracts (${escalation_contracts['annual_commitment_usd'].sum():,.0f})")
            
            if not penalty_contracts.empty:
                analysis.append(f"• Penalty Risk Vendors: {', '.join(penalty_contracts['counterparty'].unique())}")
            if not escalation_contracts.empty:
                analysis.append(f"• Escalation Risk Vendors: {', '.join(escalation_contracts['counterparty'].unique())}")
        
        # Specific Vendor Analysis (e.g., IBM)
        for vendor in ['IBM', 'Microsoft', 'Oracle', 'SAP', 'Salesforce', 'HCL', 'Infosys', 'Alpha']:
            if vendor.lower() in prompt_lower:
                vendor_contracts = df_clean[df_clean['counterparty'].str.contains(vendor, case=False, na=False)]
                if not vendor_contracts.empty:
                    analysis.append(f"\n=== {vendor.upper()} CONTRACTS DETAILED ANALYSIS ===")
                    analysis.append(f"• Contract Count: {len(vendor_contracts)}")
                    analysis.append(f"• Total Value: ${vendor_contracts['total_value_usd'].sum():,.0f}")
                    analysis.append(f"• Annual Commitment: ${vendor_contracts['annual_commitment_usd'].sum():,.0f}")
                    analysis.append(f"• Portfolio Share: {(vendor_contracts['total_value_usd'].sum() / total_value) * 100:.1f}%")
                    
                    # Timeline for this vendor
                    vendor_timeline = vendor_contracts[['contract_id', 'start_date', 'end_date', 'annual_commitment_usd', 'status']].copy()
                    vendor_timeline['start_date'] = pd.to_datetime(vendor_timeline['start_date'])
                    vendor_timeline['end_date'] = pd.to_datetime(vendor_timeline['end_date'])
                    vendor_timeline = vendor_timeline.sort_values('start_date')
                    
                    analysis.append(f"• Contract Timeline:")
                    for _, contract in vendor_timeline.iterrows():
                        analysis.append(f"  - {contract['contract_id']}: {contract['start_date'].strftime('%Y-%m')} to {contract['end_date'].strftime('%Y-%m')} (${contract['annual_commitment_usd']:,.0f}/yr) - {contract['status']}")
        
        # Contract Types
        if 'type' in df_clean.columns:
            contract_types = df_clean['type'].value_counts()
            analysis.append(f"\n=== CONTRACT TYPES BREAKDOWN ===")
            for ctype, count in contract_types.items():
                pct = (count / total_contracts) * 100
                analysis.append(f"• {ctype}: {count} contracts ({pct:.1f}%)")
        
        # Financial Summary
        analysis.append(f"\n=== FINANCIAL SUMMARY ===")
        analysis.append(f"• Total Portfolio Value: ${total_value:,.0f}")
        analysis.append(f"• Annual Commitments: ${total_annual:,.0f}")
        analysis.append(f"• Average Annual Commitment: ${total_annual/total_contracts:,.0f}")
        analysis.append(f"• Contract Value Range: ${df_clean['total_value_usd'].min():,.0f} - ${df_clean['total_value_usd'].max():,.0f}")
        
        return "\n".join(analysis)
    
    except Exception as e:
        return f"Error in CSV analysis: {str(e)}"

def build_combined_context(graph_context: str, cluster_summaries: dict, csv_analysis: str, prompt: str) -> str:
    """Build combined context with correct priority: Graph Context → Cluster Summaries → CSV
    Limits context size to avoid OpenAI capacity issues"""
    context_parts = []
    
    # PRIORITY 1: Graph Context (Primary - Deepest Insights) - Limit to 2000 chars
    if graph_context and graph_context.strip():
        graph_truncated = graph_context[:2000] if len(graph_context) > 2000 else graph_context
        if len(graph_context) > 2000:
            graph_truncated += "\n[... graph context truncated to avoid capacity limits ...]"
        context_parts.append(f"PRIMARY - KNOWLEDGE GRAPH CONTEXT (Deep Insights):\n{graph_truncated}")
    
    # PRIORITY 2: Cluster Summaries (Secondary - Risk & KPI Insights) - Limit to 1500 chars
    if cluster_summaries:
        cluster_text = []
        for cid in sorted(cluster_summaries.keys()):
            summary_data = cluster_summaries[cid]
            label = summary_data.get("KPI/Risk Label", f"Cluster {cid}")
            summary = summary_data.get("Cluster Summary", "No summary available.")
            cluster_text.append(f"Cluster {cid} - {label}:\n{summary}")
        
        cluster_str = "\n\n".join(cluster_text)
        cluster_truncated = cluster_str[:1500] if len(cluster_str) > 1500 else cluster_str
        if len(cluster_str) > 1500:
            cluster_truncated += "\n[... cluster summaries truncated to avoid capacity limits ...]"
        context_parts.append(f"SECONDARY - CLUSTER SUMMARIES (Risk & KPI Insights):\n{cluster_truncated}")
    
    # PRIORITY 3: CSV Analysis (Tertiary - Structured Data) - Limit to 2000 chars
    if csv_analysis and csv_analysis.strip():
        csv_truncated = csv_analysis[:2000] if len(csv_analysis) > 2000 else csv_analysis
        if len(csv_analysis) > 2000:
            csv_truncated += "\n[... CSV analysis truncated to avoid capacity limits ...]"
        context_parts.append(f"TERTIARY - CONTRACT CSV ANALYSIS (Structured Data):\n{csv_truncated}")
    
    # Add direct analysis instructions
    context_parts.append("""
ANALYSIS INSTRUCTIONS:
- PRIORITY ORDER: Use Graph Context first (deepest insights), then Cluster Summaries (risk/KPI insights), then CSV data (structured facts)
- BE DIRECT: Answer the specific question asked, no extra analysis unless requested
- FORMAT: Use bullet points (•) for clarity
- BE SPECIFIC: Include exact dollar amounts, contract counts, vendor names, and dates
- USE ALL SOURCES: Combine insights from graph context, cluster summaries, and CSV data when relevant
- NO AUTO-RISK ASSESSMENT: Only provide risk/opportunity analysis when specifically asked
- FINANCIAL ACCURACY: All financial figures must be precise and properly formatted
""")
    
    combined = "\n\n".join(context_parts)
    
    # Final safety check - limit total context to 4000 chars to allow for all sources
    if len(combined) > 4000:
        combined = combined[:4000] + "\n[... context truncated to avoid capacity limits ...]"
    
    return combined

def generate_openai_response(prompt: str, context: str) -> str:
    """Generate CFO-level response using CSV data (priority) + Graph context for insights
    Uses standard OpenAI API (not Azure)"""
    try:
        # Use standard OpenAI only (no Azure)
        from openai import OpenAI
        # Always read API key from Streamlit secrets or environment variable
        current_api_key = None
        
        # Try Streamlit secrets first (for Streamlit Cloud)
        try:
            current_api_key = st.secrets.get("OPENAI_API_KEY", None)
        except (AttributeError, KeyError, FileNotFoundError):
            pass
        
        # Fallback to environment variable
        if not current_api_key:
            current_api_key = os.getenv("OPENAI_API_KEY")
        
        # Fallback to .env file (for local development)
        if not current_api_key:
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
            current_api_key = os.getenv("OPENAI_API_KEY")
        
        if not current_api_key:
            error_msg = "OPENAI_API_KEY not found. Please set it in Streamlit secrets (for cloud) or .env file (for local)."
            st.error(f"⚠️ {error_msg}")
            raise ValueError(error_msg)
        
        # Debug: Log that we're using the API key (first 10 chars only for security)
        if st.session_state.get('debug_mode', False):
            st.info(f"🔑 Using OpenAI API key: {current_api_key[:10]}...")
        
        client = OpenAI(api_key=current_api_key)
        model_name = "gpt-4"  # Use GPT-4.1 series for better quality
        
        messages = [
            {
                "role": "system",
                "content": """You are a direct, efficient CFO contract analyst assistant. Provide comprehensive answers using all available data sources.

CRITICAL INSTRUCTIONS - DATA SOURCE PRIORITY:
1. GRAPH CONTEXT (Primary) - Use deep insights from knowledge graph relationships and patterns
2. CLUSTER SUMMARIES (Secondary) - Use risk/KPI insights from contract clusters
3. CSV DATA (Tertiary) - Use exact numbers, dates, and contract details for factual accuracy

RESPONSE REQUIREMENTS:
1. BE COMPREHENSIVE - Combine insights from all available sources (graph, clusters, CSV)
2. BE DIRECT - Answer the specific question asked, no extra fluff
3. FORMAT: Use bullet points (•) for clarity and easy scanning
4. BE SPECIFIC: Include exact dollar amounts, contract counts, vendor names, and dates
5. USE GRAPH INSIGHTS: When graph context is available, use it to provide deeper analysis
6. USE CLUSTER INSIGHTS: When cluster summaries are available, reference relevant risks/KPIs
7. ONLY PROVIDE RISK/OPPORTUNITY ANALYSIS when explicitly asked about "risk", "opportunities", or "assessment"

RESPONSE STYLE:
• Straightforward answers to the question asked
• Use bullet points (•) for key findings
• Include exact financial figures ($X,XXX,XXX)
• Mention specific vendor names and contract IDs when relevant
• Combine graph insights with CSV data for comprehensive answers
• Reference cluster summaries when discussing risks or KPIs
• NO automatic risk assessments unless specifically requested
• NO strategic recommendations unless asked for

Focus on answering the specific question using ALL available data sources in priority order."""
            },
            {
                "role": "user",
                "content": f"""CONTRACT DATA ANALYSIS (Multiple Sources - Priority: Graph → Clusters → CSV):
{context}

QUESTION: {prompt}

Provide a comprehensive answer using:
• Graph context insights (if available) for deep analysis
• Cluster summaries (if available) for risk/KPI insights
• CSV data for exact numbers, dates, and contract details
• Clear bullet points for easy reading

Answer the question comprehensively using all available sources - be direct and specific."""
            }
        ]
        
        # Optimized for GPT-4.1 series - single request for quality responses
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1200,  # Increased for GPT-4.1 quality responses
            temperature=0.2,
            timeout=30  # 30 second timeout for GPT-4.1 processing
        )
                
        # Validate response
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            else:
                raise ValueError("Empty response from OpenAI")
        else:
            raise ValueError("Invalid response from OpenAI")
    
    except Exception as e:
        error_msg = str(e)
        # Log error for debugging
        print(f"[Chatbot OpenAI Error] {error_msg}")
        # Re-raise to be caught by outer handler (will trigger CSV fallback)
        raise Exception("Unable to generate AI response. Using contract data analysis instead.")

def generate_simple_csv_response(prompt: str, df: pd.DataFrame) -> str:
    """Simple fallback response using CSV data when OpenAI fails"""
    try:
        # Remove duplicates first
        df_clean = df.drop_duplicates(subset=['contract_id'], keep='first')
        prompt_lower = prompt.lower()
        
        if "total" in prompt_lower and "value" in prompt_lower:
            total_value = df_clean['total_value_usd'].sum()
            return f"Based on contract analysis, your total contract portfolio value is ${total_value:,.0f} across {len(df_clean)} contracts."
        
        elif "vendor" in prompt_lower and "top" in prompt_lower:
            vendor_analysis = df_clean.groupby('counterparty')['total_value_usd'].sum().sort_values(ascending=False)
            response = "Top vendors by contract value:\n"
            for i, (vendor, value) in enumerate(vendor_analysis.head(5).items()):
                response += f"{i+1}. {vendor}: ${value:,.0f}\n"
            return response
        
        elif "expiring" in prompt_lower and "6 months" in prompt_lower:
            df_temp = df_clean.copy()
            df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
            df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
            expiring_contracts = df_temp[df_temp['months_to_expiry'] <= 6]
            
            if not expiring_contracts.empty:
                return f"You have {len(expiring_contracts)} contracts expiring in the next 6 months: {', '.join(expiring_contracts['counterparty'].unique())}. Total value at risk: ${expiring_contracts['annual_commitment_usd'].sum():,.0f}."
            else:
                return "No contracts are expiring in the next 6 months."
        
        elif "ibm" in prompt_lower:
            ibm_contracts = df_clean[df_clean['counterparty'].str.contains('IBM', case=False, na=False)]
            if not ibm_contracts.empty:
                total_value = ibm_contracts['total_value_usd'].sum()
                total_annual = ibm_contracts['annual_commitment_usd'].sum()
                response = f"IBM Contracts Overview:\n"
                response += f"• Contract Count: {len(ibm_contracts)}\n"
                response += f"• Total Value: ${total_value:,.0f}\n"
                response += f"• Annual Commitment: ${total_annual:,.0f}\n"
                response += f"• Portfolio Share: {(total_value / df_clean['total_value_usd'].sum()) * 100:.1f}%\n\n"
                response += f"Contract Timeline:\n"
                for _, contract in ibm_contracts.iterrows():
                    response += f"• {contract['contract_id']}: {contract['start_date']} to {contract['end_date']} (${contract['annual_commitment_usd']:,.0f}/yr) - {contract.get('status', 'N/A')}\n"
                return response
            else:
                return "No IBM contracts found in the portfolio."
        
        elif "payment" in prompt_lower and "terms" in prompt_lower:
            unfavorable_terms = df_clean[df_clean['payment_terms'].isin(['Net 60', 'Net 90', 'Milestone-based'])]
            if not unfavorable_terms.empty:
                return f"Found {len(unfavorable_terms)} contracts with unfavorable payment terms (Net 60, Net 90, or Milestone-based). Total annual commitment at risk: ${unfavorable_terms['annual_commitment_usd'].sum():,.0f}."
            else:
                return "Great news! All your contracts have favorable payment terms."
        
        elif "penalty" in prompt_lower:
            penalty_contracts = df_clean[df_clean['sla_penalty'].notna() & (df_clean['sla_penalty'] != '')]
            return f"You have {len(penalty_contracts)} contracts with SLA penalty clauses, representing ${penalty_contracts['total_value_usd'].sum():,.0f} in total contract value at risk."
        
        else:
            # Try to provide a general answer based on the prompt keywords
            response_parts = []
            response_parts.append(f"Based on analysis of your {len(df_clean)} contracts:\n")
            
            # Check for vendor mentions
            for vendor in ['IBM', 'Microsoft', 'Oracle', 'SAP', 'Salesforce', 'HCL', 'Infosys', 'Alpha']:
                if vendor.lower() in prompt_lower:
                    vendor_contracts = df_clean[df_clean['counterparty'].str.contains(vendor, case=False, na=False)]
                    if not vendor_contracts.empty:
                        response_parts.append(f"• {vendor} Contracts: {len(vendor_contracts)} contracts, ${vendor_contracts['total_value_usd'].sum():,.0f} total value")
            
            # Check for contract-related keywords
            if any(word in prompt_lower for word in ['contract', 'agreement', 'deal']):
                total_value = df_clean['total_value_usd'].sum()
                response_parts.append(f"• Total Portfolio Value: ${total_value:,.0f}")
                response_parts.append(f"• Active Contracts: {len(df_clean[df_clean['status'].str.contains('Active', na=False)])}")
            
            if response_parts:
                return "\n".join(response_parts)
            else:
                return f"Based on analysis of your {len(df_clean)} contracts, I can provide insights about payment terms, renewals, vendor relationships, SLA requirements, and financial metrics. Please ask a more specific question or mention a vendor name, contract type, or metric you're interested in."
    
    except Exception as e:
        return f"Error in simple analysis: {str(e)}. Please try a different question."


def load_contract_insights():
    """Load contract extraction insights from JSONL file"""
    try:
        # Check parent directory where the file is located
        insights_file = "../cfo_contract_insights.jsonl"
        
        if os.path.exists(insights_file):
            insights = []
            with open(insights_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        insights.append(json.loads(line))
            return insights
        return []
    except Exception as e:
        return []
