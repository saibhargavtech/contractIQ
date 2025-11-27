"""
ContractIQ Utils
Shared utility functions for data loading and processing
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, List, Any
from datetime import datetime

# Import our CFO analytics modules
import sys
import os
# Add parent directory to path (works for both local and Streamlit Cloud)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import config, if not available create a fallback
try:
    import config
except ImportError:
    # Create a fallback config for deployment
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
    import cfo_analytics
except ImportError:
    # Create a fallback cfo_analytics for deployment
    class CFOAnalytics:
        @staticmethod
        def generate_cfo_insights_from_context(context):
            return []
        
        @staticmethod
        def assess_contract_risks(graph, partition):
            return {}
        
        @staticmethod
        def extract_financial_metrics_from_graph(graph):
            return {}
        
        @staticmethod
        def identify_cost_optimization_opportunities(graph):
            return []
        
        @staticmethod
        def generate_executive_summary(portfolio, risks, metrics, opportunities):
            return {}
    cfo_analytics = CFOAnalytics()

@st.cache_data
def load_demo_contracts():
    """Load demo/sample contracts for presentation"""
    try:
        if os.path.exists("dummy_contracts_50.csv"):
            demo_df = pd.read_csv("dummy_contracts_50.csv")
            return demo_df
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_uploaded_contracts():
    """Load uploaded contracts from Gradio processing"""
    try:
        # Check multiple locations for uploaded contracts
        contract_files = [
            "uploaded_contracts.csv",  # Current directory
            "../uploaded_contracts.csv",  # Parent directory
            os.path.join(os.path.dirname(__file__), "..", "uploaded_contracts.csv"),  # Absolute path
            os.path.join(os.path.dirname(__file__), "uploaded_contracts.csv")  # Frontend directory
        ]
        for contract_file in contract_files:
            if os.path.exists(contract_file):
                uploaded_df = pd.read_csv(contract_file)
                return uploaded_df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_unified_contracts():
    """Load unified contracts (dummy + uploaded combined)"""
    try:
        # Check multiple locations for uploaded contracts first
        contract_files = [
            "uploaded_contracts.csv",
            "../uploaded_contracts.csv",
            os.path.join(os.path.dirname(__file__), "..", "uploaded_contracts.csv"),
            os.path.join(os.path.dirname(__file__), "uploaded_contracts.csv")
        ]
        for contract_file in contract_files:
            if os.path.exists(contract_file):
                unified_df = pd.read_csv(contract_file)
                return unified_df
        
        # Fallback to dummy contracts
        demo_files = [
            "dummy_contracts_50.csv",
            "../dummy_contracts_50.csv",
            os.path.join(os.path.dirname(__file__), "..", "dummy_contracts_50.csv"),
            os.path.join(os.path.dirname(__file__), "dummy_contracts_50.csv")
        ]
        for demo_file in demo_files:
            if os.path.exists(demo_file):
                dummy_df = pd.read_csv(demo_file)
                return dummy_df
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60)  # Cache for 1 minute to allow faster refresh after uploads
def load_cfo_jsonl_insights():
    """Load CFO insights from JSONL file - checks multiple locations"""
    try:
        # Check multiple possible locations (similar to graph context memory)
        jsonl_files = [
            "../cfo_contract_insights.jsonl",  # Parent directory (from frontend/)
            "cfo_contract_insights.jsonl",      # Current directory
            os.path.join(os.path.dirname(__file__), "..", "cfo_contract_insights.jsonl"),  # Parent directory (absolute)
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfo_contract_insights.jsonl")  # Root directory
        ]
        
        for jsonl_file in jsonl_files:
            if os.path.exists(jsonl_file):
                insights = []
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                insights.append(json.loads(line))
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
                if insights:
                    print(f"[Utils] Loaded {len(insights)} CFO insights from {jsonl_file}")
                    return insights
        
        print(f"[Utils] CFO insights file not found in any location")
        return []
    except Exception as e:
        print(f"[Utils] Error loading CFO insights: {e}")
        return []

def generate_basic_insights_disabled(df: pd.DataFrame) -> Dict:
    """Generate basic insights when GraphRAG is disabled"""
    return {
        "cfo_insights": [],
        "risk_assessment": {},
        "executive_summary": {}
    }

@st.cache_data(ttl=60)  # Cache for 1 minute to allow faster refresh after uploads
def load_cfo_analytics_data(mode="unified"):
    """Load comprehensive CFO analytics data with consistent caching"""
    analytics_data = {
        "contract_csv": None,
        "cfo_insights": [],
        "risk_assessment": {},
        "executive_summary": {}
    }
    
    # Unified loading logic - always load combined data
    if mode == "unified":
        df = load_unified_contracts()
        if not df.empty:
            analytics_data["contract_csv"] = df
            
            # Use GraphRAG insights if available
            jsonl_insights = load_cfo_jsonl_insights()
            
            if jsonl_insights:
                # Use GraphRAG-generated insights
                analytics_data["cfo_insights"] = jsonl_insights
                analytics_data["risk_assessment"] = generate_risk_assessment(df)
                analytics_data["executive_summary"] = generate_executive_summary(df, jsonl_insights)
            else:
                # Fallback to basic insights
                analytics_data.update(generate_basic_insights_disabled(df))
        else:
            return analytics_data
            
    elif mode == "demo":
        df = load_demo_contracts()
        if not df.empty:
            analytics_data["contract_csv"] = df
            analytics_data.update(generate_basic_insights_disabled(df))
        else:
            return analytics_data
            
    elif mode == "live":
        df = load_uploaded_contracts()
        if not df.empty:
            analytics_data["contract_csv"] = df
            
            # Try to load JSONL insights first (from GraphRAG)
            jsonl_insights = load_cfo_jsonl_insights()
            
            if jsonl_insights:
                # Use JSONL insights (from GraphRAG export)
                analytics_data["cfo_insights"] = jsonl_insights
                analytics_data["risk_assessment"] = generate_risk_assessment(df)
                analytics_data["executive_summary"] = generate_executive_summary(df, jsonl_insights)
            else:
                # Fallback to basic insights
                analytics_data.update(generate_basic_insights_disabled(df))
        else:
            return analytics_data
    
    return analytics_data

def generate_risk_assessment(df: pd.DataFrame) -> Dict:
    """Generate risk assessment from contract data"""
    if df.empty:
        return {}
    
    # Calculate risk metrics
    total_contracts = len(df)
    active_contracts = len(df[df['status'].str.contains('Active', na=False)])
    
    # Vendor concentration risk
    vendor_concentration = df.groupby('counterparty')['total_value_usd'].sum()
    top_vendor_pct = vendor_concentration.max() / vendor_concentration.sum() * 100
    
    # Expiration risk
    df_temp = df.copy()
    df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
    df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
    expiring_soon = len(df_temp[df_temp['months_to_expiry'] <= 6])
    
    return {
        "total_contracts": total_contracts,
        "active_contracts": active_contracts,
        "vendor_concentration_risk": top_vendor_pct,
        "expiring_soon": expiring_soon
    }

def generate_executive_summary(df: pd.DataFrame, insights: List) -> Dict:
    """Generate executive summary from contract data and insights"""
    if df.empty:
        return {}
    
    total_value = df['total_value_usd'].sum()
    annual_commitment = df['annual_commitment_usd'].sum()
    
    return {
        "total_contract_value": int(total_value) if pd.notna(total_value) else 0,
        "annual_commitment": int(annual_commitment) if pd.notna(annual_commitment) else 0
    }

def build_contract_knowledge_graph(df: pd.DataFrame) -> bool:
    """Build knowledge graph from contract DataFrame"""
    try:
        import graph
        import config
        
        # Convert contract data to text format
        contract_texts = []
        for _, row in df.iterrows():
            contract_text = f"""
            Contract ID: {row['contract_id']}
            Type: {row['type']}
            Counterparty: {row['counterparty']}
            Start Date: {row['start_date']}
            End Date: {row['end_date']}
            Status: {row['status']}
            Total Value: ${row['total_value_usd']:,.0f}
            Annual Commitment: ${row['annual_commitment_usd']:,.0f}
            Escalation: {row['escalation']}
            Payment Terms: {row['payment_terms']}
            Variable Costs: {row['variable_costs']}
            SLA Uptime: {row['sla_uptime']}
            SLA Response Critical: {row['sla_response_critical']}
            SLA Resolution Critical: {row['sla_resolution_critical']}
            SLA Penalty: {row['sla_penalty']}
            Compliance: {row['compliance']}
            ESG Reporting: {row['esg_reporting']}
            Termination: {row['termination']}
            Governing Law: {row['governing_law']}
            Arbitration: {row['arbitration']}
            Summary: {row.get('summary', 'N/A')}
            """
            contract_texts.append(contract_text)
        
        # Combine all contract texts
        combined_text = "\n\n".join(contract_texts)
        
        # Initialize and build graph
        G = graph.initialize_graph()
        graph.process_contracts_to_graph(combined_text, G)
        
        # Generate context memory
        context_memory = graph.generate_graph_context_memory(G)
        
        # Store in global config
        config.G = G
        config.GRAPH_CONTEXT_MEMORY = context_memory
        
        return True
        
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        return False
