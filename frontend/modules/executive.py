"""
Executive Insights Page
Comprehensive CFO insights and analytics
"""

import streamlit as st
import pandas as pd
from typing import Dict, List

def create_page(analytics_data: Dict):
    """Create Executive Insights page"""
    cfo_insights = analytics_data.get("cfo_insights", [])
    
    st.markdown("## 📊 Executive Insights")
    
    # Add refresh button to clear cache and reload
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Insights", type="secondary"):
            # Clear cache for insights loading
            # Import from frontend/utils.py (ensure frontend directory is in path first)
            import sys
            import os
            frontend_dir = os.path.dirname(os.path.dirname(__file__))
            if frontend_dir not in sys.path:
                sys.path.insert(0, frontend_dir)
            from utils import load_cfo_jsonl_insights, load_cfo_analytics_data
            load_cfo_jsonl_insights.clear()
            load_cfo_analytics_data.clear()
            st.success("Cache cleared! Reloading insights...")
            st.rerun()
    
    # Debug info (can be removed later)
    with st.expander("🔧 Debug Info", expanded=False):
        st.write(f"CFO Insights Count: {len(cfo_insights) if cfo_insights else 0}")
        st.write(f"Analytics Data Keys: {list(analytics_data.keys())}")
        if cfo_insights:
            st.write(f"First Insight Sample: {cfo_insights[0] if len(cfo_insights) > 0 else 'N/A'}")
        else:
            # Check if file exists
            import os
            jsonl_files = [
                "../cfo_contract_insights.jsonl",
                "cfo_contract_insights.jsonl",
                os.path.join(os.path.dirname(__file__), "..", "..", "cfo_contract_insights.jsonl")
            ]
            found_files = [f for f in jsonl_files if os.path.exists(f)]
            st.write(f"JSONL File Locations Checked: {jsonl_files}")
            st.write(f"Files Found: {found_files}")
            if found_files:
                st.write(f"✅ File exists but not loaded - cache may need clearing")
    
    # Display insights by dimension
    if cfo_insights:
        dimensions = {}
        for insight in cfo_insights:
            dim = insight['dimension']
            if dim not in dimensions:
                dimensions[dim] = []
            dimensions[dim].append(insight)
        
        for dimension, insights in dimensions.items():
            with st.expander(f"📁 {dimension} ({len(insights)} insights)", expanded=True):
                for insight in insights:
                    formatted_text = format_insights_clean_text(insight)
                    st.markdown(formatted_text)
                    st.markdown("---")
    else:
        st.info("No CFO insights available. Please run GraphRAG processing to generate insights.")

def format_insights_clean_text(insight_data):
    """Format insight data into clean, readable text"""
    
    formatted_text = f"""
    **Question:** {insight_data.get('question', 'N/A')}
    
    **Insight:** {insight_data.get('insight', 'N/A')}
    """
    
    # Add KPIs if available
    if insight_data.get('kpis'):
        kpi_text = "**Key Metrics:** "
        kpi_items = []
        for key, value in insight_data['kpis'].items():
            if isinstance(value, (int, float)):
                kpi_items.append(f"{key}: {value:,.0f}")
            else:
                kpi_items.append(f"{key}: {value}")
        formatted_text += f"\n{kpi_text}{', '.join(kpi_items)}\n"
    
    # Add risks if available
    if insight_data.get('risks'):
        risk_text = "**Risks:** "
        formatted_text += f"\n{risk_text}{', '.join(insight_data['risks'])}\n"
    
    # Add opportunities if available
    if insight_data.get('opportunities'):
        opp_text = "**Opportunities:** "
        formatted_text += f"\n{opp_text}{', '.join(insight_data['opportunities'])}\n"
    
    # Add evidence if available
    if insight_data.get('evidence'):
        formatted_text += "\n**Supporting Evidence:**\n"
        for evidence in insight_data['evidence']:
            formatted_text += f"- {evidence.get('snippet', 'N/A')} (Source: {evidence.get('source', 'N/A')})\n"
    
    # Add data gaps if available
    if insight_data.get('data_gaps'):
        gap_text = "**Data Gaps:** "
        formatted_text += f"\n{gap_text}{', '.join(insight_data['data_gaps'])}\n"
    
    # Add confidence score
    if insight_data.get('confidence'):
        formatted_text += f"\n**Confidence:** {insight_data['confidence']:.1%}\n"
    
    return formatted_text



