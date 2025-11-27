#!/usr/bin/env python3
"""
Quick test to verify dashboard expander fix
"""

import streamlit as st
import json

# Test data
test_insights = [
    {
        "dimension": "Financial Exposure & Obligations",
        "question": "What is the total value of active contracts?",
        "insight": "Total value is unknown",
        "kpis": {"total_value": "Unknown"},
        "risks": ["Lack of visibility"],
        "opportunities": ["Implement tracking"],
        "confidence": 0.2
    },
    {
        "dimension": "Revenue & Profitability",
        "question": "Which contracts drive profitability?",
        "insight": "Profitability drivers unclear",
        "kpis": {"profit_margin": "Unknown"},
        "risks": ["Revenue uncertainty"],
        "opportunities": ["Optimize pricing"],
        "confidence": 0.3
    }
]

def test_expander_structure():
    """Test the expander structure without nesting"""
    
    st.title("Test Dashboard Expander Fix")
    
    # Group insights by dimension
    dimensions = {}
    for insight in test_insights:
        dim = insight.get('dimension', 'Unknown')
        if dim not in dimensions:
            dimensions[dim] = []
        dimensions[dim].append(insight)
    
    # Display each dimension separately (NO NESTED EXPANDERS)
    st.markdown(f"## 📋 CFO Insights ({len(test_insights)} total)")
    
    for dimension, insights_in_dim in dimensions.items():
        st.markdown(f"### 📁 {dimension} ({len(insights_in_dim)} insights)")
        
        for i, insight in enumerate(insights_in_dim):
            with st.expander(f"💡 {insight.get('question', 'Question not specified')} [#{i+1}]"):
                st.markdown(f"""
                **Insight:** {insight.get('insight', 'No insight available')}
                
                **📊 KPIs:** {insight.get('kpis', {})}
                
                **⚠️ Risks:** {', '.join(insight.get('risks', [])) if insight.get('risks') else 'None identified'}
                
                **💰 Opportunities:** {', '.join(insight.get('opportunities', [])) if insight.get('opportunities') else 'None identified'}
                
                **🎯 Confidence:** {insight.get('confidence', 0):.0%}
                """)
            
            if i < len(insights_in_dim) - 1:
                st.markdown("---")

if __name__ == "__main__":
    test_expander_structure()
