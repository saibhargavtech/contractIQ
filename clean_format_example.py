#!/usr/bin/env python3
"""
Example of clean CFO insights formatting
"""

def format_insights_clean_text(insight_data):
    """Convert JSON insights to clean, readable text format"""
    
    if not insight_data:
        return "No insights available"
    
    formatted_text = ""
    
    # Question
    formatted_text += f"**Question:** {insight_data.get('question', 'Not specified')}\n\n"
    
    # Insight
    formatted_text += f"**Insight:** {insight_data.get('insight', 'No insight available')}\n\n"
    
    # KPIs as bullet points
    kpis = insight_data.get('kpis', {})
    if kpis:
        formatted_text += "**Key Metrics:**\n"
        for key, value in kpis.items():
            clean_key = key.replace('_', ' ').title()
            formatted_text += f"• {clean_key}: {value}"\n"
        formatted_text += "\n"
    
    # Risks as bullet points
    risks = insight_data.get('risks', [])
    if risks:
        formatted_text += "**Risks Identified:**\n"
        for risk in risks:
            formatted_text += f"• {risk}"\n"
        formatted_text += "\n"
    
    # Opportunities as bullet points
    opportunities = insight_data.get('opportunities', [])
    if opportunities:
        formatted_text += "**Opportunities:**\n"
        for opp in opportunities:
            formatted_text += f"• {opp}"\n"
        formatted_text += "\n"
    
    # Confidence
    confidence = insight_data.get('confidence', 0)
    formatted_text += f"**Confidence Level:** {confidence:.0%}"
    
    return formatted_text

# Example insight
example_insight = {
    "dimension": "Financial Exposure & Obligations",
    "question": "What is the total value of active contracts by business unit and geography?",
    "insight": "Total active contract value is $15.2M across Technology, Operations, and Sales divisions with major concentration in North America and Asia-Pacific regions.",
    "kpis": {
        "total_contract_value": "$15.2M",
        "active_contracts": "47",
        "average_contract_value": "$324K",
        "geographic_spread": "5 regions"
    },
    "risks": [
        "Concentration risk in North America (70% of value)",
        "Multiple medium-value contracts increase management complexity"
    ],
    "opportunities": [
        "Consolidate smaller contracts for better pricing",
        "Negotiate volume discounts with top vendors"
    ],
    "confidence": 0.85
}

# Show clean formatted output
formatted = format_insights_clean_text(example_insight)
print(formatted)
