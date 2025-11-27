"""
Enhanced CFO Dashboard with Full Analytics Integration
Integrates the 30 CFO questions with Streamlit visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from typing import Dict, List, Any

# Import our CFO analytics modules
import sys
sys.path.append('..')
import config
import utils
import extraction
import graph
import clustering
import cfo_analytics
import export

# Set page config
st.set_page_config(
    page_title="Enhanced CFO Contract Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ContractIQ styling
st.markdown("""
<style>
    /* Reduce top spacing */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* ContractIQ header styling */
    h1 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .cfo-metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .risk-high { 
        color: #ff6b6b; 
        font-weight: bold;
    }
    .risk-medium { 
        color: #feca57; 
        font-weight: bold;
    }
    .risk-low { 
        color: #48dbfb; 
        font-weight: bold;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_demo_contracts():
    """Load demo/sample contracts for presentation"""
    try:
        if os.path.exists("../dummy_contracts_50.csv"):
            demo_df = pd.read_csv("../dummy_contracts_50.csv")
            st.info(f"📁 Demo Mode: Loaded {len(demo_df)} sample contracts")
            return demo_df
        else:
            st.warning("No demo contracts found")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Could not load demo contracts: {e}")
        return pd.DataFrame()

def load_uploaded_contracts():
    """Load uploaded contracts from Gradio processing"""
    try:
        if os.path.exists("../uploaded_contracts.csv"):
            uploaded_df = pd.read_csv("../uploaded_contracts.csv")
            return uploaded_df
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def load_unified_contracts():
            """Load unified contracts (dummy + uploaded combined)"""
            try:
                # Always try uploaded_contracts.csv first (contains dummy + uploaded)
                if os.path.exists("../uploaded_contracts.csv"):
                    unified_df = pd.read_csv("../uploaded_contracts.csv")
                    if not unified_df.empty:
                        return unified_df
                
                # Fallback to dummy contracts if no combined file exists
                elif os.path.exists("../dummy_contracts_50.csv"):
                    dummy_df = pd.read_csv("../dummy_contracts_50.csv")
                    if not dummy_df.empty:
                        st.info("📊 Fallback: Using demo contracts")
                        return dummy_df
                
                st.info("📊 No contract data found")
                return pd.DataFrame()
                
            except Exception as e:
                st.error(f"Could not load contract data: {e}")
                return pd.DataFrame()

def load_cfo_jsonl_insights(jsonl_path: str = "../cfo_contract_insights.jsonl"):
    """Load CFO insights from JSONL file and parse properly"""
    insights = []
    try:
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            insight = json.loads(line)
                            insights.append(insight)
                        except json.JSONDecodeError as e:
                            print(f"[Dashboard] Error parsing JSON line: {e}")
                            continue
            print(f"[Dashboard] Loaded {len(insights)} insights from {jsonl_path}")
        else:
            print(f"[Dashboard] No JSONL file found at {jsonl_path}")
    except Exception as e:
        print(f"[Dashboard] Error loading JSONL insights: {e}")
    
    return insights

def extract_risk_assessment_from_jsonl(insights):
    """Extract risk assessment from JSONL insights"""
    all_risks = []
    for insight in insights:
        risks = insight.get('risks', [])
        if risks:
            all_risks.extend(risks)
    
    # Count risk categories
    risk_counts = {}
    for risk in all_risks:
        # Simple risk categorization
        if 'compliance' in risk.lower():
            risk_counts['compliance_risk'] = risk_counts.get('compliance_risk', 0) + 1
        elif 'financial' in risk.lower() or 'cost' in risk.lower():
            risk_counts['financial_risk'] = risk_counts.get('financial_risk', 0) + 1
        elif 'vendor' in risk.lower():
            risk_counts['vendor_risk'] = risk_counts.get('vendor_risk', 0) + 1
        else:
            risk_counts['other_risk'] = risk_counts.get('other_risk', 0) + 1
    
    return {
        "total_risks": len(all_risks),
        "risk_categories": risk_counts,
        "top_risks": all_risks[:5] if all_risks else []
    }

def extract_financial_metrics_from_jsonl(insights):
    """Extract financial metrics from JSONL insights"""
    total_contracts = len([i for i in insights if 'contract' in i.get('question', '').lower()])
    
    # Look for KPIs in insights
    kpi_data = {}
    for insight in insights:
        kpis = insight.get('kpis', {})
        if kpis:
            kpi_data.update(kpis)
    
    return {
        "total_contracts": total_contracts,
        "insights_generated": len(insights),
        "kpis_extracted": kpi_data
    }

def generate_executive_summary_from_jsonl(insights):
    """Generate executive summary from JSONL insights"""
    high_confidence = [i for i in insights if i.get('confidence', 0) > 0.7]
    total_risks = sum(len(i.get('risks', [])) for i in insights)
    total_opportunities = sum(len(i.get('opportunities', [])) for i in insights)
    
    return {
        "portfolio_overview": f"Analysis completed for {len(insights)} CFO questions",
        "key_findings": f"{len(high_confidence)} high-confidence insights generated",
        "risk_summary": f"{total_risks} potential risks identified across all contracts",
        "opportunities": f"{total_opportunities} opportunities identified for optimization",
        "recommended_actions": "Review high-risk contracts, explore optimization opportunities"
    }

def format_kpis_for_display(kpis_dict):
    """Format KPIs for clean display instead of raw JSON"""
    if not kpis_dict:
        return "No KPIs available"
    
    formatted_items = []
    for key, value in kpis_dict.items():
        # Clean up key names
        clean_key = key.replace('_', ' ').title()
        formatted_items.append(f"• **{clean_key}:** {value}")
    
    return "\n".join(formatted_items)

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
            formatted_text += f"• {clean_key}: {value}\n"
        formatted_text += "\n"
    
    # Risks as bullet points
    risks = insight_data.get('risks', [])
    if risks:
        formatted_text += "**Risks Identified:**\n"
        for risk in risks:
            formatted_text += f"• {risk}\n"
        formatted_text += "\n"
    
    # Opportunities as bullet points
    opportunities = insight_data.get('opportunities', [])
    if opportunities:
        formatted_text += "**Opportunities:**\n"
        for opp in opportunities:
            formatted_text += f"• {opp}\n"
        formatted_text += "\n"
    
    # Confidence
    confidence = insight_data.get('confidence', 0)
    formatted_text += f"**Confidence Level:** {confidence:.0%}"
    
    return formatted_text

def load_cfo_analytics_data(mode="demo"):
    """Load comprehensive CFO analytics data with dual-mode support"""
    analytics_data = {
        "contract_csv": None,
        "cfo_insights": [],
        "risk_assessment": {},
        "financial_metrics": {},
        "executive_summary": {},
        "mode": mode
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
                analytics_data["risk_assessment"] = extract_risk_assessment_from_jsonl(jsonl_insights)
                analytics_data["financial_metrics"] = extract_financial_metrics_from_jsonl(jsonl_insights)
                analytics_data["executive_summary"] = generate_executive_summary_from_jsonl(jsonl_insights)
            else:
                # Fallback to basic insights
                try:
                    analytics_data.update(generate_fresh_cfo_insights(df))
                    st.info("📝 Using basic insights (GraphRAG processing recommended)")
                except Exception as e:
                    st.warning(f"Analysis failed: {e}")
                    analytics_data.update(generate_fresh_cfo_insights(df))
        else:
            st.error("❌ No contracts found")
            st.info("📤 Please upload documents in Gradio first")
            return analytics_data
    elif mode == "demo":
        df = load_demo_contracts()
        if not df.empty:
            analytics_data["contract_csv"] = df
            analytics_data.update(generate_basic_insights_disabled(df))
        else:
            st.error("❌ No demo contracts found")
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
                analytics_data["risk_assessment"] = extract_risk_assessment_from_jsonl(jsonl_insights)
                analytics_data["financial_metrics"] = extract_financial_metrics_from_jsonl(jsonl_insights)
                analytics_data["executive_summary"] = generate_executive_summary_from_jsonl(jsonl_insights)
            else:
                # Fallback to basic insights from CSV data
                try:
                    analytics_data.update(generate_fresh_cfo_insights(df))
                    st.warning("📝 Using basic insights (GraphRAG data not available)")
                except Exception as e:
                    st.warning(f"Analysis failed: {e}")
                    analytics_data.update(generate_fresh_cfo_insights(df))
        else:
            st.error("❌ No uploaded contracts found")
            st.info("📤 Please upload documents in Gradio first")
            return analytics_data
    
    
    return analytics_data

def generate_fresh_cfo_insights(df: pd.DataFrame) -> Dict:
    """Generate CFO insights using the GraphRAG system"""
    try:
        # Simulate building a graph from contract data
        G = create_graph_from_contracts(df)
        
        # Detect communities
        G, partition = graph.detect_communities(G)
        
        # Build context memory
        summaries = clustering.summarize_clusters(G, partition, df.to_string())
        context_memory = clustering.build_graph_context_memory(G, partition, summaries)
        
        # Generate CFO insights
        cfo_insights = cfo_analytics.generate_cfo_insights_from_context(context_memory)
        
        # Calculate risk assessment
        risk_assessment = cfo_analytics.assess_contract_risks(G, partition)
        
        # Extract financial metrics
        financial_metrics = cfo_analytics.extract_financial_metrics_from_graph(G)
        
        # Generate executive summary
        opportunities = cfo_analytics.identify_cost_optimization_opportunities(G)
        exec_summary = cfo_analytics.generate_executive_summary(
            {"total_contracts": len(df)}, risk_assessment, financial_metrics, opportunities
        )
        
        return {
            "cfo_insights": cfo_insights,
            "risk_assessment": risk_assessment,
            "financial_metrics": financial_metrics,
            "executive_summary": exec_summary
        }
        
    except Exception as e:
        st.warning(f"GraphRAG analysis failed: {e}")
        st.error("❌ GraphRAG analysis failed. Please upload and process documents in Gradio first.")
        return analytics_data

def create_graph_from_contracts(df: pd.DataFrame):
    """Create a simple graph from contract data for analytics"""
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add nodes for contracts, vendors, compliance standards
    for _, contract in df.iterrows():
        contract_id = contract['contract_id']
        counterparty = contract['counterparty']
        
        # Add contract node
        G.add_node(contract_id, 
                  type='contract',
                  counterparty=counterparty,
                  value=contract['total_value_usd'],
                  status=contract['status'])
        
        # Add vendor node
        G.add_node(counterparty, type='vendor')
        
        # Relationship: contract involves vendor
        G.add_edge(contract_id, counterparty, 
                  relation='involves_counterparty',
                  evidence=f"Contract {contract_id} involves {counterparty}")
        
        # Add compliance standards as nodes and relationships
        compliance_standards = parse_compliance_standards(contract['compliance'])
        
        if compliance_standards:
            for standard in compliance_standards:
                G.add_node(standard, type='compliance_standard')
                G.add_edge(contract_id, standard,
                          relation='recognised_under',
                          evidence=f"Contract governed by {standard}")
        
        # Payment terms relationship
        payment_term = contract['payment_terms']
        G.add_edge(contract_id, payment_term,
                  relation='governed_by',
                  evidence=f"Payment terms: {payment_term}")
    
    return G

# Basic insights function removed - only using GraphRAG data from uploaded documents
def generate_basic_insights_disabled(df: pd.DataFrame) -> Dict:
    """Generate basic CFO insights from CSV data only"""
    
    # Calculate basic KPIs
    total_value = df['total_value_usd'].sum()
    annual_commitment = df['annual_commitment_usd'].sum()
    active_contracts = len(df[df['status'] == 'Active'])
    
    # Vendor analysis
    vendor_spend = df.groupby('counterparty')['total_value_usd'].sum().sort_values(ascending=False)
    top_vendor = vendor_spend.index[0] if len(vendor_spend) > 0 else "Unknown"
    top_vendor_percentage = df[df['counterparty'] == top_vendor]['total_value_usd'].sum() / total_value * 100 if total_value > 0 else 0
    
    # Payment terms analysis
    payment_terms = df['payment_terms'].value_counts()
    avg_payment_days = 45  # Default estimate
    
    # Risk analysis
    high_sla_risk = len(df[df['sla_uptime'].str.contains('99.5', na=False)])
    expiring_soon = len(df[(pd.to_datetime(df['end_date']) - datetime.now()).dt.days < 180])
    
    # Create structured insights for all 30 CFO questions
    cfo_insights = []
    
    # Import the CFO questions from config
    import sys
    sys.path.append('..')
    import config
    
    for dimension, question in config.CFO_DIMENSIONS_AND_QUESTIONS:
        # Generate basic insights based on available data
        if "total value" in question.lower():
            insight = f"Total Contract Value: ${total_value:,.0f}. Top vendor: {top_vendor} ({top_vendor_percentage:.1f}%)"
            kpis = {
                "total_contract_value": int(total_value) if pd.notna(total_value) else 0,
                "top_vendor": str(top_vendor),
                "top_vendor_percentage": float(top_vendor_percentage) if pd.notna(top_vendor_percentage) else 0.0
            }
            opportunities = ["Negotiate additional discounts with top vendors"]
            confidence = 0.8
        elif "payment terms" in question.lower():
            insight = f"Payment terms distribution: {dict(payment_terms)}. Estimated average: {avg_payment_days} days"
            kpis = {
                "avg_payment_days": int(avg_payment_days),
                "net_30_count": int(len(df[df['payment_terms'].str.contains('Net 30', na=False)]))
            }
            opportunities = ["Optimize payment terms for cash flow"]
            confidence = 0.7
        elif "vendor" in question.lower() or "counterparty" in question.lower():
            insight = f"Top vendor concentration: {top_vendor} represents {top_vendor_percentage:.1f}% of total spend"
            kpis = {
                "top_vendor": str(top_vendor),
                "top_vendor_percentage": float(top_vendor_percentage) if pd.notna(top_vendor_percentage) else 0.0,
                "vendor_count": len(df['counterparty'].unique())
            }
            opportunities = ["Diversify vendor portfolio", "Negotiate better terms with top vendors"]
            confidence = 0.8
        elif "risk" in question.lower() or "compliance" in question.lower():
            insight = f"Risk assessment: {high_sla_risk} contracts with high SLA risk, {expiring_soon} contracts expiring soon"
            kpis = {
                "high_sla_risk_count": high_sla_risk,
                "expiring_soon_count": expiring_soon,
                "total_contracts": len(df)
            }
            opportunities = ["Implement contract review cycle", "Address SLA risks"]
            confidence = 0.7
        else:
            # Generic insight for other questions
            insight = f"Analysis based on {len(df)} contracts with total value ${total_value:,.0f}"
            kpis = {
                "total_contracts": len(df),
                "total_value": int(total_value) if pd.notna(total_value) else 0
            }
            opportunities = ["Conduct detailed contract analysis"]
            confidence = 0.6
        
        cfo_insights.append({
            "dimension": dimension,
            "question": question,
            "insight": insight,
            "kpis": kpis,
            "risks": [],
            "opportunities": opportunities,
            "confidence": confidence
        })
    
    risk_assessment = {
        "concentration_risk": {
            "level": "high" if top_vendor_percentage > 30 else "medium",
            "top_vendor": top_vendor
        },
        "operational_risks": {
            "high_sla_risk_count": high_sla_risk,
            "expiring_soon": expiring_soon
        }
    }
    
    executive_summary = {
        "portfolio_overview": f"Portfolio: {len(df)} contracts valued at ${total_value:,.0f}",
        "key_risks": f"High vendor concentration with {top_vendor} ({top_vendor_percentage:.1f}%), {expiring_soon} contracts expiring soon",
        "opportunities": "Vendor consolidation opportunities identified",
        "recommended_actions": "Implement contract review cycle, negotiate vendor discounts"
    }
    
    return {
        "cfo_insights": cfo_insights,
        "risk_assessment": risk_assessment,
        "executive_summary": executive_summary,
        "financial_metrics": {
            "total_contract_value": int(total_value) if pd.notna(total_value) else 0,
            "annual_commitment": int(annual_commitment) if pd.notna(annual_commitment) else 0
        }
    }


def parse_compliance_standards(compliance_str: str) -> List[str]:
    """Parse compliance standards from comma-separated string"""
    if pd.isna(compliance_str):
        return []
    
    standards = [s.strip() for s in compliance_str.split(',')]
    return [s for s in standards if s]

def create_insightiq_page(analytics_data: Dict):
    """Create InsightIQ page with KPIs, vendor analysis, critical contracts, and expiring timeline"""
    
    df = analytics_data["contract_csv"]
    cfo_insights = analytics_data["cfo_insights"]
    risk_assessment = analytics_data["risk_assessment"]
    executive_summary = analytics_data["executive_summary"]
    
    # Calculate only the 4 critical CFO metrics
    if not df.empty:
        # Cash Flow at Risk (unfavorable payment terms)
        unfavorable_terms = df[df['payment_terms'].isin(['Net 60', 'Net 90', 'Milestone-based'])]
        cash_flow_risk = unfavorable_terms['annual_commitment_usd'].sum()
        
        # Escalation Exposure
        escalation_contracts = df[df['escalation'].notna() & (df['escalation'] != 'None')]
        escalation_exposure = escalation_contracts['annual_commitment_usd'].sum()
        
        # Penalty Risk Exposure
        penalty_contracts = df[df['sla_penalty'].notna() & (df['sla_penalty'] != '')]
        penalty_risk = penalty_contracts['total_value_usd'].sum()
        
        # Critical Renewal Pipeline (active contracts expiring in 6 months)
        df_temp = df.copy()
        df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
        df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
        critical_renewals = len(df_temp[(df_temp['status'].str.contains('Active', na=False)) & (df_temp['months_to_expiry'] <= 6)])
        
    else:
        cash_flow_risk = 0
        escalation_exposure = 0
        penalty_risk = 0
        critical_renewals = 0
    
    # Critical Financial KPIs - pushed upward
    st.markdown("## 🎯 Critical Financial KPIs")
    
    # Single row of 4 Critical CFO KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="cfo-metric-card">
            <div class="metric-value">${cash_flow_risk:,.0f}</div>
            <div class="metric-label">🚨 Cash Flow at Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="cfo-metric-card">
            <div class="metric-value">${escalation_exposure:,.0f}</div>
            <div class="metric-label">📈 Escalation Exposure</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="cfo-metric-card">
            <div class="metric-value">${penalty_risk:,.0f}</div>
            <div class="metric-label">⚠️ Penalty Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="cfo-metric-card">
            <div class="metric-value">{critical_renewals}</div>
            <div class="metric-label">🔄 Critical Renewals</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Vendor Analysis Section
    st.markdown("## 🏢 Vendor Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top vendors by value
        vendor_concentration = df.groupby('counterparty')['total_value_usd'].sum().sort_values(ascending=False).head(10)
        
        fig = px.pie(
            vendor_concentration, 
            values=vendor_concentration.values,
            names=vendor_concentration.index,
            title="Top 10 Vendors by Contract Value"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Vendor concentration risk
        top_vendor_share = vendor_concentration.max()
        total_value = df['total_value_usd'].sum()
        concentration_pct = (top_vendor_share / total_value) * 100
        
        st.metric("Top Vendor Concentration", f"{concentration_pct:.1f}%")
        st.metric("Top Vendor", vendor_concentration.index[0])
        st.metric("Top Vendor Value", f"${top_vendor_share:,.0f}")
    
    st.markdown("---")
    
    # Critical Contracts Section
    st.markdown("## ⚠️ Critical Contracts")
    
    # Show contracts with high risk factors
    critical_contracts = []
    
    # Contracts with unfavorable payment terms
    unfavorable_contracts = df[df['payment_terms'].isin(['Net 60', 'Net 90', 'Milestone-based'])]
    if not unfavorable_contracts.empty:
        critical_contracts.extend(unfavorable_contracts[['contract_id', 'counterparty', 'payment_terms', 'annual_commitment_usd']].to_dict('records'))
    
    # Contracts expiring soon
    expiring_contracts = df_temp[(df_temp['status'].str.contains('Active', na=False)) & (df_temp['months_to_expiry'] <= 6)]
    if not expiring_contracts.empty:
        critical_contracts.extend(expiring_contracts[['contract_id', 'counterparty', 'end_date', 'annual_commitment_usd']].to_dict('records'))
    
    if critical_contracts:
        critical_df = pd.DataFrame(critical_contracts)
        st.dataframe(critical_df, use_container_width=True)
    else:
        st.success("✅ No critical contracts identified")
    
    st.markdown("---")
    
    # Expiring Timeline Visualization
    st.markdown("## 📅 Contract Expiry Timeline")
    
    # Timeline filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        timeline_filter = st.selectbox(
            "Filter Timeline:",
            ["6 Months", "1 Year", "2 Years", "All"],
            index=2
        )
    
    # Create timeline data
    df_timeline = df.copy()
    df_timeline['end_date'] = pd.to_datetime(df_timeline['end_date'])
    df_timeline['months_to_expiry'] = (df_timeline['end_date'] - pd.Timestamp.now()).dt.days / 30
    
    # Apply filter
    if timeline_filter == "6 Months":
        timeline_data = df_timeline[df_timeline['months_to_expiry'] <= 6].copy()
        title_suffix = "Next 6 Months"
    elif timeline_filter == "1 Year":
        timeline_data = df_timeline[df_timeline['months_to_expiry'] <= 12].copy()
        title_suffix = "Next 1 Year"
    elif timeline_filter == "2 Years":
        timeline_data = df_timeline[df_timeline['months_to_expiry'] <= 24].copy()
        title_suffix = "Next 2 Years"
    else:  # All
        timeline_data = df_timeline.copy()
        title_suffix = "All Contracts"
    
    if not timeline_data.empty:
        fig = px.timeline(
            timeline_data,
            x_start="start_date",
            x_end="end_date", 
            y="contract_id",
            color="annual_commitment_usd",
            title=f"Contract Expiry Timeline ({title_suffix})",
            labels={'annual_commitment_usd': 'Annual Value (USD)'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No contracts expiring in the {title_suffix.lower()}")

def create_portfolio_management_page(analytics_data: Dict):
    """Create Portfolio Management page"""
    df = analytics_data["contract_csv"]
    cfo_insights = analytics_data["cfo_insights"]
    
    st.markdown("## 📊 Portfolio Management")
    
    # Move content from strategic planning tab
    create_strategic_planning_tab(df)

def create_cash_flow_analysis_page(analytics_data: Dict):
    """Create Cash Flow Analysis page"""
    df = analytics_data["contract_csv"]
    cfo_insights = analytics_data["cfo_insights"]
    
    st.markdown("## 💰 Cash Flow Analysis")
    
    # Move content from financial analysis tab
    create_financial_analysis_tab(df, cfo_insights)

def create_cost_control_risk_page(analytics_data: Dict):
    """Create Cost Control & Risk Management page"""
    df = analytics_data["contract_csv"]
    cfo_insights = analytics_data["cfo_insights"]
    risk_assessment = analytics_data["risk_assessment"]
    
    st.markdown("## 💸 Cost Control & Risk Management")
    
    # Move content from opportunities tab and risk assessment tab
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💸 Cost Control")
        create_opportunities_tab(cfo_insights)
    
    with col2:
        st.markdown("### 🚨 Risk Management")
        create_risk_assessment_tab(df, risk_assessment)

def create_executive_insights_page(analytics_data: Dict):
    """Create Executive Insights page"""
    cfo_insights = analytics_data["cfo_insights"]
    
    st.markdown("## 📊 Executive Insights")
    
    # Move content from all insights tab
    create_all_insights_tab(cfo_insights)

def create_executive_dashboard(analytics_data: Dict):
    """Create the main executive dashboard"""
    
    df = analytics_data["contract_csv"]
    cfo_insights = analytics_data["cfo_insights"]
    risk_assessment = analytics_data["risk_assessment"]
    executive_summary = analytics_data["executive_summary"]
    
    # Clean ContractIQ Sidebar
    st.sidebar.markdown("## 🎯 Quick Actions")
    if st.sidebar.button("📊 Generate Executive Report"):
        st.session_state['generate_report'] = True
    if st.sidebar.button("🚨 View Critical Issues"):
        st.session_state['view_critical'] = True
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Portfolio Overview")
    total_value = df['total_value_usd'].sum()
    total_annual = df['annual_commitment_usd'].sum()
    active_contracts = len(df[df['status'].str.contains('Active', na=False)])
    
    st.sidebar.metric("💰 Total Portfolio Value", f"${total_value:,.0f}")
    st.sidebar.metric("💸 Annual Commitments", f"${total_annual:,.0f}")
    st.sidebar.metric("⚡ Active Contracts", active_contracts)
    
    # Executive Summary Section - Critical CFO KPIs
    st.markdown("## 🎯 Critical Financial KPIs")
    
    # Calculate only the 4 critical CFO metrics
    if not df.empty:
        # Cash Flow at Risk (unfavorable payment terms)
        unfavorable_terms = df[df['payment_terms'].isin(['Net 60', 'Net 90', 'Milestone-based'])]
        cash_flow_risk = unfavorable_terms['annual_commitment_usd'].sum()
        
        # Escalation Exposure
        escalation_contracts = df[df['escalation'].notna() & (df['escalation'] != 'None')]
        escalation_exposure = escalation_contracts['annual_commitment_usd'].sum()
        
        # Penalty Risk Exposure
        penalty_contracts = df[df['sla_penalty'].notna() & (df['sla_penalty'] != '')]
        penalty_risk = penalty_contracts['total_value_usd'].sum()
        
        # Critical Renewal Pipeline (active contracts expiring in 6 months)
        df_temp = df.copy()
        df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
        df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
        critical_renewals = len(df_temp[(df_temp['status'].str.contains('Active', na=False)) & (df_temp['months_to_expiry'] <= 6)])
        
    else:
        cash_flow_risk = 0
        escalation_exposure = 0
        penalty_risk = 0
        critical_renewals = 0
    
    # Single row of 4 Critical CFO KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="cfo-metric-card">
            <div class="metric-value">${cash_flow_risk:,.0f}</div>
            <div class="metric-label">🚨 Cash Flow at Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="cfo-metric-card">
            <div class="metric-value">${escalation_exposure:,.0f}</div>
            <div class="metric-label">📈 Escalation Exposure</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="cfo-metric-card">
            <div class="metric-value">${penalty_risk:,.0f}</div>
            <div class="metric-label">⚠️ Penalty Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="cfo-metric-card">
            <div class="metric-value">{critical_renewals}</div>
            <div class="metric-label">🔄 Critical Renewals</div>
        </div>
        """, unsafe_allow_html=True)
    
    
    # Executive Analytics Tabs - CFO-Focused
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💰 Cash Flow Impact", "🚨 Risk Management", "💸 Cost Control", "📅 Strategic Planning", "📊 Executive Insights"])
    
    with tab1:
        create_financial_analysis_tab(df, cfo_insights)
    
    with tab2:
        create_risk_assessment_tab(df, risk_assessment)
    
    with tab3:
        create_opportunities_tab(cfo_insights)
    
    with tab4:
        create_strategic_planning_tab(df)
    
    with tab5:
        create_all_insights_tab(cfo_insights)

def create_financial_analysis_tab(df: pd.DataFrame, cfo_insights: List):
    """Create financial analysis tab with CFO-focused cash flow impact"""
    
    # Cash Flow Impact Section
    st.markdown("## 💰 Cash Flow Impact Analysis")
    
    # Key cash flow metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_annual = df['annual_commitment_usd'].sum()
        st.metric("Total Annual Commitments", f"${total_annual:,.0f}")
    
    with col2:
        unfavorable_terms = df[df['payment_terms'].isin(['Net 60', 'Net 90', 'Milestone-based'])]
        unfavorable_value = unfavorable_terms['annual_commitment_usd'].sum()
        st.metric("Unfavorable Payment Terms", f"${unfavorable_value:,.0f}")
    
    with col3:
        # Calculate average payment days
        payment_days = df['payment_terms'].str.extract(r'(\d+)').astype(float)
        avg_payment_days = payment_days.mean().iloc[0] if not payment_days.empty else 0
        st.metric("Average Payment Terms", f"{avg_payment_days:.0f} days")
    
    # Cash flow charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment terms impact
        if not df.empty and 'payment_terms' in df.columns:
            payment_analysis = df.groupby('payment_terms')['annual_commitment_usd'].sum().reset_index()
            if not payment_analysis.empty:
                fig = px.bar(
                    payment_analysis, 
                    x='payment_terms', 
                    y='annual_commitment_usd',
                    title="Annual Commitments by Payment Terms",
                    labels={'annual_commitment_usd': 'Annual Value (USD)', 'payment_terms': 'Payment Terms'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No payment terms data available")
        else:
            st.info("No contract data available for payment terms analysis")
    
    with col2:
        # Contract value distribution
        if not df.empty and 'type' in df.columns:
            type_analysis = df.groupby('type')['total_value_usd'].sum().reset_index()
            if not type_analysis.empty:
                fig1 = px.pie(
                    type_analysis,
                    values='total_value_usd', 
                    names='type',
                    title="Contract Value by Type"
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No contract type data available")
        else:
            st.info("No contract data available for type analysis")
    
    # Vendor spend analysis
    st.markdown("## 🏢 Vendor Analysis")
    if not df.empty and 'counterparty' in df.columns:
        vendor_spend = df.groupby('counterparty')['total_value_usd'].sum().sort_values(ascending=False).head(10)
        if not vendor_spend.empty:
            fig2 = px.bar(
                x=vendor_spend.index, 
                y=vendor_spend.values,
                title="Top 10 Vendors by Spend",
                labels={'x': 'Vendor', 'y': 'Value (USD)'}
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No vendor data available")
    else:
        st.info("No contract data available for vendor analysis")
    
    # Extracted Contract KPIs Summary
    if df is not None and not df.empty:
        st.markdown("### 📊 Contract Portfolio KPIs")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Total Contract Value", value=f"${df['total_value_usd'].sum():,.0f}")
            st.metric(label="Average Contract", value=f"${df['total_value_usd'].mean():,.0f}")
        
        with col2:
            st.metric(label="Active Contracts", value=f"{len(df[df['status'].str.contains('Active', na=False)])}")
            st.metric(label="Annual Commitment", value=f"${df['annual_commitment_usd'].sum():,.0f}")
            
        with col3:
            high_value = len(df[df['total_value_usd'] > 1000000])
            st.metric(label="High Value (>$1M)", value=high_value)
            st.metric(label="Total Vendors", value=df['counterparty'].nunique())
        
        # Contract Distribution Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Contract Values Distribution")
            fig = px.histogram(df, x='total_value_usd', title="Contract Value Distribution", 
                             labels={'total_value_usd': 'Contract Value ($)', 'count': 'Number of Contracts'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Contract Types")
            contract_types = df['type'].value_counts()
            fig = px.pie(values=contract_types.values, names=contract_types.index, 
                        title="Contract Types Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Financial Insights Summary
    for insight in cfo_insights:
        if "Financial" in insight['dimension']:
            st.markdown(f"**{insight['question']}**")
            st.markdown(insight['insight'])
            if insight.get('kpis'):
                st.markdown("**KPIs:**")
                for kpi_name, kpi_value in insight['kpis'].items():
                    st.markdown(f"• {kpi_name.replace('_', ' ').title()}: {kpi_value}")
                st.markdown("---")

def create_risk_assessment_tab(df: pd.DataFrame, risk_assessment: Dict):
    """Create risk assessment tab with CFO-focused risk management"""
    
    st.markdown("## 🚨 Risk Management Dashboard")
    
    # Risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Vendor concentration risk
        top_vendor_share = df.groupby('counterparty')['total_value_usd'].sum().nlargest(1).iloc[0]
        total_value = df['total_value_usd'].sum()
        concentration_pct = (top_vendor_share / total_value) * 100
        st.metric("Top Vendor Concentration", f"{concentration_pct:.1f}%")
    
    with col2:
        # Penalty exposure
        penalty_contracts = df[df['sla_penalty'].notna() & (df['sla_penalty'] != '')]
        penalty_value = penalty_contracts['total_value_usd'].sum()
        st.metric("Penalty Exposure", f"${penalty_value:,.0f}")
    
    with col3:
        # Compliance coverage
        compliance_coverage = df['compliance'].notna().sum() / len(df) * 100
        st.metric("Compliance Coverage", f"{compliance_coverage:.1f}%")
    
    # Risk charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Vendor concentration risk
        vendor_concentration = df.groupby('counterparty')['total_value_usd'].sum().sort_values(ascending=False).head(10)
        
        fig = px.pie(
            vendor_concentration, 
            values=vendor_concentration.values,
            names=vendor_concentration.index,
            title="Top 10 Vendors by Contract Value (Concentration Risk)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Compliance coverage
        compliance_standards = ['SOC2', 'GDPR', 'CCPA', 'ISO 27001', 'HIPAA', 'FCPA', 'PCI DSS']
        compliance_data = []
        
        for standard in compliance_standards:
            count = df['compliance'].str.contains(standard, na=False).sum()
            compliance_data.append({'Standard': standard, 'Coverage': count})
        
        compliance_df = pd.DataFrame(compliance_data)
        
        fig = px.bar(
            compliance_df,
            x='Standard',
            y='Coverage',
            title="Compliance Standards Coverage",
            color='Coverage',
            color_continuous_scale='RdYlGn'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk alerts
    st.markdown("## 🚨 Risk Alerts")
    
    # High concentration risk
    high_concentration = df.groupby('counterparty')['total_value_usd'].sum()
    high_concentration_pct = (high_concentration / total_value * 100).sort_values(ascending=False)
    
    if high_concentration_pct.iloc[0] > 20:
        st.warning(f"⚠️ **High Concentration Risk**: {high_concentration_pct.index[0]} represents {high_concentration_pct.iloc[0]:.1f}% of total portfolio value")
    
    # Missing compliance
    missing_compliance = df[df['compliance'].isna() | (df['compliance'] == '')]
    if not missing_compliance.empty:
        st.warning(f"⚠️ **Missing Compliance Data**: {len(missing_compliance)} contracts lack compliance information")
    
    # SLA Risk Distribution
    df['sla_risk'] = df['sla_uptime'].apply(lambda x: 'High Risk' if '99.5' in str(x) else 'Standard')
    
    fig = px.pie(df['sla_risk'].value_counts().reset_index(),
                  values='count', names='sla_risk',
                  title="SLA Risk Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
def create_opportunities_tab(cfo_insights: List):
    """Create opportunities tab with CFO-focused cost control"""
    
    st.markdown("## 💸 Cost Control & Optimization")
    
    # Load contract data for cost analysis
    try:
        df = pd.read_csv("../uploaded_contracts.csv")
        
        # Cost metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_annual = df['annual_commitment_usd'].sum()
            st.metric("Total Annual Commitment", f"${total_annual:,.0f}")
        
        with col2:
            escalation_contracts = df[df['escalation'].notna() & (df['escalation'] != 'None')]
            escalation_value = escalation_contracts['annual_commitment_usd'].sum()
            st.metric("Escalation Exposure", f"${escalation_value:,.0f}")
        
        with col3:
            variable_cost_contracts = df[df['variable_costs'].notna() & (df['variable_costs'] != '')]
            variable_value = variable_cost_contracts['annual_commitment_usd'].sum()
            st.metric("Variable Cost Exposure", f"${variable_value:,.0f}")
        
        # Cost optimization charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Escalation clauses impact
            escalation_data = df[df['escalation'].notna() & (df['escalation'] != 'None')]
            
            if not escalation_data.empty:
                fig = px.bar(
                    escalation_data,
                    x='contract_id',
                    y='annual_commitment_usd',
                    color='escalation',
                    title="Contracts with Escalation Clauses",
                    labels={'annual_commitment_usd': 'Annual Value (USD)', 'contract_id': 'Contract ID'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No contracts with escalation clauses found")
        
        with col2:
            # Cost optimization opportunities
            df_temp = df.copy()
            df_temp['cost_per_year'] = df_temp['total_value_usd'] / ((pd.to_datetime(df_temp['end_date']) - pd.to_datetime(df_temp['start_date'])).dt.days / 365)
            
            fig = px.scatter(
                df_temp,
                x='total_value_usd',
                y='cost_per_year',
                color='type',
                size='annual_commitment_usd',
                hover_data=['counterparty', 'escalation'],
                title="Contract Value vs Annual Cost (Optimization Opportunities)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost optimization recommendations
        st.markdown("## 💡 Cost Optimization Opportunities")
        
        # Long-term contracts analysis
        df_temp = df.copy()
        df_temp['contract_duration'] = (pd.to_datetime(df_temp['end_date']) - pd.to_datetime(df_temp['start_date'])).dt.days / 365
        
        long_term_contracts = df_temp[df_temp['contract_duration'] > 3]
        if not long_term_contracts.empty:
            st.info(f"📊 **Long-term Contracts**: {len(long_term_contracts)} contracts over 3 years - consider volume discounts")
        
        # High-value contracts
        high_value_contracts = df[df['total_value_usd'] > df['total_value_usd'].quantile(0.8)]
        if not high_value_contracts.empty:
            st.info(f"💰 **High-Value Contracts**: {len(high_value_contracts)} contracts in top 20% - negotiate better terms")
    
    except Exception as e:
        st.warning(f"Could not load contract data: {e}")
    
    # CFO Insights Opportunities
    st.markdown("## 🎯 CFO Insights Opportunities")
    
    for insight in cfo_insights:
        if insight.get("opportunities"):
            st.markdown(f"**{insight['dimension']}**")
            for opportunity in insight["opportunities"]:
                st.success(f"🎯 {opportunity}")
    
    # Strategic recommendations
    st.markdown("## 🚀 Strategic Recommendations")
    
    recommendations = [
        "📊 Implement automated contract review cycle",
        "💰 Renegotiate payment terms for cash flow optimization", 
        "🔄 Develop vendor audit framework",
        "📈 Create contract performance dashboards",
        "⚖️ Standardize compliance requirements"
    ]
    
    for rec in recommendations:
        st.info(rec)

def create_strategic_planning_tab(df: pd.DataFrame):
    """Create strategic planning tab with CFO-focused renewal pipeline"""
    
    st.markdown("## 📅 Strategic Planning & Renewal Pipeline")
    
    # Strategic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Contracts expiring in next 12 months
        df_temp = df.copy()
        df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
        df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
        
        expiring_12m = df_temp[df_temp['months_to_expiry'] <= 12]
        expiring_value = expiring_12m['total_value_usd'].sum()
        st.metric("Expiring in 12 Months", f"${expiring_value:,.0f}")
    
    with col2:
        # Average contract duration
        df_temp['contract_duration'] = (df_temp['end_date'] - pd.to_datetime(df_temp['start_date'])).dt.days / 365
        avg_duration = df_temp['contract_duration'].mean()
        st.metric("Average Contract Duration", f"{avg_duration:.1f} years")
    
    with col3:
        # Vendor diversification
        unique_vendors = df['counterparty'].nunique()
        total_contracts = len(df)
        diversification_ratio = unique_vendors / total_contracts
        st.metric("Vendor Diversification", f"{diversification_ratio:.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Renewal pipeline
        df_temp = df.copy()
        df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
        df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
        
        # Create renewal buckets
        renewal_buckets = []
        for _, row in df_temp.iterrows():
            months = row['months_to_expiry']
            if months <= 6:
                bucket = "0-6 months"
            elif months <= 12:
                bucket = "6-12 months"
            elif months <= 24:
                bucket = "1-2 years"
            else:
                bucket = "2+ years"
            
            renewal_buckets.append({
                'Bucket': bucket,
                'Value': row['total_value_usd'],
                'Contract': row['contract_id']
            })
        
        renewal_df = pd.DataFrame(renewal_buckets)
        renewal_summary = renewal_df.groupby('Bucket')['Value'].sum().reset_index()
        
        fig = px.funnel(
            renewal_summary,
            x='Value',
            y='Bucket',
            title="Contract Renewal Pipeline"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Vendor diversification by type
        vendor_diversity = df.groupby('type')['counterparty'].nunique().reset_index()
        vendor_diversity.columns = ['Contract Type', 'Unique Vendors']
        
        fig = px.bar(
            vendor_diversity,
            x='Contract Type',
            y='Unique Vendors',
            title="Vendor Diversification by Contract Type",
            color='Unique Vendors',
            color_continuous_scale='Blues'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategic recommendations
    st.markdown("## 🎯 Strategic Recommendations")
    
    # Renewal urgency
    urgent_renewals = df_temp[df_temp['months_to_expiry'] <= 6]
    if not urgent_renewals.empty:
        st.warning(f"🚨 **Urgent Renewals**: {len(urgent_renewals)} contracts expiring in 6 months - start renewal negotiations")
    
    # Vendor consolidation opportunities
    vendor_counts = df['counterparty'].value_counts()
    multiple_contracts = vendor_counts[vendor_counts > 1]
    if not multiple_contracts.empty:
        st.info(f"🤝 **Consolidation Opportunity**: {len(multiple_contracts)} vendors have multiple contracts - consider master agreements")
    
    # Contract timeline
    st.markdown("## 📅 Contract Timeline")
    
    # Create timeline data
    timeline_data = []
    for _, row in df_temp.head(20).iterrows():  # Show top 20 for readability
        timeline_data.append({
            'Contract': row['contract_id'],
            'Vendor': row['counterparty'],
            'Start': row['start_date'],
            'End': row['end_date'],
            'Value': row['total_value_usd']
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.timeline(
        timeline_df,
        x_start="Start", 
        x_end="End", 
        y="Contract",
        color="Value",
        title="Contract Timeline (Top 20 by Value)"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_all_insights_tab(cfo_insights: List):
    """Create tab showing all 30 CFO insights"""
    
    st.markdown("## 📋 Complete CFO Analytics Report")
    
    dimensions = {}
    for insight in cfo_insights:
        dim = insight['dimension']
        if dim not in dimensions:
            dimensions[dim] = []
        dimensions[dim].append(insight)
    
    for dimension, insights in dimensions.items():
        with st.expander(f"📁 {dimension} ({len(insights)} insights)", expanded=False):
            for insight in insights:
                formatted_text = format_insights_clean_text(insight)
                st.markdown(formatted_text)
                st.markdown("---")


def generate_contract_specific_insights(contract_data: pd.Series) -> List[Dict]:
    """Generate contract-specific insights for all 30 CFO questions"""
    
    import sys
    sys.path.append('..')
    import config
    
    insights = []
    
    for dimension, question in config.CFO_DIMENSIONS_AND_QUESTIONS:
        # Generate contract-specific insights
        if "total value" in question.lower():
            insight = f"Contract value: ${contract_data['total_value_usd']:,.0f} with {contract_data['counterparty']}"
            kpis = {
                "contract_value": int(contract_data['total_value_usd']),
                "counterparty": str(contract_data['counterparty']),
                "contract_type": str(contract_data['type'])
            }
            opportunities = ["Review pricing structure", "Negotiate volume discounts"]
            confidence = 0.9
        elif "payment terms" in question.lower():
            insight = f"Payment terms: {contract_data['payment_terms']} - Impact on cash flow"
            kpis = {
                "payment_terms": str(contract_data['payment_terms']),
                "annual_commitment": int(contract_data['annual_commitment_usd'])
            }
            opportunities = ["Optimize payment schedule", "Negotiate early payment discounts"]
            confidence = 0.8
        elif "vendor" in question.lower() or "counterparty" in question.lower():
            insight = f"Vendor: {contract_data['counterparty']} - {contract_data['type']} contract"
            kpis = {
                "vendor": str(contract_data['counterparty']),
                "contract_type": str(contract_data['type']),
                "relationship_strength": "High" if contract_data['total_value_usd'] > 5000000 else "Medium"
            }
            opportunities = ["Strengthen vendor relationship", "Explore additional services"]
            confidence = 0.8
        elif "risk" in question.lower() or "compliance" in question.lower():
            insight = f"Risk assessment: SLA {contract_data['sla_uptime']}, Compliance {contract_data['compliance']}"
            kpis = {
                "sla_uptime": str(contract_data['sla_uptime']),
                "compliance": str(contract_data['compliance']),
                "risk_level": "High" if '99.5' in str(contract_data['sla_uptime']) else "Medium",
                "escalation_rate": str(contract_data.get('escalation', 'Not specified')),
                "termination_notice": str(contract_data.get('termination', 'Not specified'))
            }
            opportunities = ["Improve SLA monitoring", "Enhance compliance framework"]
            confidence = 0.7
        elif "financial" in question.lower() or "revenue" in question.lower() or "profitability" in question.lower():
            insight = f"Financial position: ${contract_data['total_value_usd']:,.0f} total value, ${contract_data['annual_commitment_usd']:,.0f} annual commitment"
            kpis = {
                "total_contract_value": int(contract_data['total_value_usd']),
                "annual_commitment": int(contract_data['annual_commitment_usd']),
                "contract_years": str(contract_data.get('escalation', 'Standard')),
                "payment_terms": str(contract_data['payment_terms']),
                "variable_costs": str(contract_data.get('variable_costs', 'None'))
            }
            opportunities = ["Optimize payment terms", "Negotiate better pricing", "Review escalation clauses"]
            confidence = 0.9
        elif "sla" in question.lower() or "performance" in question.lower():
            insight = f"Performance metrics: {contract_data['sla_uptime']} uptime target, {contract_data['sla_response_critical']} critical response"
            kpis = {
                "sla_uptime": str(contract_data['sla_uptime']),
                "critical_response_time": str(contract_data['sla_response_critical']),
                "critical_resolution_time": str(contract_data['sla_resolution_critical']),
                "sla_penalties": str(contract_data.get('sla_penalty', 'Not specified')),
                "performance_tier": "High" if '99.9' in str(contract_data['sla_uptime']) else "Standard"
            }
            opportunities = ["Monitor SLA performance", "Negotiate penalty terms", "Improve response times"]
            confidence = 0.8
        else:
            # Generic contract insight
            insight = f"Contract analysis: {contract_data['type']} with {contract_data['counterparty']}"
            kpis = {
                "contract_id": str(contract_data['contract_id']),
                "status": str(contract_data['status']),
                "value": int(contract_data['total_value_usd'])
            }
            opportunities = ["Conduct detailed contract review"]
            confidence = 0.6
        
        insights.append({
            "dimension": dimension,
            "question": question,
            "insight": insight,
            "kpis": kpis,
            "risks": [],
            "opportunities": opportunities,
            "confidence": confidence
        })
    
    return insights

def main():
    """Main dashboard function"""
    # Navigation sidebar with ContractIQ branding
    st.sidebar.markdown("# 🧠 ContractIQ")
    st.sidebar.markdown("*Clarity, Compliance, and Control*")
    st.sidebar.markdown("---")
    
    # Navigation menu - clickable buttons
    st.sidebar.markdown("### 🧭 Navigation")
    
    if st.sidebar.button("🧠 InsightIQ", use_container_width=True):
        st.session_state['current_page'] = "InsightIQ"
    if st.sidebar.button("📊 Portfolio Management", use_container_width=True):
        st.session_state['current_page'] = "Portfolio Management"
    if st.sidebar.button("💰 Cash Flow Analysis", use_container_width=True):
        st.session_state['current_page'] = "Cash Flow Analysis"
    if st.sidebar.button("💸 Cost Control & Risk", use_container_width=True):
        st.session_state['current_page'] = "Cost Control & Risk Management"
    if st.sidebar.button("📈 Executive Insights", use_container_width=True):
        st.session_state['current_page'] = "Executive Insights"
    
    # Set default page if not set
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "InsightIQ"
    
    page = st.session_state['current_page']
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Portfolio Overview")
    
    # Load analytics data
    with st.spinner("Loading ContractIQ analytics..."):
        analytics_data = load_cfo_analytics_data("unified")
    
    # Show portfolio overview in sidebar
    if analytics_data["contract_csv"] is not None:
        df = analytics_data["contract_csv"]
        total_value = df['total_value_usd'].sum()
        total_annual = df['annual_commitment_usd'].sum()
        active_contracts = len(df[df['status'].str.contains('Active', na=False)])
        
        st.sidebar.metric("💰 Total Portfolio Value", f"${total_value:,.0f}")
        st.sidebar.metric("💸 Annual Commitments", f"${total_annual:,.0f}")
        st.sidebar.metric("⚡ Active Contracts", active_contracts)
    
    if analytics_data["contract_csv"] is None:
        st.error("No contract data available")
        return
    
    # Route to different pages based on navigation
    if page == "InsightIQ":
        create_insightiq_page(analytics_data)
    elif page == "Portfolio Management":
        create_portfolio_management_page(analytics_data)
    elif page == "Cash Flow Analysis":
        create_cash_flow_analysis_page(analytics_data)
    elif page == "Cost Control & Risk Management":
        create_cost_control_risk_page(analytics_data)
    elif page == "Executive Insights":
        create_executive_insights_page(analytics_data)
    
    # Export functionality
    st.markdown("---")
    st.markdown("## 📤 Export CFO Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Complete CFO Analysis"):
            try:
                # Export comprehensive report
                export_files = export.export_all_cfo_formats("cfo_reports")
                
                if export_files:
                    st.success(f"✅ Reports exported to {list(export_files.keys())}")
                    
                    # Create download links
                    for file_type, file_path in export_files.items():
                        if file_path and os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label=f"📥 Download {file_type}",
                                    data=f.read(),
                                    file_name=os.path.basename(file_path),
                                    mime="application/octet-stream"
                                )
                else:
                    st.error("Export failed. Please check the system logs.")
                    
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        if st.button("🔄 Refresh Dashboard with Latest Export"):
            st.cache_data.clear()
            st.rerun()
    
    # Manual JSONL Import Section
    st.markdown("---")
    st.markdown("## 📋 Import JSONL CFO Insights")
    
    uploaded_file = st.file_uploader(
        "Upload your CFO JSONL file:", 
        type=['jsonl', 'json'],
        help="Upload exported CFO insights JSONL file for manual dashboard refresh"
    )
    
    if uploaded_file is not None:
        try:
            # Parse uploaded JSONL
            jsonl_content = uploaded_file.read().decode('utf-8')
            manual_insights = []
            
            for line_num, line in enumerate(jsonl_content.split('\n'), 1):
                line = line.strip()
                if line:
                    try:
                        insight = json.loads(line)
                        manual_insights.append(insight)
                    except json.JSONDecodeError as e:
                        st.warning(f"Skipping malformed JSON on line {line_num}: {e}")
            
            if manual_insights:
                st.success(f"✅ Successfully loaded {len(manual_insights)} insights from uploaded file")
                
                # Group insights by dimension
                dimensions = {}
                for insight in manual_insights:
                    dim = insight.get('dimension', 'Unknown')
                    if dim not in dimensions:
                        dimensions[dim] = []
                    dimensions[dim].append(insight)
                
                # Display each dimension separately (NO NESTED EXPANDERS)
                st.markdown(f"## 📋 CFO Insights ({len(manual_insights)} total)")
                
                for dimension, insights_in_dim in dimensions.items():
                    st.markdown(f"### 📁 {dimension} ({len(insights_in_dim)} insights)")
                    
                    for i, insight in enumerate(insights_in_dim):
                        with st.expander(f"💡 {insight.get('question', 'Question not specified')} [#{i+1}]"):
                            formatted_text = format_insights_clean_text(insight)
                            st.markdown(formatted_text)
                        
                        if i < len(insights_in_dim) - 1:
                            st.markdown("---")
            else:
                st.error("No valid insights found in uploaded file")
                
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

if __name__ == "__main__":
    main()
