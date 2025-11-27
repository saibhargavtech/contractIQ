"""
Cost Control & Risk Management Page
Combined cost control and risk management functionality
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List

def create_page(analytics_data: Dict):
    """Create Cost Control & Risk Management page"""
    df = analytics_data["contract_csv"]
    cfo_insights = analytics_data["cfo_insights"]
    risk_assessment = analytics_data["risk_assessment"]
    
    st.markdown("## 💸 Cost Control & Risk Management")
    
    # Key Performance Indicators
    st.markdown("### 📊 Key Performance Indicators")
    
    # Cost Control KPIs
    st.markdown("#### 💸 Cost Control")
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
    
    # Risk Management KPIs
    st.markdown("#### 🚨 Risk Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Top vendor concentration
        vendor_concentration = df.groupby('counterparty')['total_value_usd'].sum().sort_values(ascending=False)
        top_vendor_pct = (vendor_concentration.iloc[0] / vendor_concentration.sum()) * 100
        st.metric("Top Vendor Concentration", f"{top_vendor_pct:.1f}%")
    
    with col2:
        # Penalty exposure
        penalty_contracts = df[df['sla_penalty'].notna() & (df['sla_penalty'] != '')]
        penalty_value = penalty_contracts['total_value_usd'].sum()
        st.metric("Penalty Exposure", f"${penalty_value:,.0f}")
    
    with col3:
        # Compliance coverage
        compliance_contracts = df[df['compliance'].notna() & (df['compliance'] != '')]
        compliance_pct = (len(compliance_contracts) / len(df)) * 100
        st.metric("Compliance Coverage", f"{compliance_pct:.1f}%")
    
    st.markdown("---")
    
    # Cost Control Section
    st.markdown("## 💸 Cost Control Analysis")
    create_cost_control_charts(df)
    
    st.markdown("---")
    
    # Risk Management Section
    st.markdown("## 🚨 Risk Management Analysis")
    create_risk_management_charts(df)

def create_cost_control_charts(df: pd.DataFrame):
    """Create cost control charts"""
    
    # Escalation contracts chart
    escalation_contracts = df[df['escalation'].notna() & (df['escalation'] != 'None')]
    if not escalation_contracts.empty:
        fig = px.bar(
            escalation_contracts,
            x='counterparty',
            y='annual_commitment_usd',
            color='escalation',
            title="Contracts with Escalation Clauses",
            labels={'annual_commitment_usd': 'Annual Value (USD)', 'counterparty': 'Vendor/Party'},
            hover_data=['contract_id']
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No contracts with escalation clauses found")
    
    # High-cost contracts optimization
    df_temp = df.copy()
    df_temp['cost_efficiency'] = df_temp['annual_commitment_usd'] / df_temp['total_value_usd']
    high_cost_contracts = df_temp[df_temp['cost_efficiency'] > df_temp['cost_efficiency'].quantile(0.8)]
    
    if not high_cost_contracts.empty:
        fig = px.scatter(
            high_cost_contracts,
            x='total_value_usd',
            y='annual_commitment_usd',
            size='annual_commitment_usd',
            hover_data=['contract_id'],
            title="High-Cost Contracts (Optimization Opportunities)",
            labels={'total_value_usd': 'Total Value (USD)', 'annual_commitment_usd': 'Annual Commitment (USD)', 'counterparty': 'Vendor/Party'}
        )
        # Add vendor names as text labels
        fig.add_scatter(
            x=high_cost_contracts['total_value_usd'],
            y=high_cost_contracts['annual_commitment_usd'],
            mode='text',
            text=high_cost_contracts['counterparty'],
            textposition='top center',
            showlegend=False,
            textfont=dict(size=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No high-cost contracts identified")

def create_risk_management_charts(df: pd.DataFrame):
    """Create risk management charts"""
    
    # Vendor concentration chart
    vendor_concentration = df.groupby('counterparty')['total_value_usd'].sum().sort_values(ascending=False).head(10)
    if not vendor_concentration.empty:
        fig = px.pie(
            values=vendor_concentration.values,
            names=vendor_concentration.index,
            title="Top 10 Vendors by Contract Value (Concentration Risk)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Compliance standards chart
    compliance_standards = []
    for compliance_str in df['compliance'].dropna():
        if isinstance(compliance_str, str):
            standards = [s.strip() for s in compliance_str.split(',')]
            compliance_standards.extend(standards)
    
    if compliance_standards:
        compliance_counts = pd.Series(compliance_standards).value_counts().head(10)
        fig = px.bar(
            x=compliance_counts.values,
            y=compliance_counts.index,
            orientation='h',
            title="Top Compliance Standards",
            labels={'x': 'Number of Contracts', 'y': 'Compliance Standard'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # CFO Insights Opportunities
    st.markdown("## 🎯 Cost Optimization Opportunities")
    
    for insight in cfo_insights:
        if insight.get("opportunities"):
            st.markdown(f"**{insight['dimension']}**")
            for opportunity in insight["opportunities"]:
                st.success(f"🎯 {opportunity}")
    
    # Strategic recommendations
    st.markdown("## 🚀 Cost Control Recommendations")
    
    recommendations = [
        "📊 Implement automated contract review cycle",
        "💰 Renegotiate payment terms for cash flow optimization", 
        "🔄 Develop vendor audit framework",
        "📈 Create contract performance dashboards",
        "⚖️ Standardize compliance requirements"
    ]
    
    for rec in recommendations:
        st.info(rec)

def create_risk_management_section(df: pd.DataFrame, risk_assessment: Dict):
    """Create risk management section"""
    
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
        compliance_standards = []
        for compliance_str in df['compliance'].dropna():
            if isinstance(compliance_str, str):
                standards = [s.strip() for s in compliance_str.split(',')]
                compliance_standards.extend(standards)
        
        if compliance_standards:
            compliance_counts = pd.Series(compliance_standards).value_counts().head(10)
            
            fig = px.bar(
                compliance_counts,
                x=compliance_counts.values,
                y=compliance_counts.index,
                orientation='h',
                title="Top Compliance Standards",
                labels={'x': 'Number of Contracts', 'y': 'Compliance Standard'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No compliance data available")
    
    # Risk assessment summary
    st.markdown("## ⚠️ Risk Assessment Summary")
    
    # High-risk contracts
    high_risk_contracts = df[df['sla_uptime'].str.contains('99.5', na=False)]
    if not high_risk_contracts.empty:
        st.warning(f"⚠️ **High SLA Risk**: {len(high_risk_contracts)} contracts with uptime below 99.9%")
    
    # Missing compliance
    missing_compliance = df[df['compliance'].isna() | (df['compliance'] == '')]
    if not missing_compliance.empty:
        st.warning(f"⚠️ **Missing Compliance Data**: {len(missing_compliance)} contracts lack compliance information")

