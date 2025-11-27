"""
CFO Dashboard for Contract Analysis
Streamlit application for comprehensive contract portfolio management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import io

# Set page config
st.set_page_config(
    page_title="CFO Contract Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
    }
    .main-metric {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-metric {
        font-size: 1.2rem;
        color: #666;
    }
    .risk-high { color: #d62728; }
    .risk-medium { color: #ff7f0e; }
    .risk-low { color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_contract_data():
    """Load contract data from CSV"""
    try:
        df = pd.read_csv("../dummy_contracts_50.csv")
        
        # Convert date columns
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        # Parse escalation percentages
        df['escalation_rate'] = df['escalation'].str.extract('(\d+)%').astype(float)
        
        # Parse SLA uptime percentages
        df['sla_uptime_pct'] = df['sla_uptime'].str.extract('(\d+\.?\d*)').astype(float)
        
        # Create risk categories based on SLA
        df['risk_category'] = df['sla_uptime_pct'].apply(
            lambda x: 'Low' if x >= 99.99 else 'Medium' if x >= 99.5 else 'High'
        )
        
        # Parse termination policies
        df['immediate_termination'] = df['termination'].str.contains('Immediate termination', na=False)
        
        # Calculate days until expiration
        df['days_to_expiry'] = (df['end_date'] - datetime.now()).dt.days
        
        # Create expiry risk categories
        df['expiry_risk'] = df['days_to_expiry'].apply(
            lambda x: 'Critical' if x < 90 else 'High' if x < 180 else 'Medium' if x < 365 else 'Low'
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_kpis(df):
    """Calculate key performance indicators"""
    active_df = df[df['status'] == 'Active']
    
    total_contract_value = df['total_value_usd'].sum()
    total_annual_commitment = df['annual_commitment_usd'].sum()
    
    status_counts = df['status'].value_counts()
    active_count = len(active_df)
    expiring_soon = len(df[df['days_to_expiry'] < 180])
    
    risk_high_count = len(df[df['risk_category'] == 'High'])
    risk_high_pct = (risk_high_count / len(df)) * 100
    
    esg_count = len(df[df['esg_reporting'] != 'None required'])
    esg_pct = (esg_count / len(df)) * 100
    
    avg_payment_days = df['payment_terms'].str.extract('Net (\d+)').astype(float).mean()[0] if len(df) > 0 else 0
    
    top_vendors = df.groupby('counterparty')['total_value_usd'].sum().nlargest(5)
    
    return {
        'total_contract_value': total_contract_value,
        'total_annual_commitment': total_annual_commitment,
        'active_count': active_count,
        'expiring_soon': expiring_soon,
        'risk_high_pct': risk_high_pct,
        'esg_pct': esg_pct,
        'avg_payment_days': avg_payment_days,
        'top_vendors': top_vendors,
        'status_counts': status_counts
    }

def create_kpi_cards(kpis):
    """Create KPI metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="main-metric">${kpis['total_contract_value']:,.0f}</div>
            <div class="sub-metric">Total Contract Value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="main-metric">${kpis['total_annual_commitment']:,.0f}</div>
            <div class="sub-metric">Annual Commitment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="main-metric">{kpis['active_count']}</div>
            <div class="sub-metric">Active Contracts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="main-metric" class="risk-high">{kpis['expiring_soon']}</div>
            <div class="sub-metric">Expiring < 6 Months</div>
        </div>
        """, unsafe_allow_html=True)

def create_financial_charts(df):
    """Create financial analysis charts"""
    st.subheader("💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spend by Contract Type
        spend_by_type = df.groupby('type')['total_value_usd'].sum().sort_values(ascending=False)
        fig1 = px.bar(
            x=spend_by_type.values, 
            y=spend_by_type.index,
            orientation='h',
            title="Total Spend by Contract Type",
            labels={'x': 'Value (USD)', 'y': 'Contract Type'}
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Spend by Counterparty
        spend_by_vendor = df.groupby('counterparty')['total_value_usd'].sum().nlargest(10)
        fig2 = px.bar(
            x=spend_by_vendor.index,
            y=spend_by_vendor.values,
            title="Top 10 Vendors by Spend",
            labels={'x': 'Vendor', 'y': 'Value (USD)'}
        )
        fig2.update_layout(height=400)
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Payment Terms Distribution
    st.subheader("📅 Payment Terms Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        payment_terms = df['payment_terms'].value_counts()
        fig3 = px.pie(values=payment_terms.values, names=payment_terms.index, title="Payment Terms Distribution")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Average Contract Value by Type
        avg_value_by_type = df.groupby('type')['total_value_usd'].mean().sort_values(ascending=False)
        fig4 = px.bar(
            x=avg_value_by_type.index,
            y=avg_value_by_type.values,
            title="Average Contract Value by Type",
            labels={'x': 'Contract Type', 'y': 'Avg Value (USD)'}
        )
        fig4.update_xaxes(tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)

def create_risk_monitoring(df):
    """Create risk monitoring dashboard"""
    st.subheader("⚠️ Risk & SLA Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk Distribution
        risk_counts = df['risk_category'].value_counts()
        colors = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
        fig1 = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            title="Contract Risk Distribution",
            color_discrete_map=colors
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Expiry Risk
        expiry_counts = df['expiry_risk'].value_counts()
        colors = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#ffff00', 'Low': '#2ca02c'}
        fig2 = px.pie(
            values=expiry_counts.values, 
            names=expiry_counts.index,
            title="Contract Expiry Risk",
            color_discrete_map=colors
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        # SLA Distribution
        sla_counts = df['sla_uptime'].value_counts()
        fig3 = px.bar(
            x=sla_counts.index,
            y=sla_counts.values,
            title="SLA Uptime Targets",
            labels={'x': 'SLA Target', 'y': 'Number of Contracts'}
        )
        st.plotly_chart(fig3, use_container_width=True)

def create_compliance_matrix(df):
    """Create compliance tracking"""
    st.subheader("📋 Compliance & Governance")
    
    # Parse compliance requirements
    compliance_standards = ['SOC2', 'GDPR', 'CCPA', 'ISO 27001', 'HIPAA', 'FCPA', 'PCI DSS', 'India DPA 2023', 'UK Bribery Act']
    
    # Create compliance matrix
    compliance_matrix = pd.DataFrame(index=compliance_standards, columns=range(len(df)))
    
    for idx, standard in enumerate(compliance_standards):
        compliance_matrix.iloc[idx] = df['compliance'].str.contains(standard, na=False).astype(int)
    
    compliance_pct = (compliance_matrix.sum(axis=1) / len(df)) * 100
    
    # Create compliance heatmap
    fig = px.bar(
        x=compliance_pct.values,
        y=compliance_pct.index,
        orientation='h',
        title="Compliance Coverage by Standard",
        labels={'x': 'Coverage (%)', 'y': 'Compliance Standard'},
        color=compliance_pct.values,
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def create_contract_lifecycle(df):
    """Create contract lifecycle analysis"""
    st.subheader("🔄 Contract Lifecycle & Obligations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Contract Status Distribution
        status_counts = df['status'].value_counts()
        fig1 = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Contract Status Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Contracts by Jurisdiction
        jurisdiction_counts = df['governing_law'].value_counts()
        fig2 = px.bar(
            x=jurisdiction_counts.index,
            y=jurisdiction_counts.values,
            title="Contracts by Jurisdiction",
            labels={'x': 'Governing Law', 'y': 'Number of Contracts'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # ESG Reporting Analysis
    st.subheader("🌱 ESG Reporting Tracker")
    esg_analysis = df['esg_reporting'].value_counts()
    fig3 = px.bar(
        x=esg_analysis.index,
        y=esg_analysis.values,
        title="ESG Reporting Requirements",
        labels={'x': 'ESG Reporting Type', 'y': 'Number of Contracts'}
    )
    fig3.update_xaxes(tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)

def create_cash_flow_impact_dashboard(df):
    """Create CFO-focused cash flow impact dashboard"""
    st.subheader("💰 Cash Flow Impact Analysis")
    
    # Key metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_annual = df['annual_commitment_usd'].sum()
        st.metric("Total Annual Commitments", f"${total_annual:,.0f}")
    
    with col2:
        unfavorable_terms = df[df['payment_terms'].isin(['Net 60', 'Net 90', 'Milestone-based'])]
        unfavorable_value = unfavorable_terms['annual_commitment_usd'].sum()
        st.metric("Unfavorable Payment Terms", f"${unfavorable_value:,.0f}")
    
    with col3:
        avg_payment_days = df['payment_terms'].str.extract(r'(\d+)').astype(float).mean().iloc[0]
        st.metric("Average Payment Terms", f"{avg_payment_days:.0f} days")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment terms impact on cash flow
        payment_analysis = df.groupby('payment_terms')['annual_commitment_usd'].sum().reset_index()
        payment_analysis['cash_flow_impact'] = payment_analysis['payment_terms'].str.extract(r'(\d+)').astype(float)
        
        fig = px.bar(
            payment_analysis, 
            x='payment_terms', 
            y='annual_commitment_usd',
            title="Annual Commitments by Payment Terms",
            labels={'annual_commitment_usd': 'Annual Value (USD)', 'payment_terms': 'Payment Terms'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cash flow timeline
        df_temp = df.copy()
        df_temp['start_date'] = pd.to_datetime(df_temp['start_date'])
        df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
        
        # Create timeline data
        timeline_data = []
        for _, row in df_temp.iterrows():
            timeline_data.append({
                'Contract': row['contract_id'],
                'Vendor': row['counterparty'],
                'Start': row['start_date'],
                'End': row['end_date'],
                'Value': row['annual_commitment_usd']
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig = px.timeline(
            timeline_df.head(20),  # Show top 20 for readability
            x_start="Start", 
            x_end="End", 
            y="Contract",
            color="Value",
            title="Contract Timeline (Top 20 by Value)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Unfavorable payment terms table
    st.subheader("⚠️ Contracts with Unfavorable Payment Terms")
    if not unfavorable_terms.empty:
        unfavorable_display = unfavorable_terms[['contract_id', 'counterparty', 'payment_terms', 'annual_commitment_usd']].copy()
        unfavorable_display['annual_commitment_usd'] = unfavorable_display['annual_commitment_usd'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(unfavorable_display, use_container_width=True)
    else:
        st.success("✅ No contracts with unfavorable payment terms!")

def create_risk_management_dashboard(df):
    """Create CFO-focused risk management dashboard"""
    st.subheader("🚨 Risk Management Dashboard")
    
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
    
    # Charts
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
    st.subheader("🚨 Risk Alerts")
    
    # High concentration risk
    high_concentration = df.groupby('counterparty')['total_value_usd'].sum()
    high_concentration_pct = (high_concentration / total_value * 100).sort_values(ascending=False)
    
    if high_concentration_pct.iloc[0] > 20:
        st.warning(f"⚠️ **High Concentration Risk**: {high_concentration_pct.index[0]} represents {high_concentration_pct.iloc[0]:.1f}% of total portfolio value")
    
    # Missing compliance
    missing_compliance = df[df['compliance'].isna() | (df['compliance'] == '')]
    if not missing_compliance.empty:
        st.warning(f"⚠️ **Missing Compliance Data**: {len(missing_compliance)} contracts lack compliance information")

def create_cost_control_dashboard(df):
    """Create CFO-focused cost control dashboard"""
    st.subheader("💸 Cost Control & Optimization")
    
    # Cost metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Total annual commitment
        total_annual = df['annual_commitment_usd'].sum()
        st.metric("Total Annual Commitment", f"${total_annual:,.0f}")
    
    with col2:
        # Contracts with escalation
        escalation_contracts = df[df['escalation'].notna() & (df['escalation'] != 'None')]
        escalation_value = escalation_contracts['annual_commitment_usd'].sum()
        st.metric("Escalation Exposure", f"${escalation_value:,.0f}")
    
    with col3:
        # Variable cost contracts
        variable_cost_contracts = df[df['variable_costs'].notna() & (df['variable_costs'] != '')]
        variable_value = variable_cost_contracts['annual_commitment_usd'].sum()
        st.metric("Variable Cost Exposure", f"${variable_value:,.0f}")
    
    # Charts
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
    st.subheader("💡 Cost Optimization Opportunities")
    
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

def create_strategic_planning_dashboard(df):
    """Create CFO-focused strategic planning dashboard"""
    st.subheader("📅 Strategic Planning & Renewal Pipeline")
    
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
    st.subheader("🎯 Strategic Recommendations")
    
    # Renewal urgency
    urgent_renewals = df_temp[df_temp['months_to_expiry'] <= 6]
    if not urgent_renewals.empty:
        st.warning(f"🚨 **Urgent Renewals**: {len(urgent_renewals)} contracts expiring in 6 months - start renewal negotiations")
    
    # Vendor consolidation opportunities
    vendor_counts = df['counterparty'].value_counts()
    multiple_contracts = vendor_counts[vendor_counts > 1]
    if not multiple_contracts.empty:
        st.info(f"🤝 **Consolidation Opportunity**: {len(multiple_contracts)} vendors have multiple contracts - consider master agreements")

def create_filters_sidebar(df):
    """Create filters in sidebar"""
    st.sidebar.header("🔍 Filters")
    
    # Contract Type Filter
    contract_types = ['All'] + list(df['type'].unique())
    selected_type = st.sidebar.selectbox("Contract Type", contract_types)
    
    # Counterparty Filter
    counterparties = ['All'] + list(df['counterparty'].unique())
    selected_counterparty = st.sidebar.selectbox("Counterparty", counterparties)
    
    # Status Filter
    statuses = ['All'] + list(df['status'].unique())
    selected_status = st.sidebar.selectbox("Status", statuses)
    
    # Risk Level Filter
    risk_levels = ['All'] + list(df['risk_category'].unique())
    selected_risk = st.sidebar.selectbox("Risk Level", risk_levels)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['type'] == selected_type]
    
    if selected_counterparty != 'All':
        filtered_df = filtered_df[filtered_df['counterparty'] == selected_counterparty]
    
    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['status'] == selected_status]
    
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['risk_category'] == selected_risk]
    
    st.sidebar.write(f"**Filtered Contracts:** {len(filtered_df)} of {len(df)}")
    
    return filtered_df

def main():
    """Main dashboard function"""
    st.columns([1, 1, 1])
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("💼 CFO Contract Dashboard")
        st.markdown("*Comprehensive contract portfolio management and analysis*")
    
    # Load data
    df = load_contract_data()
    if df is None:
        st.error("Failed to load contract data. Please check the CSV file.")
        return
    
    # Create filters and filter data
    filtered_df = create_filters_sidebar(df)
    
    # Calculate KPIs
    kpis = calculate_kpis(filtered_df)
    
    # Display KPI cards
    create_kpi_cards(kpis)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💰 Cash Flow Impact", "🚨 Risk Management", "💸 Cost Control", "📅 Strategic Planning", "📊 Financial Analysis", "📋 Contract Details"])
    
    with tab1:
        create_cash_flow_impact_dashboard(filtered_df)
    
    with tab2:
        create_risk_management_dashboard(filtered_df)
    
    with tab3:
        create_cost_control_dashboard(filtered_df)
    
    with tab4:
        create_strategic_planning_dashboard(filtered_df)
    
    with tab5:
        create_financial_charts(filtered_df)
    
    with tab6:
        st.subheader("📋 Contract Details Table")
        
        # Contract details table
        display_columns = [
            'contract_id', 'type', 'counterparty', 'start_date', 'end_date', 
            'status', 'total_value_usd', 'annual_commitment_usd', 
            'risk_category', 'esg_reporting', 'governing_law'
        ]
        
        # Add summary column if it exists
        if 'summary' in df.columns:
            display_columns.append('summary')
        
        display_df = filtered_df[display_columns].copy()
        display_df['total_value_usd'] = display_df['total_value_usd'].apply(lambda x: f"${x:,.0f}")
        display_df['annual_commitment_usd'] = display_df['annual_commitment_usd'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_contracts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()





