"""
Portfolio Management Page
Strategic planning and renewal pipeline management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict

def create_page(analytics_data: Dict):
    """Create Portfolio Management page"""
    df = analytics_data["contract_csv"]
    cfo_insights = analytics_data["cfo_insights"]
    
    st.markdown("## 📊 Portfolio Management")
    
    # Portfolio Overview (first section)
    st.markdown("## 📈 Portfolio Overview")
    create_portfolio_overview(df)
    
    st.markdown("---")
    
    # Strategic Planning (second section)
    st.markdown("## 📅 Strategic Planning")
    create_strategic_planning(df, cfo_insights)
    
    st.markdown("---")
    
    # Renewal Pipeline (third section)
    st.markdown("## 🔄 Renewal Pipeline")
    create_renewal_pipeline(df)
    
    st.markdown("---")
    
    # JSONL Import Section (fourth section)
    st.markdown("## 📋 Import JSONL CFO Insights")
    create_jsonl_import_section()
    
def create_portfolio_overview(df: pd.DataFrame):
    """Create portfolio overview section"""
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_contracts = len(df)
        st.metric("💰 Total Contracts", total_contracts)
    
    with col2:
        total_value = df['total_value_usd'].sum()
        st.metric("💸 Total Portfolio Value", f"${total_value:,.0f}")
    
    with col3:
        total_annual = df['annual_commitment_usd'].sum()
        st.metric("⚡ Annual Commitments", f"${total_annual:,.0f}")
    
    with col4:
        active_contracts = len(df[df['status'].str.contains('Active', na=False)])
        st.metric("🟢 Active Contracts", active_contracts)
    
    # Portfolio charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Contract types distribution
        contract_types = df['type'].value_counts()
        fig = px.pie(
            values=contract_types.values,
            names=contract_types.index,
            title="Contract Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top vendors by value
        vendor_values = df.groupby('counterparty')['total_value_usd'].sum().sort_values(ascending=False).head(5)
        fig = px.bar(
            x=vendor_values.values,
            y=vendor_values.index,
            orientation='h',
            title="Top 5 Vendors by Contract Value",
            labels={'x': 'Total Value (USD)', 'y': 'Vendor'}
        )
        st.plotly_chart(fig, use_container_width=True)

def create_strategic_planning(df: pd.DataFrame, cfo_insights):
    """Create strategic planning section"""
    
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
    
    # Strategic planning charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Contract expiry timeline
        df_temp = df.copy()
        df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
        df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
        
        # Group by expiry periods
        expiry_periods = []
        for _, row in df_temp.iterrows():
            months = row['months_to_expiry']
            if months <= 6:
                expiry_periods.append('Next 6 Months')
            elif months <= 12:
                expiry_periods.append('6-12 Months')
            elif months <= 24:
                expiry_periods.append('1-2 Years')
            else:
                expiry_periods.append('Beyond 2 Years')
        
        df_temp['expiry_period'] = expiry_periods
        expiry_counts = df_temp['expiry_period'].value_counts()
        
        fig = px.bar(
            x=expiry_counts.values,
            y=expiry_counts.index,
            orientation='h',
            title="Contract Expiry Timeline",
            labels={'x': 'Number of Contracts', 'y': 'Expiry Period'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Contract value distribution
        fig = px.histogram(
            df,
            x='total_value_usd',
            nbins=10,
            title="Contract Value Distribution",
            labels={'total_value_usd': 'Contract Value (USD)', 'count': 'Number of Contracts'}
        )
        st.plotly_chart(fig, use_container_width=True)

def create_renewal_pipeline(df: pd.DataFrame):
    """Create renewal pipeline section"""
    
    # Renewal metrics
    df_temp = df.copy()
    df_temp['end_date'] = pd.to_datetime(df_temp['end_date'])
    df_temp['months_to_expiry'] = (df_temp['end_date'] - pd.Timestamp.now()).dt.days / 30
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Contracts expiring in 6 months (only future)
        expiring_6m = df_temp[(df_temp['months_to_expiry'] >= 0) & (df_temp['months_to_expiry'] <= 6)]
        st.metric("🟠 Expiring in 6 Months", len(expiring_6m))
    
    with col2:
        # Contracts expiring in 1 year (only future)
        expiring_1y = df_temp[(df_temp['months_to_expiry'] >= 0) & (df_temp['months_to_expiry'] <= 12)]
        st.metric("🟡 Expiring in 1 Year", len(expiring_1y))
    
    with col3:
        # Total renewal value (only future expiries)
        expiring_1y_future = df_temp[(df_temp['months_to_expiry'] >= 0) & (df_temp['months_to_expiry'] <= 12)]
        renewal_value = expiring_1y_future['total_value_usd'].sum()
        st.metric("💰 Total Renewal Value", f"${renewal_value:,.0f}")
    
    # Renewal pipeline table
    st.markdown("### 📋 Renewal Pipeline")
    
    # Filter contracts expiring in next 12 months (only future expiries, not past)
    renewal_contracts = df_temp[(df_temp['months_to_expiry'] >= 0) & (df_temp['months_to_expiry'] <= 12)].copy()
    renewal_contracts = renewal_contracts.sort_values('months_to_expiry')
    
    if not renewal_contracts.empty:
        # Prepare table data
        table_data = renewal_contracts[[
            'contract_id', 'counterparty', 'type', 'end_date', 'months_to_expiry',
            'total_value_usd', 'annual_commitment_usd', 'status'
        ]].copy()
        
        # Format data
        table_data['end_date'] = table_data['end_date'].dt.strftime('%Y-%m-%d')
        table_data['months_to_expiry'] = table_data['months_to_expiry'].round(1)
        table_data['total_value_usd'] = table_data['total_value_usd'].apply(lambda x: f"${x:,.0f}")
        table_data['annual_commitment_usd'] = table_data['annual_commitment_usd'].apply(lambda x: f"${x:,.0f}")
        
        # Add priority status
        def get_renewal_priority(months):
            if months <= 3:
                return "🔴 Critical"
            elif months <= 6:
                return "🟠 High"
            elif months <= 12:
                return "🟡 Medium"
            else:
                return "🟢 Low"
        
        table_data['priority'] = table_data['months_to_expiry'].apply(get_renewal_priority)
        
        # Rename columns
        table_data = table_data.rename(columns={
            'contract_id': 'Contract ID',
            'counterparty': 'Vendor',
            'type': 'Type',
            'end_date': 'End Date',
            'months_to_expiry': 'Months to Expiry',
            'total_value_usd': 'Total Value',
            'annual_commitment_usd': 'Annual Commitment',
            'status': 'Status',
            'priority': 'Priority'
        })
        
        # Display table
        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No contracts expiring in the next 12 months.")

def create_jsonl_import_section():
    """Create JSONL import section for CFO insights"""
    import json
    
    # Initialize session state for uploaded insights
    if 'uploaded_insights' not in st.session_state:
        st.session_state.uploaded_insights = []
    
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    uploaded_file = st.file_uploader(
        "Upload your CFO JSONL file:", 
        type=['jsonl', 'json'],
        help="Upload exported CFO insights JSONL file for manual dashboard refresh"
    )
    
    # If a new file is uploaded, process it and store in session state
    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.uploaded_insights = []  # Clear previous data
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
                
                # Store in session state
                st.session_state.uploaded_insights = manual_insights
                
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
                st.session_state.uploaded_insights = []
    
    # Display persistent insights from session state
    if st.session_state.uploaded_insights:
        st.success(f"✅ Loaded {len(st.session_state.uploaded_insights)} insights from {st.session_state.uploaded_file_name}")
        
        # Add clear button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear Insights", type="secondary"):
                st.session_state.uploaded_insights = []
                st.session_state.uploaded_file_name = None
                st.rerun()
        
        # Group insights by dimension
        dimensions = {}
        for insight in st.session_state.uploaded_insights:
            dim = insight.get('dimension', 'Unknown')
            if dim not in dimensions:
                dimensions[dim] = []
            dimensions[dim].append(insight)
        
        # Display each dimension separately
        st.markdown(f"### 📋 CFO Insights ({len(st.session_state.uploaded_insights)} total)")
        
        for dimension, insights_in_dim in dimensions.items():
            st.markdown(f"#### 📁 {dimension} ({len(insights_in_dim)} insights)")
            
            for i, insight in enumerate(insights_in_dim):
                with st.expander(f"💡 {insight.get('question', 'Question not specified')} [#{i+1}]"):
                    # Format insight for display
                    formatted_text = f"**Question:** {insight.get('question', 'Not specified')}\n\n"
                    formatted_text += f"**Insight:** {insight.get('insight', 'No insight available')}\n\n"
                    
                    # KPIs as bullet points
                    kpis = insight.get('kpis', {})
                    if kpis:
                        formatted_text += "**Key Metrics:**\n"
                        for key, value in kpis.items():
                            clean_key = key.replace('_', ' ').title()
                            formatted_text += f"• {clean_key}: {value}\n"
                        formatted_text += "\n"
                    
                    # Risks as bullet points
                    risks = insight.get('risks', [])
                    if risks:
                        formatted_text += "**Risks Identified:**\n"
                        for risk in risks:
                            formatted_text += f"• {risk}\n"
                        formatted_text += "\n"
                    
                    # Opportunities as bullet points
                    opportunities = insight.get('opportunities', [])
                    if opportunities:
                        formatted_text += "**Opportunities:**\n"
                        for opp in opportunities:
                            formatted_text += f"• {opp}\n"
                        formatted_text += "\n"
                    
                    # Confidence
                    confidence = insight.get('confidence', 0)
                    formatted_text += f"**Confidence Level:** {confidence:.0%}"
                    
                    st.markdown(formatted_text)
                
                if i < len(insights_in_dim) - 1:
                    st.markdown("---")
    elif st.session_state.uploaded_file_name is None:
        st.info("📤 Upload a JSONL file to view CFO insights")

