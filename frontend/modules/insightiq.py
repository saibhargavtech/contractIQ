"""
InsightIQ Page
Main insights page with KPIs, vendor analysis, critical contracts, and expiring timeline
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict

def create_page(analytics_data: Dict):
    """Create InsightIQ page with KPIs, vendor analysis, critical contracts, and expiring timeline"""
    
    # AGGRESSIVE KPI card CSS - Force prevent shrinking
    st.markdown("""
    <style>
        /* Force KPI cards to maintain size */
        .cfo-metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            padding: 1.5rem !important;
            border-radius: 0.8rem !important;
            color: white !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            margin-bottom: 1rem !important;
            min-height: 120px !important;
            min-width: 350px !important;
            width: 100% !important;
            max-width: none !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            flex-shrink: 0 !important;
            flex-grow: 1 !important;
            box-sizing: border-box !important;
        }
        .metric-value {
            font-size: 2.5rem !important;
            font-weight: bold !important;
            margin-bottom: 0.5rem !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
            width: 100% !important;
            min-width: 200px !important;
        }
        .metric-label {
            font-size: 1rem !important;
            opacity: 0.9 !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            width: 100% !important;
        }
        
        /* Force column stability */
        .stColumns > div {
            min-width: 350px !important;
            flex-shrink: 0 !important;
            flex-grow: 1 !important;
        }
        
        /* Override any Streamlit CSS that might interfere */
        div[data-testid="metric-container"] {
            min-width: 350px !important;
            width: 100% !important;
        }
        
        div[data-testid="metric-container"] > div {
            min-width: 350px !important;
            width: 100% !important;
        }
    </style>
    
    <script>
    // Force CSS to stick - prevent Streamlit from overriding
    setInterval(function() {
        var cards = document.querySelectorAll('.cfo-metric-card');
        cards.forEach(function(card) {
            card.style.minWidth = '350px';
            card.style.width = '100%';
            card.style.flexShrink = '0';
        });
        
        var values = document.querySelectorAll('.metric-value');
        values.forEach(function(value) {
            value.style.whiteSpace = 'nowrap';
            value.style.overflow = 'visible';
            value.style.textOverflow = 'clip';
        });
    }, 100);
    </script>
    """, unsafe_allow_html=True)
    
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
    
    # Critical Contracts Section - Show only expiring contracts to match KPI
    st.markdown("## ⚠️ Critical Renewals (Expiring in 6 Months)")
    
    # Show only active contracts expiring in 6 months (to match KPI count)
    expiring_contracts = df_temp[(df_temp['status'].str.contains('Active', na=False)) & (df_temp['months_to_expiry'] <= 6)]
    
    if not expiring_contracts.empty:
        # Create display table with contract names
        display_contracts = expiring_contracts[['contract_id', 'counterparty', 'end_date', 'annual_commitment_usd']].copy()
        display_contracts['Contract Name'] = display_contracts['counterparty'] + ' - ' + display_contracts['contract_id']
        display_contracts['Days to Expiry'] = (pd.to_datetime(display_contracts['end_date']) - pd.Timestamp.now()).dt.days
        display_contracts['Annual Value'] = display_contracts['annual_commitment_usd'].apply(lambda x: f"${x:,.0f}")
        
        # Show only relevant columns
        display_df = display_contracts[['Contract Name', 'Days to Expiry', 'Annual Value']].copy()
        st.dataframe(display_df, use_container_width=True)
    else:
        st.success("✅ No critical renewals identified")
    
    st.markdown("---")
    
    # All Contracts Table
    st.markdown("## 📋 All Contracts")
    
    # Create enhanced contract data for table
    df_table = df.copy()
    df_table['end_date'] = pd.to_datetime(df_table['end_date'])
    df_table['start_date'] = pd.to_datetime(df_table['start_date'])
    df_table['months_to_expiry'] = (df_table['end_date'] - pd.Timestamp.now()).dt.days / 30
    df_table['days_to_expiry'] = (df_table['end_date'] - pd.Timestamp.now()).dt.days
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status:",
            ["All", "Active", "Expired", "Terminated", "Renewed"],
            index=0
        )
    with col2:
        expiry_filter = st.selectbox(
            "Filter by Expiry:",
            ["All", "Expiring in 6 months", "Expiring in 1 year", "Expiring in 2 years"],
            index=0
        )
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["End Date", "Status", "Total Value", "Vendor", "Contract ID"],
            index=0
        )
    
    # Apply filters
    filtered_df = df_table.copy()
    
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['status'].str.contains(status_filter, case=False, na=False)]
    
    if expiry_filter == "Expiring in 6 months":
        filtered_df = filtered_df[filtered_df['months_to_expiry'] <= 6]
    elif expiry_filter == "Expiring in 1 year":
        filtered_df = filtered_df[filtered_df['months_to_expiry'] <= 12]
    elif expiry_filter == "Expiring in 2 years":
        filtered_df = filtered_df[filtered_df['months_to_expiry'] <= 24]
    
    # Apply sorting
    if sort_by == "End Date":
        filtered_df = filtered_df.sort_values('end_date')
    elif sort_by == "Status":
        filtered_df = filtered_df.sort_values('status')
    elif sort_by == "Total Value":
        filtered_df = filtered_df.sort_values('total_value_usd', ascending=False)
    elif sort_by == "Vendor":
        filtered_df = filtered_df.sort_values('counterparty')
    elif sort_by == "Contract ID":
        filtered_df = filtered_df.sort_values('contract_id')
    
    # Prepare table data
    table_data = filtered_df[[
        'contract_id', 'counterparty', 'type', 'status', 
        'start_date', 'end_date', 'days_to_expiry',
        'total_value_usd', 'annual_commitment_usd', 'payment_terms'
    ]].copy()
    
    # Format dates and values
    table_data['start_date'] = table_data['start_date'].dt.strftime('%Y-%m-%d')
    table_data['end_date'] = table_data['end_date'].dt.strftime('%Y-%m-%d')
    table_data['total_value_usd'] = table_data['total_value_usd'].apply(lambda x: f"${x:,.0f}")
    table_data['annual_commitment_usd'] = table_data['annual_commitment_usd'].apply(lambda x: f"${x:,.0f}")
    
    # Add expiry status
    def get_expiry_status(days):
        if days < 0:
            return "🔴 Expired"
        elif days <= 180:  # 6 months
            return "🟠 Expiring Soon"
        elif days <= 365:  # 1 year
            return "🟡 Expiring This Year"
        else:
            return "🟢 Active"
    
    table_data['expiry_status'] = table_data['days_to_expiry'].apply(get_expiry_status)
    
    # Rename columns for better display
    table_data = table_data.rename(columns={
        'contract_id': 'Contract ID',
        'counterparty': 'Vendor',
        'type': 'Type',
        'status': 'Status',
        'start_date': 'Start Date',
        'end_date': 'End Date',
        'days_to_expiry': 'Days to Expiry',
        'total_value_usd': 'Total Value',
        'annual_commitment_usd': 'Annual Commitment',
        'payment_terms': 'Payment Terms',
        'expiry_status': 'Expiry Status'
    })
    
    # Display table
    if not table_data.empty:
        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Days to Expiry": st.column_config.NumberColumn(
                    "Days to Expiry",
                    help="Number of days until contract expires",
                    format="%d"
                )
            }
        )
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Contracts", len(table_data))
        with col2:
            total_value = filtered_df['total_value_usd'].sum()
            st.metric("Total Value", f"${total_value:,.0f}")
        with col3:
            total_annual = filtered_df['annual_commitment_usd'].sum()
            st.metric("Annual Commitments", f"${total_annual:,.0f}")
        with col4:
            expiring_soon = len(filtered_df[filtered_df['days_to_expiry'] <= 180])
            st.metric("Expiring Soon", expiring_soon)
    else:
        st.info("No contracts match the selected filters.")
