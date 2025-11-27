"""
Cash Flow Analysis Page
Financial analysis with CFO-focused cash flow impact
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List

def create_page(analytics_data: Dict):
    """Create Cash Flow Analysis page"""
    df = analytics_data["contract_csv"]
    cfo_insights = analytics_data["cfo_insights"]
    
    st.markdown("## 💰 Cash Flow Analysis")
    
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
        if not df.empty:
            fig = px.histogram(
                df,
                x='annual_commitment_usd',
                title="Distribution of Annual Commitments",
                labels={'annual_commitment_usd': 'Annual Commitment (USD)', 'count': 'Number of Contracts'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No contract data available")
    
    # Payment terms analysis
    st.markdown("## 💳 Payment Terms Analysis")
    
    # Payment terms breakdown
    payment_terms_breakdown = df['payment_terms'].value_counts()
    
    fig = px.pie(
        payment_terms_breakdown,
        values=payment_terms_breakdown.values,
        names=payment_terms_breakdown.index,
        title="Payment Terms Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cash flow timeline
    st.markdown("## 📅 Cash Flow Timeline")
    
    # Create timeline data
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
    
    if not timeline_df.empty:
        fig = px.timeline(
            timeline_df.head(20),  # Show top 20 for readability
            x_start="Start", 
            x_end="End", 
            y="Contract",
            color="Value",
            title="Contract Timeline (Top 20 by Value)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeline data available")
    
    # Unfavorable payment terms table
    st.markdown("## ⚠️ Contracts with Unfavorable Payment Terms")
    if not unfavorable_terms.empty:
        unfavorable_display = unfavorable_terms[['contract_id', 'counterparty', 'payment_terms', 'annual_commitment_usd']].copy()
        unfavorable_display['annual_commitment_usd'] = unfavorable_display['annual_commitment_usd'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(unfavorable_display, use_container_width=True)
    else:
        st.success("✅ No contracts with unfavorable payment terms!")



