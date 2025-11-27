"""
ContractIQ Main Dashboard
Main entry point for the ContractIQ dashboard application
"""

import streamlit as st
import pandas as pd
from typing import Dict

# Import page modules (moved from pages/ to modules/ to disable default navigation)
from modules import insightiq, portfolio, cashflow, cost_control, executive, development_centre

# Import from frontend/utils.py (not root utils.py)
# Fix import conflict: ensure we import from frontend directory, not root
import sys
import os
# Get the directory where this file is located (frontend/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure frontend directory is first in path (before parent directory)
if current_dir in sys.path:
    sys.path.remove(current_dir)
sys.path.insert(0, current_dir)
# Now import utils (will get frontend/utils.py)
from utils import load_cfo_analytics_data

# Set page config - CRITICAL: Disable default navigation
st.set_page_config(
    page_title="ContractIQ Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# CRITICAL: Hide Streamlit's default page navigation completely

# Apply aggressive CSS immediately after page config to prevent default navigation
st.markdown("""
<style>
    /* AGGRESSIVE CSS - Prevent default Streamlit navigation from rendering */
    .stDeployButton, .stDeployButton > div, [data-testid="stDeployButton"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Hide hamburger menu */
    .stActionButton, .stActionButton > div, [data-testid="stActionButton"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Hide main menu */
    .stMainMenu, .stMainMenu > div, [data-testid="stMainMenu"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Hide header elements */
    .stHeader, .stHeader > div, [data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Hide any default navigation */
    .stNavigation, .stNavigation > div, [data-testid="stNavigation"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* HIDE DEFAULT STREAMLIT SIDEBAR NAVIGATION - MOST AGGRESSIVE */
    .css-1d391kg .css-1cypcdb, 
    .css-1d391kg [data-testid="stSidebarNav"],
    .css-1d391kg .stSidebarNav,
    .css-1d391kg .stSidebarNav > div,
    .css-1d391kg .stSidebarNav > ul,
    .css-1d391kg .stSidebarNav > li,
    .css-1d391kg .stSidebarNav a,
    .css-1d391kg .stSidebarNav button {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Hide all default sidebar content except our custom content */
    .css-1d391kg > div:not(.stMarkdown):not(.stButton) {
        display: none !important;
    }
    
    /* Force hide with JavaScript - MOST AGGRESSIVE */
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setInterval(function() {
            // Hide all default Streamlit navigation elements
            var elements = document.querySelectorAll(
                '[data-testid="stDeployButton"], [data-testid="stActionButton"], [data-testid="stMainMenu"], [data-testid="stHeader"], [data-testid="stNavigation"], [data-testid="stSidebarNav"], .stSidebarNav, .css-1cypcdb'
            );
            elements.forEach(function(el) {
                el.style.display = 'none';
                el.style.visibility = 'hidden';
                el.style.opacity = '0';
                el.style.pointerEvents = 'none';
                el.style.height = '0';
                el.style.width = '0';
                el.style.margin = '0';
                el.style.padding = '0';
            });
            
            // Hide any sidebar navigation links
            var navLinks = document.querySelectorAll('.css-1d391kg a, .css-1d391kg button');
            navLinks.forEach(function(link) {
                if (link.textContent && (link.textContent.includes('main dashboard') || link.textContent.includes('cashflow') || link.textContent.includes('chatbot'))) {
                    link.style.display = 'none';
                    link.style.visibility = 'hidden';
                    link.style.opacity = '0';
                    link.style.pointerEvents = 'none';
                }
            });
        }, 50);
    });
    </script>
</style>
""", unsafe_allow_html=True)

# Custom CSS for ContractIQ styling - Consolidated and Stable
st.markdown("""
<style>
    /* Stable layout - prevent shrinking */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Stable sidebar */
    .css-1d391kg {
        width: 21rem !important;
    }
    
    /* Prevent content shrinking */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        margin-top: 0rem;
    }
    
    /* ContractIQ header styling */
    h1 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* KPI Cards - Stable styling */
    .cfo-metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.2;
    }
    
    /* Risk styling */
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
    
    /* Insight boxes */
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    
    /* Stable columns */
    .stColumns > div {
        min-width: 0;
    }
    
    /* Compact sidebar */
    .css-1d391kg {
        overflow-y: auto !important;
        max-height: 100vh !important;
        padding-top: 0.5rem !important;
    }
    
    /* Reduce spacing in sidebar */
    .css-1d391kg .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact navigation buttons */
    .css-1d391kg button {
        margin-bottom: 0.3rem !important;
        padding: 0.5rem !important;
    }
    
    /* Prevent KPI card shrinking - AGGRESSIVE */
    .cfo-metric-card {
        min-width: 300px !important;
        width: 100% !important;
        max-width: none !important;
        flex-shrink: 0 !important;
        flex-grow: 1 !important;
        box-sizing: border-box !important;
    }
    .metric-value {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        width: 100% !important;
        font-size: 2.5rem !important;
    }
    .metric-label {
        white-space: nowrap !important;
        width: 100% !important;
    }
    
    /* Force column stability */
    .stColumns > div {
        min-width: 300px !important;
        flex-shrink: 0 !important;
    }
    
    /* Disable command palette completely */
    .stCommandPalette,
    [data-testid="stCommandPalette"],
    .stCommandPaletteOverlay,
    .stCommandPaletteContainer {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Prevent command palette from showing */
    body:has(.stCommandPalette) .stCommandPalette {
        display: none !important;
    }
    
    /* Force hide command palette with JavaScript */
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setInterval(function() {
            var palette = document.querySelector('[data-testid="stCommandPalette"]');
            if (palette) palette.style.display = 'none';
        }, 100);
    });
    </script>
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    # Ensure CSS is always loaded - add it at the start of main function
    st.markdown("""
    <style>
        .cfo-metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            padding: 1.5rem !important;
            border-radius: 0.8rem !important;
            color: white !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            margin-bottom: 1rem !important;
            min-height: 120px !important;
            min-width: 300px !important;
            width: 100% !important;
            max-width: none !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            flex-shrink: 0 !important;
            flex-grow: 1 !important;
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
        }
        .metric-label {
            font-size: 1rem !important;
            opacity: 0.9 !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state consistently
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "InsightIQ"
    
    # Load analytics data once and cache it
    # Clear cache if contracts were just uploaded
    if st.session_state.get('upload_success', False):
        from utils import load_cfo_jsonl_insights, load_unified_contracts
        load_cfo_analytics_data.clear()
        load_cfo_jsonl_insights.clear()  # Also clear insights cache
        load_unified_contracts.clear()  # Also clear contracts cache
        st.session_state['upload_success'] = False
    
    with st.spinner("Loading ContractIQ analytics..."):
        analytics_data = load_cfo_analytics_data("unified")
    
    if analytics_data["contract_csv"] is None:
        st.error("No contract data available")
        return
    
    # Navigation sidebar with ContractIQ branding - Compact
    st.sidebar.markdown("# 🧠 ContractIQ")
    st.sidebar.markdown("*Clarity, Compliance, and Control*")
    st.sidebar.markdown("---")
    
    # Make sidebar scrollable and compact
    st.sidebar.markdown("""
    <style>
    .css-1d391kg {
        overflow-y: auto !important;
        max-height: 100vh !important;
    }
    .css-1d391kg > div {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation menu - clickable buttons with consistent styling
    pages = [
        ("🧠 InsightIQ", "InsightIQ"),
        ("📊 Portfolio Management", "Portfolio Management"),
        ("💰 Cash Flow Analysis", "Cash Flow Analysis"),
        ("💸 Cost Control & Risk", "Cost Control & Risk Management"),
        ("📈 Executive Insights", "Executive Insights"),
        ("💬 Talk to Your Contracts", "Contract Chatbot"),
        ("🔧 Development Centre", "Development Centre")
    ]
    
    for button_text, page_name in pages:
        if st.sidebar.button(button_text, use_container_width=True, 
                           type="primary" if st.session_state['current_page'] == page_name else "secondary"):
            st.session_state['current_page'] = page_name
            st.rerun()  # Force rerun to update the page
    
    
    # Route to different pages based on navigation
    current_page = st.session_state['current_page']
    
    if current_page == "InsightIQ":
        insightiq.create_page(analytics_data)
    elif current_page == "Portfolio Management":
        portfolio.create_page(analytics_data)
    elif current_page == "Cash Flow Analysis":
        cashflow.create_page(analytics_data)
    elif current_page == "Cost Control & Risk Management":
        cost_control.create_page(analytics_data)
    elif current_page == "Executive Insights":
        executive.create_page(analytics_data)
    elif current_page == "Contract Chatbot":
        from modules import chatbot
        chatbot.create_page(analytics_data)
    elif current_page == "Development Centre":
        development_centre.create_page(analytics_data)

if __name__ == "__main__":
    main()
