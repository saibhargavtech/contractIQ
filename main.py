"""
Streamlit Cloud Entry Point
This file allows deployment from the root directory
"""

import sys
import os

# Add frontend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frontend'))

# Import and run the main dashboard
from frontend.main_dashboard import main

if __name__ == "__main__":
    main()
