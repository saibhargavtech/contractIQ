"""
Launch script for the CFO Contract Dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install requirements if needed"""
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ All required packages are already installed")
        return True
    except ImportError:
        print("📦 Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Packages installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    if not install_requirements():
        return
    
    print("🚀 Launching CFO Contract Dashboard...")
    print("📊 Dashboard will open in your browser at http://localhost:8501")
    print("🔧 Make sure the dummy_contracts_50.csv file is in the parent directory")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "cfo_dashboard.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")

if __name__ == "__main__":
    run_dashboard()





