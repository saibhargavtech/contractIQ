#!/usr/bin/env python3
"""
Test script for dual-mode dashboard functionality
"""

import os
import sys
import pandas as pd

# Add the current directory to Python path
sys.path.append('.')

def test_demo_mode():
    """Test demo mode functionality"""
    print("🧪 Testing Demo Mode...")
    
    # Check if dummy contracts exist
    if os.path.exists("dummy_contracts_50.csv"):
        df = pd.read_csv("dummy_contracts_50.csv")
        print(f"✅ Demo contracts found: {len(df)} contracts")
        print(f"📊 Contract types: {df['type'].value_counts().to_dict()}")
        print(f"💰 Total value: ${df['total_value_usd'].sum():,.0f}")
        return True
    else:
        print("❌ Demo contracts not found")
        return False

def test_live_mode():
    """Test live mode functionality"""
    print("🧪 Testing Live Mode...")
    
    # Check if uploaded contracts exist
    if os.path.exists("uploaded_contracts.csv"):
        df = pd.read_csv("uploaded_contracts.csv")
        print(f"✅ Uploaded contracts found: {len(df)} contracts")
        if len(df) > 0:
            print(f"📤 Contract IDs: {df['contract_id'].tolist()}")
            print(f"💰 Total value: ${df['total_value_usd'].sum():,.0f}")
        return True
    else:
        print("❌ No uploaded contracts found")
        return False

def test_graphrag_data():
    """Test GraphRAG data availability"""
    print("🧪 Testing GraphRAG Data...")
    
    checks = [
        ("entities.csv", "Entity data"),
        ("graph_export.ttl", "Graph export"),
        ("cfo_contract_insights.jsonl", "CFO insights")
    ]
    
    results = {}
    for filename, description in checks:
        exists = os.path.exists(filename)
        results[filename] = exists
        status = "✅" if exists else "❌"
        print(f"{status} {description}: {filename}")
    
    return any(results.values())

def main():
    """Run all tests"""
    print("🚀 Testing Dual-Mode Dashboard Components\n")
    
    # Test demo mode
    demo_success = test_demo_mode()
    print()
    
    # Test live mode  
    live_success = test_live_mode()
    print()
    
    # Test GraphRAG data
    graphrag_success = test_graphrag_data()
    print()
    
    # Summary
    print("📋 Test Summary:")
    print(f"{'✅' if demo_success else '❌'} Demo Mode")
    print(f"{'✅' if live_success else '❌'} Live Mode") 
    print(f"{'✅' if graphrag_success else '❌'} GraphRAG Data")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if demo_success:
        print("• Demo mode ready - dashboard can show sample data")
    if live_success:
        print("• Live mode ready - dashboard can show uploaded contracts")
    if graphrag_success:
        print("• GraphRAG analysis available - advanced insights possible")
    
    if not live_success:
        print("• Upload documents in Gradio to enable Live Mode")
    
    if not graphrag_success:
        print("• Run GraphRAG processing to enable advanced analytics")

if __name__ == "__main__":
    main()
