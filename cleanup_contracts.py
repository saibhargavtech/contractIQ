#!/usr/bin/env python3
"""
Contract Portfolio Cleanup Script
Automatically maintains exactly 5 contracts for fast processing
"""

import pandas as pd
import os
from datetime import datetime

def cleanup_contract_portfolio(max_contracts=5):
    """
    Clean up contract portfolio to maintain exactly max_contracts
    Keeps the first max_contracts and removes the rest
    """
    
    print("🧹 Contract Portfolio Cleanup")
    print("=" * 40)
    
    # Check if uploaded_contracts.csv exists
    if not os.path.exists("uploaded_contracts.csv"):
        print("❌ No uploaded_contracts.csv found")
        return False
    
    try:
        # Load current contracts
        df = pd.read_csv("uploaded_contracts.csv")
        current_count = len(df)
        
        print(f"📊 Current contracts: {current_count}")
        
        if current_count <= max_contracts:
            print(f"✅ Portfolio already at or below {max_contracts} contracts - no cleanup needed")
            return True
        
        # Keep only the first max_contracts
        df_cleaned = df.head(max_contracts)
        
        # Create backup of original
        backup_filename = f"uploaded_contracts_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_filename, index=False)
        print(f"💾 Backup created: {backup_filename}")
        
        # Save cleaned portfolio
        df_cleaned.to_csv("uploaded_contracts.csv", index=False)
        
        print(f"✅ Cleaned portfolio: {len(df_cleaned)} contracts")
        print(f"📋 Contract IDs: {df_cleaned['contract_id'].tolist()}")
        
        # Show portfolio summary
        total_value = df_cleaned['total_value_usd'].sum()
        total_annual = df_cleaned['annual_commitment_usd'].sum()
        
        print(f"\n📊 Portfolio Summary:")
        print(f"   Total Value: ${total_value:,.0f}")
        print(f"   Annual Commitments: ${total_annual:,.0f}")
        print(f"   Average Contract: ${total_value/len(df_cleaned):,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return False

def show_portfolio_status():
    """Show current portfolio status"""
    
    if not os.path.exists("uploaded_contracts.csv"):
        print("❌ No uploaded_contracts.csv found")
        return
    
    try:
        df = pd.read_csv("uploaded_contracts.csv")
        
        print("📊 Current Portfolio Status")
        print("=" * 30)
        print(f"Total Contracts: {len(df)}")
        print(f"Total Value: ${df['total_value_usd'].sum():,.0f}")
        print(f"Annual Commitments: ${df['annual_commitment_usd'].sum():,.0f}")
        print(f"Contract IDs: {df['contract_id'].tolist()}")
        
    except Exception as e:
        print(f"❌ Error reading portfolio: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_portfolio_status()
    else:
        cleanup_contract_portfolio()
        
        print("\n🚀 Ready for fast processing!")
        print("💡 Run 'python cleanup_contracts.py status' to check portfolio")
        print("💡 Run 'python cleanup_contracts.py' to clean up again")
















