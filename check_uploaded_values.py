#!/usr/bin/env python3
"""
Check if uploaded document values are included in CFO export
"""
import pandas as pd

def check_uploaded_values():
    print("=== CHECKING CSV FILES ===")
    
    # Check uploaded_contracts.csv
    df1 = pd.read_csv('uploaded_contracts.csv')
    print(f"uploaded_contracts.csv: {len(df1)} rows")
    total1 = df1['total_value_usd'].sum()
    print(f"Total value: ${total1:,.2f}")
    print(f"Sample values: {df1['total_value_usd'].head(3).tolist()}")
    
    # Check new_contracts_only.csv  
    df2 = pd.read_csv('new_contracts_only.csv')
    print(f"\nnew_contracts_only.csv: {len(df2)} rows")
    total2 = df2['total_value_usd'].sum()
    print(f"Total value: ${total2:,.2f}")
    if len(df2) > 0:
        print(f"Sample values: {df2['total_value_usd'].head(3).tolist()}")
    
    print(f"\n=== MERGED TOTAL ===")
    print(f"uploaded_contracts: ${total1:,.2f}")
    print(f"new_contracts_only: ${total2:,.2f}")
    print(f"Combined: ${total1 + total2:,.2f}")
    
    print(f"\n=== SIMULATING EXPORT LOGIC ===")
    # Simulate what the export function does
    try:
        contract_df = pd.read_csv('uploaded_contracts.csv')
        print(f"Step 1: Load uploaded_contracts.csv -> {len(contract_df)} contracts, ${contract_df['total_value_usd'].sum():,.2f}")
        
        new_df = pd.read_csv('new_contracts_only.csv')
        print(f"Step 2: Load new_contracts_only.csv -> {len(new_df)} contracts, ${new_df['total_value_usd'].sum():,.2f}")
        
        # Merge logic
        existing_ids = set(contract_df['contract_id'].tolist()) if 'contract_id' in contract_df.columns else set()
        new_contracts = new_df[~new_df['contract_id'].isin(existing_ids)]
        print(f"Step 3: Find unique new contracts -> {len(new_contracts)} contracts")
        
        if not new_contracts.empty:
            contract_df = pd.concat([contract_df, new_contracts], ignore_index=True)
            print(f"Step 4: Merge -> {len(contract_df)} total contracts, ${contract_df['total_value_usd'].sum():,.2f}")
        
        print(f"\nFINANCIAL SUMMARY:")
        print(f"- Total contracts: {len(contract_df)}")
        print(f"- Total contract value: ${contract_df['total_value_usd'].sum():,.2f}")
        print(f"- Annual commitment: ${contract_df['annual_commitment_usd'].sum():,.2f}")
        print(f"- Average contract value: ${contract_df['total_value_usd'].mean():,.2f}")
        
        # Check if any recent contracts (check by contract_id pattern)
        if 'contract_id' in contract_df.columns:
            recent_contracts = contract_df[contract_df['contract_id'].str.contains('C-', na=False)]
            print(f"- Recent contract IDs found: {len(recent_contracts)}")
            print(f"- Sample IDs: {recent_contracts['contract_id'].head(5).tolist()}")
            
    except Exception as e:
        print(f"Error in simulation: {e}")

if __name__ == "__main__":
    check_uploaded_values()
