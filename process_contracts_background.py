"""
Background Contract Processing Script
Processes contracts using the contract extraction function from promptrefreshed.py
"""

import os
import sys
import pandas as pd
from typing import List, Tuple

# Import the contract extraction function
import promptrefreshed as pr

def process_contracts_to_graph():
    """Process contracts using the contract extraction function"""
    
    print("🚀 Starting contract processing pipeline...")
    
    # Load contract data
    contract_file = "new_contracts_only.csv"
    if not os.path.exists(contract_file):
        print(f"❌ Contract file {contract_file} not found!")
        return False
    
    print(f"📄 Loading contracts from {contract_file}...")
    df = pd.read_csv(contract_file)
    print(f"✅ Loaded {len(df)} contracts")
    
    # Convert contract data to text format for processing
    contract_texts = []
    for _, row in df.iterrows():
        contract_text = f"""
        Contract ID: {row['contract_id']}
        Type: {row['type']}
        Counterparty: {row['counterparty']}
        Start Date: {row['start_date']}
        End Date: {row['end_date']}
        Status: {row['status']}
        Total Value: ${row['total_value_usd']:,.0f}
        Annual Commitment: ${row['annual_commitment_usd']:,.0f}
        Escalation: {row['escalation']}
        Payment Terms: {row['payment_terms']}
        Variable Costs: {row['variable_costs']}
        SLA Uptime: {row['sla_uptime']}
        SLA Response Critical: {row['sla_response_critical']}
        SLA Resolution Critical: {row['sla_resolution_critical']}
        SLA Penalty: {row['sla_penalty']}
        Compliance: {row['compliance']}
        ESG Reporting: {row['esg_reporting']}
        Termination: {row['termination']}
        Governing Law: {row['governing_law']}
        Arbitration: {row['arbitration']}
        Summary: {row.get('summary', 'N/A')}
        """
        contract_texts.append((f"Contract_{row['contract_id']}", contract_text))
    
    print("🔄 Processing contracts through contract extraction pipeline...")
    
    # Create temporary files for processing
    temp_dir = "temp_contracts"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Write contract texts to temporary files
    upload_paths = []
    for i, (name, text) in enumerate(contract_texts):
        temp_file = os.path.join(temp_dir, f"{name}.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(text)
        upload_paths.append(temp_file)
    
    try:
        # Process using the contract extraction function
        print("🧠 Building knowledge graph using contract extraction...")
        
        # Call the contract extraction function
        result = pr.process_corpus_with_contract_extraction(
            upload_paths=upload_paths,
            dir_path=temp_dir,
            type_choices=["txt"],
            enable_vector=True,
            vector_topk=8,
            extract_temp=0.1,
            strict_prompt=False,
            require_evidence=True,
            enable_fallback=False,
            max_coocc_pairs=5,
            chunk_size=2500,
            relation_policy="Standard",
            alias_norm=True,
            skip_no_evidence=True,
            louvain_resolution=1.0,
            louvain_random_state=42,
            use_onto_cluster_cb=False
        )
        
        print("✅ Contract extraction processing completed!")
        
        # Check if graph was built successfully
        if hasattr(pr, 'G') and pr.G is not None:
            print(f"📊 Graph built with {pr.G.number_of_nodes()} nodes and {pr.G.number_of_edges()} edges")
            
            # Check if context memory was generated
            if hasattr(pr, 'GRAPH_CONTEXT_MEMORY') and pr.GRAPH_CONTEXT_MEMORY:
                print("🧠 Graph context memory generated successfully!")
                print(f"📝 Context memory length: {len(pr.GRAPH_CONTEXT_MEMORY)} characters")
                
                # Save context memory to file for the dashboard to use
                with open("graph_context_memory.txt", "w", encoding="utf-8") as f:
                    f.write(pr.GRAPH_CONTEXT_MEMORY)
                print("💾 Context memory saved to graph_context_memory.txt")
                
                return True
            else:
                print("⚠️ Graph built but no context memory generated")
                return False
        else:
            print("❌ Graph was not built successfully")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temporary files")

def main():
    """Main function to process contracts"""
    print("=" * 60)
    print("CONTRACT PROCESSING PIPELINE")
    print("=" * 60)
    
    success = process_contracts_to_graph()
    
    if success:
        print("\n🎉 SUCCESS! Contracts processed and knowledge graph built!")
        print("💬 You can now ask questions in the ContractIQ chatbot!")
        print("\n📋 Next steps:")
        print("1. Go to ContractIQ dashboard")
        print("2. Navigate to 'Talk to Your Contracts'")
        print("3. The knowledge graph should be ready!")
    else:
        print("\n❌ FAILED! Contract processing was not successful.")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
