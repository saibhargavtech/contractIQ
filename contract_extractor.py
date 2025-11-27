"""
Contract Data Extraction and Processing
Extracts structured contract data from uploaded documents to match CSV format
"""

import re
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

import config
import utils
from extraction import chat_strict
import re
import json

# Ensure json_safe_load is available - add it if missing
def safe_json_load(raw: str):
    """Safely load JSON with cleanup - works regardless of utils module"""
    if not raw:
        return None
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(raw)
    except Exception:
        return None

# Try to use utils.json_safe_load if available, otherwise use our fallback
if hasattr(utils, 'json_safe_load'):
    json_safe_load = utils.json_safe_load
else:
    json_safe_load = safe_json_load
    # Also add it to utils for future use
    utils.json_safe_load = safe_json_load

# Set OpenAI API key
import openai
openai.api_key = config.OPENAI_API_KEY

# ==================== Contract Data Extraction ====================

def extract_contract_data_from_text(text: str, filename: str) -> Dict[str, Any]:
    """
    Extract structured contract data from text to match CSV format
    Returns a dictionary with all required CSV columns
    """
    
    # Generate unique contract ID
    contract_id = f"C-{str(uuid.uuid4())[:4].upper()}"
    
    # Create extraction prompt
    extraction_prompt = f"""
You are a contract data extraction expert. Extract structured data from the following contract text.

Contract Text:
\"\"\"{text[:5000]}\"\"\"

Extract the following information and return as JSON:

{{
    "contract_id": "{contract_id}",
    "type": "Contract type (e.g., SaaS Licensing, Service Outsourcing, Procurement, Insurance, Partnership, Employment, IP Licensing)",
    "counterparty": "Company/organization name (the other party)",
    "start_date": "Start date in YYYY-MM-DD format",
    "end_date": "End date in YYYY-MM-DD format", 
    "status": "Status (Active, Expired, Terminated, Renewed)",
"total_value_usd": "Total contract value in USD (CAREFULLY scan entire document for: 'total contract value', 'maximum contract value', 'not to exceed', 'up to', 'budget allocation', 'total investment' - extract the LARGEST monetary figure found, convert all currencies to USD)",
    "annual_commitment_usd": "Annual commitment in USD (CAREFULLY scan for: 'annual commitment', 'annual fee', 'yearly cost', 'annual spend', 'yearly commitment', 'annual budget', 'per annum', 'per year' - extract the explicit annual figure, NOT daily/monthly amounts)",
    "escalation": "Escalation terms (e.g., '3% YoY', '2% YoY', 'None')",
    "payment_terms": "Payment terms (e.g., 'Net 30', 'Net 45', 'Net 60', 'Milestone-based')",
    "variable_costs": "Variable cost structure (e.g., 'Performance bonus structure', '2% overage on usage', 'None')",
    "sla_uptime": "SLA uptime target (e.g., '99.5%', '99.9%', '99.99%')",
    "sla_response_critical": "Critical response time SLA (e.g., '30 min', '1 hr', '2 hrs')",
    "sla_resolution_critical": "Critical resolution time SLA (e.g., '4 hrs', '8 hrs', '24 hrs')",
    "sla_penalty": "SLA penalty structure (e.g., '5% penalty per missed critical resolution', '2% fee credit per 0.1% drop')",
    "compliance": "Compliance requirements (comma-separated, e.g., 'FCPA, GDPR, CCPA, SOC2')",
    "esg_reporting": "ESG reporting requirements (e.g., 'Annual (GRI aligned)', 'Quarterly ESG report', 'Carbon Disclosure only', 'None required')",
    "termination": "Termination clause details (e.g., '90 days mutual notice required', '30 days cure for cause; 120 days convenience')",
    "governing_law": "Governing law/jurisdiction (e.g., 'US', 'UK', 'India', 'Singapore')",
    "arbitration": "Arbitration details (e.g., 'Arbitration India', 'ICC Singapore', 'SIAC Singapore')",
    "summary": "Key contract highlights as bullet points:
- Contract Type: [Extract main contract category]
- Counterparty: [Company name and relationship]
- Total Value: [Contract value with currency]
- Duration: [Start date to End date]
- Payment Structure: [Payment terms and frequency]
- Key SLAs: [Critical SLA requirements]
- Compliance Requirements: [Regulatory/framework compliance]
- Termination Terms: [Notice periods and conditions]
- Risk Factors: [Any notable risks or concerns]
- Additional Highlights: [Other important contract details]", 
    "source_file": "{filename}",
    "extraction_date": "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "extraction_status": "success"
}}

Guidelines:
- If information is not available, use "Not specified" or "None"
- For dates, use YYYY-MM-DD format
- For monetary values, extract actual numbers from text (look for $, USD, INR, amounts, fees, costs, prices, lakhs, crores, millions, thousands)
- For percentages, include the % symbol
- CONTRACT TYPE: Look for explicit mentions of "licensing", "outsourcing", "procurement", "insurance", "partnership", "employment", "SaaS", "software", "service agreement", "supply agreement"
- COUNTERPARTY: Extract the exact company/organization name (usually in header, signature, or "between X and Y" clauses)
- DATES: Convert all date formats to YYYY-MM-DD (e.g., "March 15, 2023" → "2023-03-15")
- VALUE EXTRACTION: CRITICAL - Look for "total contract value", "maximum contract value", "not to exceed", "up to", "budget allocation", "total investment" - Find the LARGEST monetary figure mentioned - convert currencies to USD ($ = USD)
- SLA TERMS: Extract specific numbers for uptime (99.x%), response times (X minutes/hours), resolution times (X hours/days)
- COMPLIANCE: Look for "compliance with", "subject to", "in accordance with" followed by regulation names (GDPR, SOX, HIPAA, etc.)
- TERMINATION: Extract notice periods, cure periods, breach conditions
- GOVERNING LAW: Look for "governed by", "subject to laws of", "jurisdiction", "courts of" clauses
- CRITICAL: Extract ONLY exact information found in the text. If not explicitly stated, use "Not specified"
"""

    try:
        response = chat_strict(
            [{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            timeout=60,
            model="gpt-4"
        )
        
        # Parse JSON response
        parsed_data = utils.json_safe_load(response)
        
        if not parsed_data:
            return create_default_contract_data(contract_id, filename)
        
        # Validate and clean the data
        cleaned_data = validate_and_clean_contract_data(parsed_data, filename)
        return cleaned_data
        
    except Exception as e:
        print(f"[Contract Extractor] Error extracting data from {filename}: {e}")
        return create_default_contract_data(contract_id, filename)

def create_default_contract_data(contract_id: str, filename: str) -> Dict[str, Any]:
    """Create default contract data when extraction fails"""
    return {
        "contract_id": contract_id,
        "type": "Not specified",
        "counterparty": "Not specified",
        "start_date": "2024-01-01",
        "end_date": "2025-01-01",
        "status": "Active",
        "total_value_usd": 0,
        "annual_commitment_usd": 0,
        "escalation": "Not specified",
        "payment_terms": "Not specified",
        "variable_costs": "None",
        "sla_uptime": "Not specified",
        "sla_response_critical": "Not specified",
        "sla_resolution_critical": "Not specified",
        "sla_penalty": "Not specified",
        "compliance": "Not specified",
        "esg_reporting": "Not specified",
        "termination": "Not specified",
        "governing_law": "Not specified",
        "arbitration": "Not specified",
        "summary": "- Contract extracted from: " + filename + "\n- Default template applied due to extraction failure\n- Please review contract manually for accuracy",
        "source_file": filename,
        "extraction_date": datetime.now().isoformat(),
        "extraction_status": "Failed - using defaults"
    }

def validate_and_clean_contract_data(data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Validate and clean extracted contract data"""
    
    # Ensure all required fields exist
    required_fields = [
        "contract_id", "type", "counterparty", "start_date", "end_date", "status",
        "total_value_usd", "annual_commitment_usd", "escalation", "payment_terms",
        "variable_costs", "sla_uptime", "sla_response_critical", "sla_resolution_critical",
        "sla_penalty", "compliance", "esg_reporting", "termination", "governing_law", "arbitration",
        "summary", "source_file", "extraction_date", "extraction_status"
    ]
    
    cleaned_data = {}
    
    for field in required_fields:
        value = data.get(field, "Not specified")
        
        # Clean and validate specific fields
        if field in ["total_value_usd", "annual_commitment_usd"]:
            if isinstance(value, (int, float)):
                cleaned_data[field] = int(value)
            else:
                # Extract full number from string (handle commas, millions, etc.)
                value_str = str(value).replace(",", "").replace(" ", "")
                
                # Handle different formats: $1,250,000 USD -> 1250000
                if "million" in value_str.lower() or value_str.lower().endswith('m'):
                    numbers = re.findall(r'\d+\.?\d*', value_str)
                    if numbers:
                        cleaned_data[field] = int(float(numbers[0]) * 1000000)
                    else:
                        cleaned_data[field] = 0
                elif "thousand" in value_str.lower() or "k" in value_str.lower():
                    numbers = re.findall(r'\d+\.?\d*', value_str)
                    if numbers:
                        cleaned_data[field] = int(float(numbers[0]) * 1000)
                    else:
                        cleaned_data[field] = 0
                else:
                    # Extract the largest complete number
                    numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', value_str)
                    if numbers:
                        # Remove commas and convert to int
                        cleaned_data[field] = int(float(numbers[-1].replace(",", "")))
                    else:
                        cleaned_data[field] = 0
        
        elif field in ["start_date", "end_date"]:
            # Validate date format
            if re.match(r'\d{4}-\d{2}-\d{2}', str(value)):
                cleaned_data[field] = value
            else:
                # Try to extract date from text
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', str(value))
                if date_match:
                    cleaned_data[field] = date_match.group(1)
                else:
                    cleaned_data[field] = "2024-01-01"  # Default date
        
        elif field == "status":
            # Normalize status
            status = str(value).lower()
            if "active" in status:
                cleaned_data[field] = "Active"
            elif "expired" in status:
                cleaned_data[field] = "Expired"
            elif "terminated" in status:
                cleaned_data[field] = "Terminated"
            elif "renewed" in status:
                cleaned_data[field] = "Renewed"
            else:
                cleaned_data[field] = "Active"  # Default
        
        else:
            # Clean string fields
            cleaned_data[field] = str(value).strip() if value else "Not specified"
    
    # Add metadata
    cleaned_data["source_file"] = filename
    cleaned_data["extraction_date"] = datetime.now().isoformat()
    cleaned_data["extraction_status"] = "Success"
    
    return cleaned_data

def process_uploaded_contracts(corpus_pairs: List[tuple]) -> pd.DataFrame:
    """
    Process uploaded contract files and extract structured data
    Returns DataFrame with same format as dummy_contracts_50.csv
    """
    
    extracted_contracts = []
    
    for filename, text in corpus_pairs:
        print(f"[Contract Extractor] Processing {filename}...")
        
        # Extract contract data
        contract_data = extract_contract_data_from_text(text, filename)
        extracted_contracts.append(contract_data)
    
    # Convert to DataFrame
    if extracted_contracts:
        df = pd.DataFrame(extracted_contracts)
        return df
    else:
        return pd.DataFrame()

def merge_with_existing_contracts(new_contracts_df: pd.DataFrame, existing_csv_path: str = "uploaded_contracts.csv") -> pd.DataFrame:
    """
    Merge new contracts with ONLY quantified contracts from uploaded_contracts.csv
    SKIPS dummy contracts for faster processing - uses only quantified contract data
    Returns combined DataFrame with quantified + new contracts only
    """
    
    combined_df = pd.DataFrame()
    
    # Load ONLY quantified contracts from uploaded_contracts.csv (skip dummy contracts)
    try:
        if os.path.exists("uploaded_contracts.csv"):
            previous_df = pd.read_csv("uploaded_contracts.csv")
            combined_df = pd.concat([combined_df, previous_df], ignore_index=True)
            print(f"[Contract Extractor] 📊 QUANTIFIED: Using {len(previous_df)} quantified contracts from uploaded_contracts.csv")
        else:
            print(f"[Contract Extractor] 📝 INFO: No uploaded_contracts.csv found - starting fresh")
    except Exception as e:
        print(f"[Contract Extractor] ❌ Could not load quantified contracts: {e}")
    
    # Add new uploaded contracts
    if not new_contracts_df.empty:
        combined_df = pd.concat([combined_df, new_contracts_df], ignore_index=True)
        print(f"[Contract Extractor] 🆕 NEW: Added {len(new_contracts_df)} newly uploaded contracts")
        print(f"[Contract Extractor] Uploaded contracts: {new_contracts_df['contract_id'].tolist()}")
    
    print(f"[Contract Extractor] 🎯 TOTAL PORTFOLIO: {len(combined_df)} contracts (foundation + previous + new)")
    return combined_df

def save_combined_contracts(combined_df: pd.DataFrame, output_path: str = "uploaded_contracts.csv") -> str:
    """Save combined contracts (dummy + uploaded) to CSV file"""
    
    try:
        combined_df.to_csv(output_path, index=False)
        print(f"[Contract Extractor] Saved {len(combined_df)} total contracts to {output_path}")
        return output_path
    except Exception as e:
        print(f"[Contract Extractor] Error saving contracts: {e}")
        return ""

def save_new_contracts_only(new_contracts_df: pd.DataFrame, output_path: str = "new_contracts_only.csv") -> str:
    """Save ONLY new contracts (no dummy merge) - FAST mode"""
    
    try:
        new_contracts_df.to_csv(output_path, index=False)
        print(f"[Contract Extractor] FAST SAVE: {len(new_contracts_df)} new contracts to {output_path}")
        return output_path
    except Exception as e:
        print(f"[Contract Extractor] Error saving new contracts: {e}")
        return ""

def generate_contract_specific_cfo_insights(contract_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate CFO insights for a specific contract
    Returns list of insights for all 30 CFO questions
    """
    
    contract_context = f"""
Contract ID: {contract_data.get('contract_id', 'Unknown')}
Type: {contract_data.get('type', 'Not specified')}
Counterparty: {contract_data.get('counterparty', 'Not specified')}
Start Date: {contract_data.get('start_date', 'Not specified')}
End Date: {contract_data.get('end_date', 'Not specified')}
Status: {contract_data.get('status', 'Not specified')}
Total Value: ${contract_data.get('total_value_usd', 0):,}
Annual Commitment: ${contract_data.get('annual_commitment_usd', 0):,}
Escalation: {contract_data.get('escalation', 'Not specified')}
Payment Terms: {contract_data.get('payment_terms', 'Not specified')}
Variable Costs: {contract_data.get('variable_costs', 'Not specified')}
SLA Uptime: {contract_data.get('sla_uptime', 'Not specified')}
SLA Response: {contract_data.get('sla_response_critical', 'Not specified')}
SLA Resolution: {contract_data.get('sla_resolution_critical', 'Not specified')}
SLA Penalty: {contract_data.get('sla_penalty', 'Not specified')}
Compliance: {contract_data.get('compliance', 'Not specified')}
ESG Reporting: {contract_data.get('esg_reporting', 'Not specified')}
Termination: {contract_data.get('termination', 'Not specified')}
Governing Law: {contract_data.get('governing_law', 'Not specified')}
Arbitration: {contract_data.get('arbitration', 'Not specified')}
Source File: {contract_data.get('source_file', 'Not specified')}
"""
    
    # Generate insights for all 30 CFO questions
    insights = []
    
    for dimension, question in config.CFO_DIMENSIONS_AND_QUESTIONS:
        insight = generate_single_contract_insight(contract_context, dimension, question, contract_data)
        insights.append(insight)
    
    return insights

def generate_single_contract_insight(contract_context: str, dimension: str, question: str, contract_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate insight for a single CFO question for a specific contract"""
    
    prompt = f"""
You are a financial analyst analyzing a specific contract. Answer the following CFO question based on the contract data provided.

Contract Data:
{contract_context}

Question: {question}

Provide a structured response in JSON format:
{{
    "dimension": "{dimension}",
    "question": "{question}",
    "insight": "Your analysis and insight (2-3 sentences)",
    "kpis": {{"key_metric": "value", "another_metric": "value"}},
    "risks": ["risk1", "risk2"],
    "opportunities": ["opportunity1", "opportunity2"],
    "evidence": [{{"snippet": "relevant contract clause", "source": "contract data"}}],
    "data_gaps": ["missing information"],
    "confidence": 0.85
}}

Guidelines:
- Be specific to this contract
- Use actual contract data in your analysis
- Identify real risks and opportunities
- Provide concrete KPIs where possible
- Confidence should reflect data completeness
- If information is missing, note it in data_gaps
"""
    
    try:
        response = chat_strict(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=45,
            model="gpt-4"
        )
        
        # Use json_safe_load function (already defined at module level)
        parsed = json_safe_load(response)
        
        if parsed:
            return parsed
        else:
            return create_default_insight(dimension, question, contract_data)
            
    except Exception as e:
        print(f"[Contract Extractor] Error generating insight: {e}")
        import traceback
        traceback.print_exc()
        return create_default_insight(dimension, question, contract_data)

def create_default_insight(dimension: str, question: str, contract_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create default insight when generation fails"""
    
    return {
        "dimension": dimension,
        "question": question,
        "insight": f"Analysis for {contract_data.get('contract_id', 'Unknown')} contract - detailed analysis pending",
        "kpis": {"contract_value": contract_data.get('total_value_usd', 0)},
        "risks": ["Analysis pending"],
        "opportunities": ["Analysis pending"],
        "evidence": [{"snippet": "Contract data extraction in progress", "source": "system"}],
        "data_gaps": ["Detailed analysis not yet completed"],
        "confidence": 0.3
    }

# ==================== Integration Functions ====================

def process_uploaded_documents_for_dashboard(corpus_pairs: List[tuple]) -> Dict[str, Any]:
    """
    Main function to process uploaded documents for dashboard integration
    Returns dictionary with processed data ready for dashboard
    """
    
    print(f"[Contract Extractor] Processing {len(corpus_pairs)} uploaded documents...")
    
    # Extract structured contract data
    new_contracts_df = process_uploaded_contracts(corpus_pairs)
    
    if new_contracts_df.empty:
        return {"error": "No contracts could be extracted from uploaded documents"}
    
    # Merge with dummy contracts
    combined_df = merge_with_existing_contracts(new_contracts_df)
    
    # Save combined contracts data
    output_path = save_combined_contracts(combined_df)
    
    # Generate CFO insights for new contracts
    new_contract_insights = []
    for _, contract_row in new_contracts_df.iterrows():
        contract_data = contract_row.to_dict()
        insights = generate_contract_specific_cfo_insights(contract_data)
        new_contract_insights.extend(insights)
    
    return {
        "uploaded_contracts_df": combined_df,
        "output_path": output_path,
        "new_contract_insights": new_contract_insights,
        "total_contracts": len(combined_df),
        "new_contracts_count": len(new_contracts_df)
    }

def process_only_new_uploads_for_dashboard(corpus_pairs: List[tuple]) -> Dict[str, Any]:
    """
    FAST processing: Only extract NEW uploaded contracts (no dummy merge)
    For CFO insights export - dashboard will handle combining with existing data
    """
    
    print(f"[Contract Extractor] FAST MODE: Processing ONLY {len(corpus_pairs)} uploaded documents...")
    
    # Extract structured contract data
    new_contracts_df = process_uploaded_contracts(corpus_pairs)
    
    if new_contracts_df.empty:
        return {"error": "No contracts could be extracted from uploaded documents"}
    
    # Save ONLY new contracts (fast!)
    new_output_path = save_new_contracts_only(new_contracts_df)
    
    # ALSO update the main uploaded_contracts.csv for CFO analysis
    try:
        # Load existing contracts
        if os.path.exists("uploaded_contracts.csv"):
            existing_df = pd.read_csv("uploaded_contracts.csv")
            existing_ids = set(existing_df['contract_id'].tolist())
            
            # Add only truly new contracts
            truly_new = new_contracts_df[~new_contracts_df['contract_id'].isin(existing_ids)]
            if not truly_new.empty:
                combined_df = pd.concat([existing_df, truly_new], ignore_index=True)
                output_path = save_combined_contracts(combined_df, "uploaded_contracts.csv")
                print(f"[Contract Extractor] Updated main CSV: {len(combined_df)} total contracts")
        else:
            # First time - just save new contracts
            output_path = save_combined_contracts(new_contracts_df, "uploaded_contracts.csv")
            print(f"[Contract Extractor] Created main CSV: {len(new_contracts_df)} contracts")
    except Exception as e:
        print(f"[Contract Extractor] Error updating main CSV: {e}")
        output_path = new_output_path
    
    # Generate CFO insights for ONLY new contracts
    new_contract_insights = []
    for _, contract_row in new_contracts_df.iterrows():
        contract_data = contract_row.to_dict()
        insights = generate_contract_specific_cfo_insights(contract_data)
        new_contract_insights.extend(insights)
    
    return {
        "uploaded_contracts_df": new_contracts_df,  # Only new contracts
        "output_path": output_path,
        "new_contract_insights": new_contract_insights,  # CFO insights for new contracts only
        "total_contracts": len(new_contracts_df),
        "new_contracts_count": len(new_contracts_df),
        "mode": "fast_new_only"
    }
