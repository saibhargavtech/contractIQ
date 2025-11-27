"""
CFO Analytics and Insights Generation
Specialized analytics for financial contract analysis and CFO reporting
"""

import json
import pandas as pd
from typing import Dict, List, Any
from collections import defaultdict

import config
import utils
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

# ==================== CFO Insights Generation ====================

def generate_cfo_insights_from_context(context_blob: str, contract_df=None) -> List[Dict]:
    """
    Generate comprehensive CFO insights from graph context
    Uses the CFO dimensions and questions framework
    """
    questions_block = "\n".join([f"- [{dim}] {q}" for dim, q in config.CFO_DIMENSIONS_AND_QUESTIONS])
    
    # Enhanced contract data for KPI calculation
    contract_data_context = ""
    if contract_df is not None and not contract_df.empty:
        # Calculate detailed financial metrics
        total_value = contract_df['total_value_usd'].sum()
        annual_commitment = contract_df['annual_commitment_usd'].sum()
        avg_contract_value = contract_df['total_value_usd'].mean()
        median_contract_value = contract_df['total_value_usd'].median()
        max_contract_value = contract_df['total_value_usd'].max()
        min_contract_value = contract_df['total_value_usd'].min()
        
        # Calculate compliance and risk metrics
        compliance_summary = contract_df['compliance'].value_counts().to_dict()
        sla_summary = contract_df['sla_uptime'].value_counts().to_dict()
        payment_terms_summary = contract_df['payment_terms'].value_counts().to_dict()
        
        # Calculate vendor concentration
        vendor_counts = contract_df['counterparty'].value_counts().to_dict()
        top_vendors = dict(list(vendor_counts.items())[:5])
        
        # Calculate contract status distribution
        status_summary = contract_df['status'].value_counts().to_dict()
        
        financial_summary = f"""
CONTRACT FINANCIAL DATA FOR KPI CALCULATION:
- Total contracts: {len(contract_df)}
- Total contract value: ${total_value:,.2f} USD
- Annual commitment: ${annual_commitment:,.2f} USD  
- Average contract value: ${avg_contract_value:,.2f} USD
- Median contract value: ${median_contract_value:,.2f} USD
- Maximum contract value: ${max_contract_value:,.2f} USD
- Minimum contract value: ${min_contract_value:,.2f} USD
- Contract types distribution: {contract_df['type'].value_counts().to_dict()}
- Top counterparties by count: {top_vendors}
- Payment terms distribution: {payment_terms_summary}
- SLA uptime targets: {sla_summary}
- Compliance requirements: {compliance_summary}
- Contract status distribution: {status_summary}

DETAILED CONTRACT DATA SAMPLE (first 10 contracts):
{contract_df[['contract_id', 'counterparty', 'total_value_usd', 'annual_commitment_usd', 'payment_terms', 'type']].head(10).to_string()}
"""
        contract_data_context = f"\n\n{financial_summary}"

    user_prompt = f"""
FINANCIAL DATA ANALYSIS TASK

You have been given contract financial data below. USE THIS DATA to answer the 30 CFO questions.

CONTRACT FINANCIAL SUMMARY:{contract_data_context}

TASK RULES:
1. MUST use the numbers above (e.g., if total contract value is $457,248,146, use exactly that)
2. Calculate KPIs from the data table provided 
3. Fill kpis fields with actual calculated values
4. Answer each question using the financial data
5. NEVER respond with "not available" or "not specified" when data is provided

ANSWER THESE 30 QUESTIONS:
{questions_block}

FORMAT: Return JSON array with exact schema for each question:
{get_cfo_json_schema_instruction()}

EXAMPLE GOOD RESPONSE:
{{"dimension": "Financial Exposure", "question": "What is the total value?", "insight": "Total contract value is $457,248,146 across 71 contracts", "kpis": {{"total_value": 457248146, "contract_count": 71}}, "confidence": 0.9}}

DO THIS NOW: Analyze the provided financial data and generate rich KPIs!
"""
    
    system_msg = (
        "You are a financial analyst WITH access to contract data provided below. "
        "TASK: Analyze the provided contract financial data and calculate KPIs. "
        "IGNORE any thoughts about not having access - use the data provided in the user message. "
        "CRITICAL INSTRUCTIONS: "
        "1. Extract and use ALL financial numbers shown (e.g., total contract value, annual commitments) "
        "2. Calculate specific KPIs from the data tables (total_value, avg_contract_size, top_vendors) "
        "3. Fill kpis field with calculated values using the provided numbers "
        "4. NEVER say 'data not available' - use the financial data provided "
        "5. Return meaningful CFO insights with specific dollar amounts "
        "You MUST work with the provided contract data! Use it immediately!"
    )
    
    prompt_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt}
    ]
    
    # Try to import extraction module for chat_strict
    try:
        from extraction import chat_strict
        raw = chat_strict(prompt_messages, temperature=0.1, timeout=120, model="gpt-4")
        
        if len(raw) == 0:
            print("[CFO Analytics] Error: LLM returned empty response, using fallback")
            return create_empty_cfo_records()
        
        # Check if response is truncated
        if raw.endswith('"confidence') or raw.endswith('"kpis": {') or '"confidence' in raw.split('"confidence')[-1]:
            print("[CFO Analytics] WARNING: Response appears truncated!")
            # Try to fix truncated JSON
            if raw.endswith('"confidence'):
                raw += '": 0.0}'
            elif not raw.endswith(']'):
                raw = raw.rstrip(' \n\r')
                if raw.endswith(','):
                    raw = raw[:-1]
                raw += ']'
        
        # Use json_safe_load function (already defined at module level)
        parsed = json_safe_load(raw)
        
        records = normalize_cfo_records(parsed)
        
        # Ensure we have responses for all questions
        if not records:
            records = create_empty_cfo_records()
        
        return records
    except Exception as e:
        print(f"[CFO Analytics] Error generating insights: {e}")
        import traceback
        traceback.print_exc()
        return create_empty_cfo_records()

def get_cfo_json_schema_instruction():
    """Return instruction for CFO JSON schema"""
    return (
        "Return STRICT JSON (an array) with one object per question. "
        "Each object MUST follow this schema:\n"
        "{\n"
        '  "dimension": "string",\n'
        '  "question": "string",\n'
        '  "insight": "string",\n'
        '  "kpis": { "name": "value/string/number", ... },\n'
        '  "risks": ["string", ...],\n'
        '  "opportunities": ["string", ...],\n'
        '  "evidence": [{"snippet": "string", "source": "string"}],\n'
        '  "data_gaps": ["string", ...],\n'
        '  "confidence": 0.0\n'
        "}\n"
        "- confidence in [0,1].\n"
        "- Use only information grounded in the provided context. If unknown, leave fields minimal and add a data_gaps note.\n"
        "- Keep answers concise and CFO-ready.\n"
    )

def normalize_cfo_records(parsed):
    """Normalize CFO records to ensure required fields"""
    records = []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return records
    
    for item in parsed:
        if not isinstance(item, dict):
            continue
        obj = {
            "dimension": item.get("dimension", ""),
            "question": item.get("question", ""),
            "insight": item.get("insight", ""),
            "kpis": item.get("kpis", {}) or {},
            "risks": item.get("risks", []) or [],
            "opportunities": item.get("opportunities", []) or [],
            "evidence": item.get("evidence", []) or [],
            "data_gaps": item.get("data_gaps", []) or [],
            "confidence": float(item.get("confidence", 0.0) or 0.0),
        }
        records.append(obj)
    return records

def create_empty_cfo_records():
    """Create empty records for all CFO questions"""
    records = []
    for dim, q in config.CFO_DIMENSIONS_AND_QUESTIONS:
        records.append({
            "dimension": dim,
            "question": q,
            "insight": "",
            "kpis": {},
            "risks": [],
            "opportunities": [],
            "evidence": [],
            "data_gaps": ["No extractable data from context memory."],
            "confidence": 0.0,
        })
    return records

# ==================== Financial Metrics Extraction ====================

def extract_financial_metrics_from_graph(G) -> Dict[str, Any]:
    """Extract financial metrics from graph structure"""
    
    # Contract volume metrics
    contract_metrics = {
        "total_contracts": G.number_of_nodes(),
        "total_relationships": G.number_of_edges(),
        "avg_connections_per_contract": G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    }
    
    # Vendor concentration
    vendor_nodes = []
    for node in G.nodes():
        if any(indicator in node.lower() for indicator in ['corp', 'inc', 'ltd', 'company', 'technologies']):
            vendor_nodes.append(node)
    
    if vendor_nodes:
        vendor_subgraph = G.subgraph(vendor_nodes)
        contract_metrics["unique_vendors"] = len(vendor_nodes)
        contract_metrics["vendor_relationships"] = vendor_subgraph.number_of_edges()
        contract_metrics["avg_vendor_connections"] = vendor_subgraph.number_of_edges() / len(vendor_nodes) if vendor_nodes else 0
    
    # Risk indicators
    degrees = [d for _, d in G.degree()]
    if degrees:
        contract_metrics["degree_distribution_gini"] = utils.gini(degrees)
        contract_metrics["max_contract_connections"] = max(degrees)
        contract_metrics["min_contract_connections"] = min(degrees)
    
    return contract_metrics

def calculate_compliance_coverage(G) -> Dict[str, float]:
    """Calculate compliance standard coverage from contract graph"""
    compliance_standards = {
        'GDPR', 'SOX', 'IFRS', 'ASC', 'GAAP', 'ISO', 'PCI', 'SOC2', 'HIPAA', 'FCPA', 
        'India DPA 2023', 'UK Bribery Act', 'CCPA'
    }
    
    # Count contracts mentioning each standard
    standard_mentions = defaultdict(int)
    total_contracts = G.number_of_nodes()
    
    for node in G.nodes():
        node_lower = node.lower()
        for standard in compliance_standards:
            standard_lower = standard.lower()
            if standard_lower.replace(' ', '') in node_lower.replace(' ', ''):
                standard_mentions[standard] += 1
    
    # Calculate coverage percentages
    coverage = {}
    for standard in compliance_standards:
        if standard in standard_mentions:
            coverage[standard] = standard_mentions[standard] / total_contracts if total_contracts > 0 else 0
        else:
            coverage[standard] = 0.0
    
    return coverage

# ==================== Risk Assessment ====================

def assess_contract_risks(G, partition=None) -> Dict[str, Any]:
    """Comprehensive contract risk assessment"""
    
    risks = {
        "concentration_risk": {},
        "dependency_risk": {},
        "compliance_gaps": {},
        "operational_risks": {}
    }
    
    # Concentration risk (few contracts dominate)
    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        if degrees:
            gini_coefficient = utils.gini(degrees)
            risks["concentration_risk"]["gini_coefficient"] = gini_coefficient
            risks["concentration_risk"]["level"] = "high" if gini_coefficient > 0.6 else "medium" if gini_coefficient > 0.4 else "low"
            
            # Identify hub contracts (top 10%)
            top_10_percent = max(1, len(degrees) // 10)
            hub_threshold = sorted(degrees, reverse=True)[top_10_percent - 1]
            hub_contracts = [node for node, degree in G.degree() if degree >= hub_threshold]
            risks["concentration_risk"]["hub_contracts"] = hub_contracts
    
    # Dependency risk (single points of failure)
    risks["dependency_risk"]["single_source_vendors"] = []
    risks["dependency_risk"]["contracts_with_one_vendor"] = []
    
    for node in G.nodes():
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        if len(neighbors) == 1:  # Single vendor dependency
            risks["dependency_risk"]["contracts_with_one_vendor"].append(node)
    
    # Compliance gaps
    compliance_coverage = calculate_compliance_coverage(G)
    low_coverage_standards = {std: coverage for std, coverage in compliance_coverage.items() if coverage < 0.3}
    risks["compliance_gaps"]["low_coverage_standards"] = low_coverage_standards
    
    # Operational risks
    risks["operational_risks"]["connected_components"] = len(list(G.subgraph(c) for c in nx.weakly_connected_components(G)))
    risks["operational_risks"]["isolated_contracts"] = [node for node in G.nodes() if G.degree(node) == 0]
    
    return risks

# ==================== Opportunity Identification ====================

def identify_cost_optimization_opportunities(G) -> List[Dict]:
    """Identify cost optimization opportunities from contract graph"""
    opportunities = []
    
    # Vendor consolidation opportunities
    vendor_groups = defaultdict(list)
    for node in G.nodes():
        # Try to identify vendor relationships
        vendor_name = node.lower()
        for indicator in ['corp', 'inc', 'ltd', 'company', 'technologies']:
            if indicator in vendor_name:
                base_name = vendor_name.split(indicator)[0].strip()
                vendor_groups[base_name].append(node)
                break
    
    # Find multiple contracts with similar vendors
    for vendor_base, contracts in vendor_groups.items():
        if len(contracts) > 1:
            opportunities.append({
                "type": "vendor_consolidation",
                "description": f"Multiple contracts with {vendor_base}",
                "contracts": contracts,
                "potential_savings": "Negotiate bulk discounts"
            })
    
    # Compliance standardization opportunities
    opportunities.append({
        "type": "compliance_standardization", 
        "description": "Contract compliance policy gaps",
        "action": "Implement standardized compliance requirements",
        "potential_savings": "Reduce audit and compliance costs"
    })
    
    return opportunities

# ==================== Executive Summary Generation ====================

def generate_executive_summary(graph_summary: Dict, risk_assessment: Dict, 
                             financial_metrics: Dict, opportunities: List[Dict]) -> Dict[str, str]:
    """Generate CFO executive summary"""
    
    summary = {
        "portfolio_overview": "",
        "key_risks": "",
        "opportunities": "",
        "recommended_actions": ""
    }
    
    # Portfolio overview
    contract_count = financial_metrics.get("total_contracts", 0)
    relationship_count = financial_metrics.get("total_relationships", 0)
    summary["portfolio_overview"] = f"Portfolio analyzed: {contract_count} contracts with {relationship_count} relationships. Network density: {graph_summary.get('network_density', 0):.3f}"
    
    # Key risks
    high_risks = []
    if risk_assessment.get("concentration_risk", {}).get("level") == "high":
        high_risks.append("High vendor concentration risk")
    if risk_assessment.get("compliance_gaps", {}).get("low_coverage_standards"):
        high_risks.append("Compliance coverage gaps identified")
    
    summary["key_risks"] = "; ".join(high_risks) if high_risks else "Risk levels are manageable"
    
    # Opportunities
    opt_types = [opt["type"] for opt in opportunities]
    summary["opportunities"] = f"Identified opportunities: {', '.join(set(opt_types))}" if opt_types else "No immediate opportunities identified"
    
    # Recommended actions
    actions = []
    if financial_metrics.get("degree_distribution_gini", 0) > 0.6:
        actions.append("Consider vendor diversification")
    if risk_assessment.get("compliance_gaps", {}).get("low_coverage_standards"):
        actions.append("Strengthen compliance framework")
    actions.append("Regular contract review cycle implementation")
    
    summary["recommended_actions"] = "; ".join(actions)
    
    return summary
