"""
Enhanced Clustering Module for CFO Analytics
Cluster summarization and context memory for financial insights
"""

import itertools
import openai
from collections import defaultdict
from typing import Dict, List

import config
import utils

# ==================== Cluster Summarization ====================

def heuristic_cluster_label_and_summary(G, cid, nodes, full_text):
    """Generate heuristic cluster labels and summaries"""
    sub = G.subgraph(nodes).copy()
    degrees = dict(sub.degree())
    top_nodes = [n for n, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]]
    top_nodes_str = ", ".join(top_nodes) if top_nodes else ", ".join(nodes[:3])
    
    label = "Theme / Risk"
    summary = (
        f"This cluster groups {len(nodes)} entities with dense links among {top_nodes_str}. "
        f"Theme inferred from connectivity and relation evidence."
    )
    
    return label, summary

def summarize_clusters(G, partition, full_text):
    """Generate comprehensive cluster summaries with CFO focus"""
    if not partition:
        return {}
    
    communities = {}
    for node, group in partition.items():
        communities.setdefault(group, []).append(node)

    combined = {}
    context = (full_text or "")[:2000]

    for comm_id, nodes in communities.items():
        ent_list = ', '.join(nodes[:50])
        fb_label, fb_summary = heuristic_cluster_label_and_summary(G, comm_id, nodes, context)
        summary = fb_summary
        label = fb_label

        # LLM-based enhancement for CFO insights
        try:
            summary_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"""
You are a financial analyst assistant. Given the following entities: {ent_list},
and the original context:
\"\"\"{context}\"\"\" 
Summarize this group's theme or risk in 2–3 sentences with CFO perspective.
Focus on: financial implications, compliance risks, vendor relationships, operational impact.
"""}],
                temperature=0.3, timeout=30
            )
            _s = (summary_response.choices[0].message.content or "").strip()
            if _s:
                summary = _s
        except Exception:
            pass

        try:
            label_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"""
Given entities: {ent_list} and the context above,
Suggest one concise CFO-focused risk or KPI label. Examples:
"Revenue Recognition Risk", "Vendor Concentration", "Compliance Gap", 
"Cost Escalation", "Payment Terms Risk". Reply with only the label.
"""}],
                temperature=0.2, timeout=20
            )
            _l = (label_response.choices[0].message.content or "").strip()
            if _l:
                label = _l
        except Exception:
            pass

        combined[comm_id] = {
            "Cluster Summary": summary,
            "KPI/Risk Label": label,
        }

    return combined

def build_graph_context_memory(G, partition, summaries, max_edges_per_cluster=6, max_nodes_list=8):
    """Build comprehensive context memory for CFO analytics"""
    if not partition:
        return "(no graph memory)"

    communities = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    lines = ["GRAPH CONTEXT MEMORY FOR CFO ANALYTICS"]
    lines.append("=" * 50)

    for cid, nodes in sorted(communities.items(), key=lambda x: x[0]):
        sub = G.subgraph(nodes).copy()
        deg = dict(sub.degree())
        top_nodes = [n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:max_nodes_list]]
        s = summaries.get(cid, {})
        label = s.get("KPI/Risk Label", "")
        summ = s.get("Cluster Summary", "")

        lines.append(f"\n[Cluster {cid}] CFO Risk/KPI: {label}")
        lines.append(f"[Cluster {cid}] Business Impact: {summ}")
        if top_nodes:
            lines.append(f"[Cluster {cid}] Key entities: {', '.join(top_nodes)}")

        # Add financial relationship evidence
        financial_edges = []
        for u, v, d in sub.edges(data=True):
            lbl = d.get("label", "")
            ev = d.get("evidence_text", "")
            
            # Prioritize financial/relevant relationships
            if any(keyword in lbl.lower() for keyword in ["recognised", "governed", "audited", "obligation", "payment"]):
                if len(ev) > config.MAX_EVIDENCE_CHARS:
                    ev = ev[:config.MAX_EVIDENCE_CHARS].rstrip() + "…"
                financial_edges.append(f"FINANCIAL: {u} —[{lbl}]→ {v}; Evidence: {ev}")
        
        # Also add general edges if less than required financial ones
        if len(financial_edges) < max_edges_per_cluster:
            general_count = 0
            for u, v, d in sub.edges(data=True):
                if general_count >= max_edges_per_cluster - len(financial_edges):
                    break
                lbl = d.get("label", "")
                ev = d.get("evidence_text", "")
                if len(financial_edges) == 0 or lbl:  # Skip if we already have financial edges
                    if len(ev) > config.MAX_EVIDENCE_CHARS:
                        ev = ev[:config.MAX_EVIDENCE_CHARS].rstrip() + "…"
                    financial_edges.append(f"[Cluster {cid}] Rel: {u} —[{lbl}]→ {v}; Ev: {ev}")
                    general_count += 1
            
            financial_edges.extend(financial_edges[:max_edges_per_cluster])
        
        for edge in financial_edges[:max_edges_per_cluster]:
            lines.append(edge)

    lines.append("\n" + "=" * 50)
    lines.append("END CFO CONTEXT MEMORY")
    
    return "\n".join(lines)

# ==================== CFO-Specific Cluster Analysis ====================

def analyze_clusters_for_cfo_insights(G, partition, summaries):
    """Analyze clusters with CFO perspective"""
    cfо_insights = {
        "high_risk_clusters": [],
        "vendor_concentration_clusters": [],
        "compliance_clusters": [],
        "financial_impact_clusters": []
    }

    communities = defaultdict(list)
    for node, group in partition.items():
        communities.setdefault(group, []).append(node)

    for cluster_id, nodes in communities.items():
        cluster_summary = summaries.get(cluster_id, {})
        label = cluster_summary.get("KPI/Risk Label", "").lower()
        
        # Categorize clusters by CFO concerns
        if any(keyword in label for keyword in ["risk", "breach", "penalty", "violation"]):
            cfо_insights["high_risk_clusters"].append({
                "cluster_id": cluster_id,
                "label": cluster_summary.get("KPI/Risk Label", ""),
                "summary": cluster_summary.get("Cluster Summary", ""),
                "entity_count": len(nodes)
            })
        
        if any(keyword in label for keyword in ["vendor", "counterparty", "supplier", "concentration"]):
            cfо_insights["vendor_concentration_clusters"].append({
                "cluster_id": cluster_id,
                "label": cluster_summary.get("KPI/Risk Label", ""),
                "summary": cluster_summary.get("Cluster Summary", ""),
                "entity_count": len(nodes)
            })
        
        if any(keyword in label for keyword in ["compliance", "regulation", "standard", "audit"]):
            cfо_insights["compliance_clusters"].append({
                "cluster_id": cluster_id,
                "label": cluster_summary.get("KPI/Risk Label", ""),
                "summary": cluster_summary.get("Cluster Summary", ""),
                "entity_count": len(nodes)
            })
        
        if any(keyword in label for keyword in ["revenue", "cost", "payment", "financial", "profit"]):
            cfо_insights["financial_impact_clusters"].append({
                "cluster_id": cluster_id,
                "label": cluster_summary.get("KPI/Risk Label", ""),
                "summary": cluster_summary.get("Cluster Summary", ""),
                "entity_count": len(nodes)
            })

    return cfо_insights

def generate_cluster_based_cfo_summary(cluster_insights):
    """Generate executive summary from cluster analysis"""
    summary_parts = []
    
    # Risk summary
    high_risk_count = len(cluster_insights["high_risk_clusters"])
    if high_risk_count > 0:
        summary_parts.append(f"⚠️ {high_risk_count} high-risk clusters identified requiring immediate attention")
    
    # Vendor concentration
    vendor_count = len(cluster_insights["vendor_concentration_clusters"])
    if vendor_count > 0:
        summary_parts.append(f"🏢 {vendor_count} vendor concentration clusters found - consider diversification")
    
    # Compliance insights
    compliance_count = len(cluster_insights["compliance_clusters"])
    summary_parts.append(f"📋 {compliance_count} compliance-related clusters requiring monitoring")
    
    # Financial impact
    financial_count = len(cluster_insights["financial_impact_clusters"])
    if financial_count > 0:
        summary_parts.append(f"💰 {financial_count} clusters with direct financial impact")
    
    return "\n".join(summary_parts) if summary_parts else "No significant cluster patterns identified"
