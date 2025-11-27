"""
Enhanced Export Module for CFO Contract Analytics
CSV, JSONL, and RDF export capabilities for CFO dashboards and analytics
"""

import json
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional

import config
import cfo_analytics
import utils

# Import RDF-related modules
try:
    from rdflib import Graph as RDFGraph, Namespace, URIRef, Literal, BNode, RDF, RDFS, OWL, SKOS, DCTERMS
    config.RDFLIB_AVAILABLE = True
except ImportError:
    config.RDFLIB_AVAILABLE = False
    RDFGraph = None
    Namespace = None
    URIRef = None
    Literal = None
    BNode = None
    RDF = None
    RDFS = None
    OWL = None
    SKOS = None
    DCTERMS = None

# Import networkx for graph operations
try:
    import networkx as nx
    from collections import defaultdict
except ImportError:
    nx = None
    defaultdict = None

# ==================== CFO CSV Export ====================

def compute_entity_table(G) -> pd.DataFrame:
    """Compute entity table for CFO CSV export"""
    nodes = list(G.nodes())
    communities = nx.get_node_attributes(G, 'community') if hasattr(nx, 'get_node_attributes') else {}
    in_deg = dict(G.in_degree()) if hasattr(G, 'in_degree') else {}
    out_deg = dict(G.out_degree()) if hasattr(G, 'out_degree') else {}
    
    rows = []
    for n in nodes:
        rows.append({
            "entity": n,
            "community": communities.get(n, None),
            "in_degree": int(in_deg.get(n, 0)),
            "out_degree": int(out_deg.get(n, 0)),
            "total_degree": int(in_deg.get(n, 0) + out_deg.get(n, 0)),
        })
    
    return pd.DataFrame(rows, columns=["entity", "community", "in_degree", "out_degree", "total_degree"])

def compute_relationship_table(G) -> pd.DataFrame:
    """Compute relationship table for CFO CSV export"""
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "source_entity": u,
            "relation": d.get("label", ""),
            "target_entity": v,
            "evidence": d.get("evidence_text", ""),
            "source_file": d.get("source", ""),
            "source_community": G.nodes[u].get("community", None),
            "target_community": G.nodes[v].get("community", None),
        })
    
    cols = ["source_entity", "relation", "target_entity", "evidence", "source_file", "source_community", "target_community"]
    return pd.DataFrame(rows, columns=cols)

def compute_cluster_table(G, summaries: Dict) -> pd.DataFrame:
    """Compute cluster summary table for CFO CSV export"""
    by_c = defaultdict(list)
    for n, c in nx.get_node_attributes(G, 'community').items():
        by_c[c].append(n)

    rows = []
    for cid, nodes in sorted(by_c.items(), key=lambda x: x[0]):
        sub = G.subgraph(nodes).copy()
        deg = dict(sub.degree())
        top_nodes = [n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:8]]
        label = summaries.get(cid, {}).get("KPI/Risk Label", "")
        summ = summaries.get(cid, {}).get("Cluster Summary", "")
        
        samples = []
        for u, v, d in sub.edges(data=True):
            samples.append(f"{u}[{d.get('label', '')}]→{v}")
        
        rows.append({
            "cluster_id": cid,
            "kpi_risk_label": label,
            "cluster_summary": summ,
            "node_count": sub.number_of_nodes(),
            "edge_count": sub.number_of_edges(),
            "top_entities": "; ".join(top_nodes),
            "sample_relations": "; ".join(samples[:8]),
        })
    
    cols = ["cluster_id", "kpi_risk_label", "cluster_summary", "node_count", "edge_count", "top_entities", "sample_relations"]
    return pd.DataFrame(rows, columns=cols)

def export_cfo_csv_and_txt():
    """Export CFO data to CSV and combined TXT files"""
    if config.G.number_of_nodes() == 0:
        return None, None, None, None

    entities_df = compute_entity_table(config.G)
    relationships_df = compute_relationship_table(config.G)
    clusters_df = compute_cluster_table(config.G, config.CLUSTER_SUMMARIES)

    ent_path = "entities.csv"
    rel_path = "relationships.csv"
    clu_path = "clusters.csv"
    
    entities_df.to_csv(ent_path, index=False, encoding="utf-8")
    relationships_df.to_csv(rel_path, index=False, encoding="utf-8")
    clusters_df.to_csv(clu_path, index=False, encoding="utf-8")

    txt_path = "cfo_dashboard_export.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# ENTITIES (CSV)\n")
        entities_df.to_csv(f, index=False, encoding="utf-8")
        f.write("\n# RELATIONSHIPS (CSV)\n")
        relationships_df.to_csv(f, index=False, encoding="utf-8")
        f.write("\n# CLUSTERS (CSV)\n")
        clusters_df.to_csv(f, index=False, encoding="utf-8")

    return ent_path, rel_path, clu_path, txt_path

# ==================== CFO JSONL Export ====================

def export_cfo_jsonl_from_context(jsonl_path: str = None) -> Optional[str]:
    """
    Export comprehensive CFO insights to JSONL format for external dashboards
    """
    jsonl_path = jsonl_path or config.DEFAULT_JSONL_PATH
    
    # Use available context or create a simple business context
    if config.GRAPH_CONTEXT_MEMORY and config.GRAPH_CONTEXT_MEMORY.strip() != "(no graph memory)":
        raw_slice = (config.RAW_COMBINED_TEXT or "")[:3500]
        context_blob = f"{config.GRAPH_CONTEXT_MEMORY}\n\nRAW_CONTEXT_SAMPLE:\n{raw_slice}"
    else:
        context_blob = "Contract portfolio analysis: Business contracts and agreements"
    
    # Load ALL contract data for CFO KPI calculation (to populate meaningful insights)
    contract_df = None
    try:
        # Always try to load the complete portfolio for full CFO analysis
        if os.path.exists("uploaded_contracts.csv"):
            contract_df = pd.read_csv("uploaded_contracts.csv")
            print(f"[Export] Using complete portfolio: {len(contract_df)} contracts for CFO analysis")
        else:
            print(f"[Export] No uploaded_contracts.csv found, CFO insights may be limited")
    except Exception as e:
        print(f"[Export] Could not load contract data: {e}")
    
    # Also try to include new contracts if available
    try:
        if os.path.exists("new_contracts_only.csv"):
            new_df = pd.read_csv("new_contracts_only.csv")
            if contract_df is not None:
                # Merge new contracts if not already included
                existing_ids = set(contract_df['contract_id'].tolist()) if 'contract_id' in contract_df.columns else set()
                new_contracts = new_df[~new_df['contract_id'].isin(existing_ids)]
                if not new_contracts.empty:
                    contract_df = pd.concat([contract_df, new_contracts], ignore_index=True)
                    print(f"[Export] Added {len(new_contracts)} additional new contracts")
            else:
                contract_df = new_df
                print(f"[Export] Using only new contracts data: {len(contract_df)} contracts")
    except Exception as e:
        print(f"[Export] Could not merge new contracts: {e}")
    
    records = cfo_analytics.generate_cfo_insights_from_context(context_blob, contract_df)
    
    if not records:
        records = cfo_analytics.create_empty_cfo_records()

    run_meta = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": "GraphRAG Studio CFO Analytics Module",
        "model": "gpt-4o"
    }

    try:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                obj = {**r, **{"_meta": run_meta}}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return jsonl_path
    except Exception as e:
        print("JSONL write error:", e)
        return None

# ==================== Enhanced CFO Analytics Export ====================

def export_comprehensive_cfo_report(output_dir: str = "cfo_reports") -> Dict[str, str]:
    """
    Export comprehensive CFO analysis report including:
    - Financial metrics
    - Risk assessment
    - Compliance analysis
    - Executive summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Financial metrics
        financial_metrics = cfo_analytics.extract_financial_metrics_from_graph(config.G)
        
        # Risk assessment
        risk_assessment = cfo_analytics.assess_contract_risks(config.G, config.PARTITION)
        
        # Opportunities
        opportunities = cfo_analytics.identify_cost_optimization_opportunities(config.G)
        
        # Executive summary
        graph_summary = {
            "network_density": nx.density(config.G) if config.G else 0,
            "total_contracts": config.G.number_of_nodes() if config.G else 0,
            "total_relationships": config.G.number_of_edges() if config.G else 0
        }
        
        exec_summary = cfo_analytics.generate_executive_summary(
            graph_summary, risk_assessment, financial_metrics, opportunities
        )
        
        # Export financial metrics
        financial_df = pd.DataFrame(list(financial_metrics.items()), columns=["Metric", "Value"])
        financial_file = os.path.join(output_dir, "financial_metrics.csv")
        financial_df.to_csv(financial_file, index=False)
        
        # Export risk assessment
        risk_file = os.path.join(output_dir, "risk_assessment.json")
        with open(risk_file, "w") as f:
            json.dump(risk_assessment, f, indent=2)
        
        # Export opportunities
        opportunities_df = pd.DataFrame(opportunities)
        opp_file = os.path.join(output_dir, "cost_optimization_opportunities.csv")
        opportunities_df.to_csv(opp_file, index=False)
        
        # Export executive summary
        summary_file = os.path.join(output_dir, "executive_summary.json")
        with open(summary_file, "w") as f:
            json.dump(exec_summary, f, indent=2)
        
        # Combined CFO insights JSONL
        jsonl_file = export_cfo_jsonl_from_context(os.path.join(output_dir, "cfo_insights.jsonl"))

        return {
            "financial_metrics": financial_file,
            "risk_assessment": risk_file,
            "opportunities": opp_file,
            "executive_summary": summary_file,
            "comprehensive_insights": jsonl_file
        }
    
    except Exception as e:
        print(f"[Export] Error creating comprehensive report: {e}")
        return {}

# ==================== RDF Export (Enhanced) ====================

def build_business_ontology(onto_iri: str = None, base_iri: str = None):
    """Build enhanced business ontology for CFO analytics"""
    if not config.RDFLIB_AVAILABLE:
        raise RuntimeError("rdflib not installed; cannot build ontology. pip install rdflib")
    
    onto_iri = onto_iri or config.DEFAULT_ONTO_IRI
    base_iri = base_iri or config.DEFAULT_BASE_IRI
    
    g = RDFGraph()
    ONTO = Namespace(onto_iri if onto_iri.endswith(('#', '/')) else onto_iri + "#")
    EX = Namespace(base_iri if base_iri.endswith(('#', '/')) else base_iri + "#")
    
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("rdf", RDF)
    g.bind("skos", SKOS)
    g.bind("dct", DCTERMS)
    g.bind("onto", ONTO)
    g.bind("ex", EX)

    onto_uri = URIRef(onto_iri.rstrip("#/"))
    g.add((onto_uri, RDF.type, OWL.Ontology))
    g.add((onto_uri, RDFS.label, Literal("CFO Contract Analytics Ontology", lang="en")))

    # Define core classes
    Entity = ONTO.Entity
    Standard = ONTO.Standard
    Contract = ONTO.Contract
    Risk = ONTO.Risk
    
    g.add((Entity, RDF.type, OWL.Class))
    g.add((Entity, RDFS.label, Literal("Entity", lang="en")))
    g.add((Standard, RDF.type, OWL.Class))
    g.add((Standard, RDFS.label, Literal("Standard", lang="en")))
    g.add((Contract, RDF.type, OWL.Class))
    g.add((Contract, RDFS.label, Literal("Contract", lang="en")))
    g.add((Risk, RDF.type, OWL.Class))
    g.add((Risk, RDFS.label, Literal("Risk", lang="en")))

    def add_obj_prop(local, label, realm=Entity, range_=Entity):
        p = ONTO[local]
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.label, Literal(label, lang="en")))
        g.add((p, RDFS.domain, realm))
        g.add((p, RDFS.range, range_))
        return p

    # Add enhanced business properties
    add_obj_prop("auditedBy", "audited by")
    add_obj_prop("governedBy", "governed by")
    add_obj_prop("recognisedUnder", "recognised under", domain=Entity, range_=Standard)
    add_obj_prop("hasObligation", "has obligation")
    add_obj_prop("involvesCounterparty", "involves counterparty")
    add_obj_prop("mapsToStandard", "maps to standard", domain=Entity, range_=Standard)
    add_obj_prop("linkedTo", "linked to")
    add_obj_prop("evidencedBy", "evidenced by")
    add_obj_prop("coOccursInSentence", "co-occurs in sentence")
    
    # Financial properties
    add_obj_prop("hasValue", "has contractual value")
    add_obj_prop("hasRisk", "has associated risk", domain=Contract, range_=Risk)
    add_obj_prop("involvesPayment", "involves payment terms")

    # Export ontology
    ttl_path = config.DEFAULT_ONTOLOGY_FILENAME
    g.serialize(destination=ttl_path, format="turtle")
    return ttl_path

def export_graph_as_rdf(base_iri: str = None, onto_iri: str = None):
    """Export graph as enhanced RDF with CFO analytics extensions including KPIs"""
    if not config.RDFLIB_AVAILABLE:
        raise RuntimeError("rdflib not installed; cannot export RDF. pip install rdflib")
    
    base_iri = base_iri or config.DEFAULT_BASE_IRI
    onto_iri = onto_iri or config.DEFAULT_ONTO_IRI
    
    g = RDFGraph()
    ONTO = Namespace(onto_iri if onto_iri.endswith(('#', '/')) else onto_iri + "#")
    EX = Namespace(base_iri if base_iri.endswith(('#', '/')) else base_iri + "#")
    
    g.bind("onto", ONTO)
    g.bind("ex", EX)
    g.bind("dct", DCTERMS)
    g.bind("cfo", Namespace("http://example.org/cfo#"))
    
    # Define CFO-specific classes
    CFO_NS = Namespace("http://example.org/cfo#")
    g.add((ONTO.Entity, RDF.type, OWL.Class))
    g.add((ONTO.Standard, RDF.type, OWL.Class))
    g.add((CFO_NS.CFOInsight, RDF.type, OWL.Class))
    g.add((CFO_NS.KPI, RDF.type, OWL.Class))
    g.add((CFO_NS.Risk, RDF.type, OWL.Class))
    g.add((CFO_NS.Opportunity, RDF.type, OWL.Class))

    # Add nodes with enhanced metadata
    for node, attrs in config.G.nodes(data=True):
        uri = URIRef(utils.node_iri(node, base_iri))
        g.add((uri, RDF.type, ONTO.Entity))
        g.add((uri, SKOS.prefLabel, Literal(str(node))))
        
        cid = attrs.get("community", None)
        if cid is not None:
            g.add((uri, DCTERMS.subject, Literal(f"cluster:{cid}")))
        
        # Add financial attributes if available
        degree = config.G.degree(node)
        g.add((uri, ONTO.degreeScore, Literal(degree)))

    # Add edges with enhanced properties
    prop_cache = {}
    
    def prop_for(rel_norm: str):
        local = config.REL_TO_PROP.get(rel_norm, rel_norm)
        if local not in prop_cache:
            prop_cache[local] = ONTO[local]
            g.add((prop_cache[local], RDF.type, OWL.ObjectProperty))
            g.add((prop_cache[local], RDFS.label, Literal(local)))
        return prop_cache[local]

    for u, v, d in config.G.edges(data=True):
        rel = (d.get("label") or "").lower()
        p = prop_for(rel)
        s = URIRef(utils.node_iri(u, base_iri))
        o = URIRef(utils.node_iri(v, base_iri))
        g.add((s, p, o))
        
        # Add evidence
        ev = d.get("evidence_text", "")
        src = d.get("source", "")
        if ev:
            b = BNode()
            g.add((s, ONTO.evidencedBy, b))
            g.add((b, RDF.type, RDFS.Resource))
            g.add((b, DCTERMS.description, Literal(ev)))
            if src:
                g.add((b, DCTERMS.source, Literal(src)))

    # CRITICAL FIX: Add CFO insights and KPIs to the RDF graph
    try:
        # Load CFO insights from JSONL file
        cfo_insights = []
        if os.path.exists("cfo_contract_insights.jsonl"):
            with open("cfo_contract_insights.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            insight = json.loads(line.strip())
                            cfo_insights.append(insight)
                        except json.JSONDecodeError:
                            continue
        
        # Add CFO insights to RDF graph
        for i, insight in enumerate(cfo_insights):
            insight_uri = URIRef(f"{base_iri}cfo_insight_{i}")
            g.add((insight_uri, RDF.type, CFO_NS.CFOInsight))
            g.add((insight_uri, DCTERMS.title, Literal(insight.get("dimension", "Unknown"))))
            g.add((insight_uri, DCTERMS.description, Literal(insight.get("question", ""))))
            g.add((insight_uri, CFO_NS.insight, Literal(insight.get("insight", ""))))
            g.add((insight_uri, CFO_NS.confidence, Literal(insight.get("confidence", 0.0))))
            
            # Add KPIs
            kpis = insight.get("kpis", {})
            for kpi_name, kpi_value in kpis.items():
                # URL encode KPI names to handle spaces and special characters
                import urllib.parse
                safe_kpi_name = urllib.parse.quote(kpi_name, safe='')
                kpi_uri = URIRef(f"{base_iri}kpi_{i}_{safe_kpi_name}")
                g.add((kpi_uri, RDF.type, CFO_NS.KPI))
                g.add((kpi_uri, DCTERMS.title, Literal(kpi_name)))
                g.add((kpi_uri, CFO_NS.value, Literal(str(kpi_value))))
                g.add((insight_uri, CFO_NS.hasKPI, kpi_uri))
            
            # Add risks
            risks = insight.get("risks", [])
            for j, risk in enumerate(risks):
                risk_uri = URIRef(f"{base_iri}risk_{i}_{j}")
                g.add((risk_uri, RDF.type, CFO_NS.Risk))
                g.add((risk_uri, DCTERMS.description, Literal(risk)))
                g.add((insight_uri, CFO_NS.hasRisk, risk_uri))
            
            # Add opportunities
            opportunities = insight.get("opportunities", [])
            for j, opp in enumerate(opportunities):
                opp_uri = URIRef(f"{base_iri}opportunity_{i}_{j}")
                g.add((opp_uri, RDF.type, CFO_NS.Opportunity))
                g.add((opp_uri, DCTERMS.description, Literal(opp)))
                g.add((insight_uri, CFO_NS.hasOpportunity, opp_uri))
            
            # Add evidence
            evidence = insight.get("evidence", [])
            for j, ev in enumerate(evidence):
                ev_uri = URIRef(f"{base_iri}evidence_{i}_{j}")
                g.add((ev_uri, RDF.type, RDFS.Resource))
                g.add((ev_uri, DCTERMS.description, Literal(ev.get("snippet", ""))))
                g.add((ev_uri, DCTERMS.source, Literal(ev.get("source", ""))))
                g.add((insight_uri, CFO_NS.evidencedBy, ev_uri))
        
        print(f"[Export] Added {len(cfo_insights)} CFO insights with KPIs to RDF graph")
        
    except Exception as e:
        print(f"[Export] Warning: Could not add CFO insights to RDF: {e}")

    # Export RDF files
    ttl_path = config.DEFAULT_RDF_TTL_FILENAME
    jsonld_path = config.DEFAULT_JSONLD_FILENAME
    
    g.serialize(destination=ttl_path, format="turtle")
    try:
        g.serialize(destination=jsonld_path, format="json-ld")
        print(f"[Export] Successfully exported enhanced JSON-LD with CFO insights and KPIs")
    except Exception as e:
        print(f"[Export] Error serializing JSON-LD: {e}")
        jsonld_path = None
    
    return ttl_path, jsonld_path

# ==================== Integration Functions ====================

def export_all_cfo_formats(output_dir: str = "cfo_output") -> Dict[str, str]:
    """Export all CFO formats: CSV, JSONL, JSON, and RDF"""
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = {}
    
    try:
        # CSV exports
        ent_path, rel_path, clu_path, txt_path = export_cfo_csv_and_txt()
        exported_files["entities_csv"] = ent_path
        exported_files["relationships_csv"] = rel_path
        exported_files["clusters_csv"] = clu_path
        exported_files["combined_txt"] = txt_path
        
        # JSONL export
        jsonl_path = export_cfo_jsonl_from_context(os.path.join(output_dir, "cfo_insights.jsonl"))
        exported_files["insights_jsonl"] = jsonl_path
        
        # Comprehensive report
        report_files = export_comprehensive_cfo_report(output_dir)
        exported_files.update(report_files)
        
        # RDF exports
        ontology_path = build_business_ontology()
        exported_files["ontology"] = ontology_path
        
        rdf_ttl, rdf_jsonld = export_graph_as_rdf()
        exported_files["rdf_ttl"] = rdf_ttl
        if rdf_jsonld:
            exported_files["rdf_jsonld"] = rdf_jsonld
        
        return exported_files
        
    except Exception as e:
        print(f"[Export] Error in comprehensive export: {e}")
        return exported_files
