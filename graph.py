"""
Enhanced Graph Building and Community Detection for CFO Analytics
Network construction and clustering for contract relationship analysis
"""

import networkx as nx
import community as community_louvain
from typing import Tuple, Dict, List

import config
import utils

# ==================== Graph Building ====================

def build_graph(extraction_output, source="document"):
    """Build NetworkX graph from extraction results"""
    global G
    if config.G is None or not isinstance(config.G, nx.DiGraph):
        config.G = nx.DiGraph()
    
    for a, rel, b, evidence in extraction_output:
        if config.G.has_edge(a, b):
            continue
        config.G.add_edge(a, b, label=rel, source=source, evidence_text=evidence)
    
    return config.G

def to_weighted_undirected(G: nx.DiGraph, use_ontology_for_clustering: bool) -> nx.Graph:
    """Convert directed graph to weighted undirected graph for clustering"""
    Gu = nx.Graph()
    
    for u, v, d in G.edges(data=True):
        base = 1.0
        if use_ontology_for_clustering:
            rel = (d.get("label") or "").lower()
            base *= config.ONTO_REL_BOOST.get(rel, 1.0)
        
        if Gu.has_edge(u, v):
            Gu[u][v]["weight"] += base
        else:
            Gu.add_edge(u, v, weight=base)
    
    return Gu

# ==================== Community Detection ====================

def detect_communities(G, resolution=None, random_state=None, use_ontology_for_clustering=False):
    """
    Detect communities using Louvain algorithm with ontology weighting
    """
    if G.number_of_nodes() == 0:
        return G, {}
    
    res = resolution if resolution is not None else config.LOUVAIN_RESOLUTION
    rs = random_state if random_state is not None else config.LOUVAIN_RANDOM_STATE
    
    Gu = to_weighted_undirected(G, use_ontology_for_clustering)
    partition = community_louvain.best_partition(
        Gu, 
        resolution=float(res), 
        random_state=int(rs), 
        weight="weight"
    )
    
    # Add community attributes to nodes
    nx.set_node_attributes(G, partition, 'community')
    return G, partition

# ==================== Graph Analysis Utilities ====================

def calculate_centrality_metrics(G):
    """Calculate various centrality metrics for CFO insights"""
    if G.number_of_nodes() == 0:
        return {}
    
    try:
        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        
        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # PageRank centrality
        pagerank = nx.pagerank(G)
        
        return {
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "pagerank": pagerank
        }
    except Exception as e:
        print(f"[Graph] Centrality calculation error: {e}")
        return {}

def identify_key_contracts(G, centrality_metrics, top_n=10):
    """Identify most important contracts based on centrality"""
    if not centrality_metrics:
        return {}
    
    metrics = {}
    
    for metric_name, metric_values in centrality_metrics.items():
        sorted_nodes = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        metrics[metric_name] = sorted_nodes[:top_n]
    
    return metrics

def analyze_vendor_relationships(G):
    """Analyze relationships between vendors/counterparties"""
    vendor_stats = {}
    
    for node in G.nodes():
        # Check if node looks like a vendor/company
        if any(indicator in node.lower() for indicator in ['corp', 'inc', 'ltd', 'company', 'technologies']):
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            vendor_stats[node] = {
                "degree": G.degree(node),
                "partners": len(set(neighbors)),
                "partner_list": neighbors[:10]  # Top 10 partners
            }
    
    # Sort by degree
    sorted_vendors = sorted(vendor_stats.items(), key=lambda x: x[1]["degree"], reverse=True)
    
    return dict(sorted_vendors[:20])  # Top 20 vendors

def extract_compliance_clusters(G):
    """Extract clusters related to compliance requirements"""
    compliance_keywords = [
        'gdpr', 'sox', 'ifrs', 'asc', 'gaap', 'iso', 'pci', 'soc2', 
        'compliance', 'audit', 'standard', 'regulation', 'policy'
    ]
    
    compliance_nodes = []
    for node in G.nodes():
        node_lower = node.lower()
        if any(keyword in node_lower for keyword in compliance_keywords):
            compliance_nodes.append(node)
    
    if not compliance_nodes:
        return {}
    
    # Find compliance-related subgraph
    compliance_subgraph = G.subgraph(compliance_nodes).copy()
    
    # Calculate compliance cluster stats
    compliance_stats = {
        "total_compliance_nodes": len(compliance_nodes),
        "compliance_edges": len(compliance_subgraph.edges()),
        "compliance_components": nx.number_connected_components(compliance_subgraph.to_undirected()),
        "key_compliance_entities": [
            node for node, _ in sorted(
                compliance_subgraph.degree(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        ]
    }
    
    return compliance_stats

def calculate_risk_indicators(G):
    """Calculate risk indicators from graph structure"""
    risk_indicators = {}
    
    if G.number_of_nodes() == 0:
        return risk_indicators
    
    # Concentration risk (Gini coefficient of degree distribution)
    degrees = [d for _, d in G.degree()]
    if degrees:
        risk_indicators["degree_concentration"] = utils.gini(degrees)
    
    # Single points of failure (high-degree nodes)
    high_degree_threshold = sorted(degrees)[-max(1, len(degrees)//10)]
    critical_nodes = [node for node, degree in G.degree() if degree >= high_degree_threshold]
    risk_indicators["critical_nodes"] = critical_nodes[:10]
    
    # Connectivity risk (fragmentation)
    num_components = nx.number_weakly_connected_components(G)
    largest_component_size = len(max(nx.weakly_connected_components(G), key=len))
    fragmentation_ratio = 1 - (largest_component_size / G.number_of_nodes())
    risk_indicators["fragmentation_ratio"] = fragmentation_ratio
    
    # Dependency concentration (top 10% nodes control how much of network)
    sorted_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_10_pct = max(1, len(sorted_by_degree) // 10)
    top_nodes_edges = sum(degree for _, degree in sorted_by_degree[:top_10_pct])
    total_edges = G.number_of_edges()
    dependency_concentration = top_nodes_edges / (2 * total_edges) if total_edges > 0 else 0
    risk_indicators["dependency_concentration"] = dependency_concentration
    
    return risk_indicators

# ==================== CFO-Specific Graph Insights ====================

def generate_cfo_graph_summary(G, partition=None, centrality_metrics=None):
    """Generate CFO-focused summary of graph structure"""
    summary = {
        "contract_count": G.number_of_nodes(),
        "relationship_count": G.number_of_edges(),
        "clusters": len(set(partition.values())) if partition else 0,
        "network_density": nx.density(G),
        "avg_connections_per_contract": G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    }
    
    # Add centrality insights
    if centrality_metrics:
        summary["top_hub_contracts"] = [
            node for node, _ in centrality_metrics.get("degree_centrality", {}).items()
        ][:5]
        
        summary["most_critical_contracts"] = [
            node for node, _ in centrality_metrics.get("betweenness_centrality", {}).items()
        ][:5]
    
    # Risk assessment
    risk_indicators = calculate_risk_indicators(G)
    summary["risk_indicators"] = risk_indicators
    
    # Vendor analysis
    vendor_analysis = analyze_vendor_relationships(G)
    summary["top_vendors"] = list(vendor_analysis.keys())[:10]
    
    # Compliance analysis
    compliance_analysis = extract_compliance_clusters(G)
    summary["compliance_status"] = compliance_analysis
    
    return summary





