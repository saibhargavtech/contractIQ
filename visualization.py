"""
Enhanced Visualization Module for CFO Analytics
Graph visualization and metrics computation for contract analysis
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

import config
import utils
import graph

def visualize_graph(G):
    """Create enhanced graph visualization with CFO focus"""
    plt.figure(figsize=(15, 10))
    
    if G.number_of_nodes() == 0:
        plt.title("CFO Contract Analytics: Knowledge Graph (No Data)", fontsize=16, fontweight='bold')
        plt.text(0.5, 0.5, 'Upload contract documents\n to build knowledge graph', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.savefig("cfo_graphrag_output.png", dpi=150, bbox_inches='tight')
        plt.close()
        return "cfo_graphrag_output.png"

    # Enhanced layout for better CFO insights
    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100)
    
    # Color nodes by community and size by centrality
    communities = nx.get_node_attributes(G, 'community')
    node_colors = [communities.get(n, 0) for n in G.nodes()]
    
    # Calculate degree centrality for node sizing
    degree_centrality = nx.degree_centrality(G)
    node_sizes = [300 + (centrality * 2000) for centrality in degree_centrality.values()]

    # Draw the graph with enhanced styling
    nx.draw(
        G, pos, 
        with_labels=True,
        node_color=node_colors, 
        cmap=plt.cm.Set3,
        node_size=node_sizes,
        font_size=8,
        font_weight='bold',
        edge_color="#666666", 
        width=1.5,
        arrows=True,
        arrowsize=15,
        arrowstyle='->'
    )

    # Add enhanced edge labels with coloring
    high_importance_relations = ['recognised_under', 'governed_by', 'audited_by', 'has_obligation']
    
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        rel = d.get('label', '')
        if rel in high_importance_relations:
            edge_labels[(u, v)] = rel
        elif len(rel) <= 15:  # Only show shorter labels
            edge_labels[(u, v)] = rel

    nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels=edge_labels, 
        font_size=7,
        font_color='darkblue',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
    )

    # Enhanced title with metrics
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    density = nx.density(G)
    
    title = f"CFO Contract Analytics: Knowledge Graph\n{node_count} Contracts • {edge_count} Relationships • Density: {density:.3f}"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend for visual elements
    legend_elements = [
        plt.scatter([], [], c='C0', s=100, label='Contracts'),
        plt.scatter([], [], c='C1', s=200, label='Financial Relations'),
        plt.scatter([], [], c='C2', s=300, label='High Risk')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.axis('off')
    try:
        plt.tight_layout()
    except Exception:
        pass
    
    plt.savefig("cfo_graphrag_output.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return "cfo_graphrag_output.png"

def compute_metrics(G, partition):
    """Calculate comprehensive graph metrics with CFO focus"""
    V = G.number_of_nodes()
    E = G.number_of_edges()
    density = nx.density(G)
    avg_out = E / V if V > 0 else 0.0
    avg_in = E / V if V > 0 else 0.0

    # Convert to undirected for additional metrics
    Gu = G.to_undirected()
    degs = [d for _, d in Gu.degree()]
    
    if degs:
        deg_hist = Counter(degs)
        entropy = utils.entropy(list(deg_hist.values()))
        gini_coeff = utils.gini(degs)
    else:
        entropy = 0.0
        gini_coeff = 0.0

    # Advanced graph metrics
    try:
        assortativity = nx.degree_assortativity_coefficient(Gu) if E > 0 else 0.0
    except Exception:
        assortativity = 0.0
    
    try:
        clustering = nx.transitivity(Gu) if E > 0 else 0.0
    except Exception:
        clustering = 0.0

    # Connectivity metrics
    wcc_count = nx.number_weakly_connected_components(G) if V > 0 else 0
    giant_ratio = 0.0
    if V > 0 and wcc_count > 0:
        try:
            largest = max(nx.weakly_connected_components(G), key=len)
            giant_ratio = len(giant) / V if V > 0 else 0.0
        except Exception:
            giant_ratio = 1.0

    # Path metrics for giant component
    asp = 0.0
    diam = 0
    if Gu.number_of_nodes() > 1:
        Gc = utils.giant_component_undirected(Gu)
        try:
            if Gc.number_of_nodes() > 1:
                asp = nx.average_shortest_path_length(Gc)
                diam = nx.diameter(Gc)
        except Exception:
            asp, diam = 0.0, 0

    # Louvain modularity
    modularity = 0.0
    try:
        if partition:
            modularity = community_louvain.modularity(partition, Gu)
    except Exception:
        modularity = 0.0

    # Evidence and provenance metrics
    edges_with_evidence = sum(1 for _, _, d in G.edges(data=True) if d.get('evidence_text'))
    prov_completeness = (edges_with_evidence / E) if E > 0 else 0.0

    # Relation diversity
    rel_types = set(d.get('label') for _, _, d in G.edges(data=True) if d.get('label'))
    rel_type_count = len(rel_types)

    # CFO-specific metrics
    financial_relations = sum(1 for _, _, d in G.edges(data=True) 
                             if d.get('label') in ['recognised_under', 'governed_by', 'audited_by', 'has_obligation'])
    financial_rel_ratio = (financial_relations / E) if E > 0 else 0.0

    # Centrality-based risk indicators
    high_centrality_threshold = 0.1
    high_centrality_nodes = sum(1 for centrality in nx.degree_centrality(G).values() 
                                if centrality > high_centrality_threshold)
    
    r = lambda x: round(float(x), 3)
    rows = [
        ("Contract Count", V),
        ("Relationship Count", E),
        ("Network Density", r(density)),
        ("Avg Out-degree", r(avg_out)),
        ("Avg In-degree", r(avg_in)),
        ("Degree Entropy", r(entropy)),
        ("Gini Coefficient (Concentration)", r(gini_coeff)),
        ("Degree Assortativity", r(assortativity)),
        ("Global Clustering Coeff.", r(clustering)),
        ("Connected Components", int(wcc_count)),
        ("Giant Component Ratio", r(giant_ratio)),
        ("Avg Path Length (Giant)", r(asp)),
        ("Diameter (Giant)", int(diam)),
        ("Modularity (Quality)", r(modularity)),
        ("Evidence Completeness", r(prov_completeness)),
        ("Relation Types", int(rel_type_count)),
        ("Financial Relations (%)", r(financial_rel_ratio * 100)),
        ("High-Centrality Contracts", int(high_centrality_nodes)),
        ("Concentration Risk", "High" if gini_coeff > 0.6 else "Medium" if gini_coeff > 0.4 else "Low"),
    ]

    df = pd.DataFrame(rows, columns=["CFO Metric", "Value"])
    return df

def create_cfo_dashboard_summary(G, partition=None):
    """Create executive dashboard summary metrics"""
    if not G or G.number_of_nodes() == 0:
        return {
            "status": "No data available",
            "contracts": 0,
            "relationships": 0,
            "risk_level": "Unknown",
            "compliance_gaps": 0
        }
    
    # Basic metrics
    contract_count = G.number_of_nodes()
    relationship_count = G.number_of_edges()
    
    # Risk assessment
    degrees = [d for _, d in G.degree()]
    gini = utils.gini(degrees) if degrees else 0.0
    risk_level = "Low"
    if gini > 0.6:
        risk_level = "High"
    elif gini > 0.4:
        risk_level = "Medium"
    
    # Compliance indicators
    compliance_nodes = [node for node in G.nodes() 
                       if any(keyword in node.lower() for keyword in ['gdpr', 'sox', 'ifrs', 'compliance'])]
    compliance_coverage = len(compliance_nodes) / contract_count if contract_count > 0 else 0.0
    
    # Network quality
    density = nx.density(G)
    network_quality = "Excellent" if density > 0.5 else "Good" if density > 0.2 else "Fragmented"
    
    return {
        "status": "✅ Analysis Complete",
        "contracts": contract_count,
        "relationships": relationship_count,
        "risk_level": risk_level,
        "concentration_percent": round(gini * 100, 1),
        "compliance_coverage": round(compliance_coverage * 100, 1),
        "network_density": round(density, 3),
        "network_quality": network_quality,
        "clusters": len(set(partition.values())) if partition else 0
    }





