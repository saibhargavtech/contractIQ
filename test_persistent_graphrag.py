#!/usr/bin/env python3
"""
Test script for persistent GraphRAG functionality
"""

import os
import sys
import networkx as nx

# Add current directory to path
sys.path.append('.')

import config
import graphrag_persistence

def test_graphrag_persistence():
    """Test persistent GraphRAG functionality"""
    print("🧪 Testing Persistent GraphRAG System\n")
    
    # 1. Test initial state
    print("1️⃣ Initial GraphRAG State:")
    stats = graphrag_persistence.get_graphrag_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 2. Test graph creation and saving
    print("\n2️⃣ Creating Test Graph...")
    test_graph = nx.DiGraph()
    test_graph.add_edge("Company A", "follows", "GDPR", label="compliance")
    test_graph.add_edge("Contract X", "involves", "Company A", label="business")
    test_graph.add_node("Company A", type="organization")
    test_graph.add_node("GDPR", type="regulation")
    
    config.G = test_graph
    
    # Save the test graph
    graphrag_persistence.save_graphrag_data()
    print(f"   ✅ Test graph saved: {test_graph.number_of_nodes()} nodes, {test_graph.number_of_edges()} edges")
    
    # 3. Test loading existing data
    print("\n3️⃣ Testing Graph Loading...")
    loaded_graph = graphrag_persistence.load_existing_graph_graphrag()
    
    if loaded_graph.number_of_nodes() > 0:
        print(f"   ✅ Graph loaded successfully: {loaded_graph.number_of_nodes()} nodes")
        print(f"   📋 Nodes: {list(loaded_graph.nodes())}")
        print(f"   🔗 Edges: {list(loaded_graph.edges())}")
    else:
        print("   ❌ Graph loading failed")
    
    # 4. Test entities export
    print("\n4️⃣ Testing Entities Export...")
    config.G = loaded_graph
    graphrag_persistence.export_entities_csv()
    
    if os.path.exists("entities.csv"):
        import pandas as pd
        df = pd.read_csv("entities.csv")
