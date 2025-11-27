"""
GraphRAG Persistence Module
Enables saving and loading GraphRAG state across sessions
"""

import os
import pickle
import networkx as nx
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import config

# Persistence file paths
GRAPH_FILE = "graphrag_graph.pkl"
PARTITION_FILE = "graphrag_partition.pkl"
CONTEXT_MEMORY_FILE = "graphrag_context_memory.pkl"
RAW_TEXT_FILE = "graphrag_raw_text.pkl"
ENTITIES_CSV = "entities.csv"
TRIPLES_LOG = "graphrag_triples_log.txt"

def save_graphrag_data():
    """Save complete GraphRAG state to files"""
    try:
        # Save Graph
        if hasattr(config, 'G') and config.G is not None:
            with open(GRAPH_FILE, 'wb') as f:
                pickle.dump(config.G, f)
            print(f"[GraphRAG Persistence] Saved graph with {config.G.number_of_nodes()} nodes, {config.G.number_of_edges()} edges")
        
        # Save Partition
        if hasattr(config, 'PARTITION') and config.PARTITION:
            with open(PARTITION_FILE, 'wb') as f:
                pickle.dump(config.PARTITION, f)
            print(f"[GraphRAG Persistence] Saved partition with {len(config.PARTITION)} node assignments")
        
        # Save Context Memory
        if hasattr(config, 'GRAPH_CONTEXT_MEMORY') and config.GRAPH_CONTEXT_MEMORY:
            with open(CONTEXT_MEMORY_FILE, 'wb') as f:
                pickle.dump(config.GRAPH_CONTEXT_MEMORY, f)
            print(f"[GraphRAG Persistence] Saved context memory")
        
        # Save Raw Combined Text
        if hasattr(config, 'RAW_COMBINED_TEXT') and config.RAW_COMBINED_TEXT:
            with open(RAW_TEXT_FILE, 'wb') as f:
                pickle.dump(config.RAW_COMBINED_TEXT, f)
            print(f"[GraphRAG Persistence] Saved raw text")
        
        # Export entities CSV for dashboard
        export_entities_csv()
        
        print(f"[GraphRAG Persistence] ✅ All GraphRAG data saved successfully")
        
    except Exception as e:
        print(f"[GraphRAG Persistence] ❌ Error saving: {e}")

def load_existing_graph_graphrag():
    """Load existing GraphRAG graph state"""
    try:
        if os.path.exists(GRAPH_FILE):
            with open(GRAPH_FILE, 'rb') as f:
                graph = pickle.load(f)
            print(f"[GraphRAG Persistence] Loaded existing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph
        else:
            print("[GraphRAG Persistence] No existing graph found, creating new empty graph")
            return nx.DiGraph()
    except Exception as e:
        print(f"[GraphRAG Persistence] Error loading graph: {e}")
        return nx.DiGraph()

def load_existing_partition():
    """Load existing partition"""
    try:
        if os.path.exists(PARTITION_FILE):
            with open(PARTITION_FILE, 'rb') as f:
                partition = pickle.load(f)
            print(f"[GraphRAG Persistence] Loaded existing partition with {len(partition)} node assignments")
            return partition
        else:
            return {}
    except Exception as e:
        print(f"[GraphRAG Persistence] Error loading partition: {e}")
        return {}

def load_existing_context_memory():
    """Load existing context memory"""
    try:
        if os.path.exists(CONTEXT_MEMORY_FILE):
            with open(CONTEXT_MEMORY_FILE, 'rb') as f:
                context_memory = pickle.load(f)
            print("[GraphRAG Persistence] Loaded existing context memory")
            return context_memory
        else:
            return "(no graph memory)"
    except Exception as e:
        print(f"[GraphRAG Persistence] Error loading context memory: {e}")
        return "(no graph memory)"

def load_existing_raw_text():
    """Load existing raw combined text"""
    try:
        if os.path.exists(RAW_TEXT_FILE):
            with open(RAW_TEXT_FILE, 'rb') as f:
                raw_text = pickle.load(f)
            print("[GraphRAG Persistence] Loaded existing raw text")
            return raw_text
        else:
            return ""
    except Exception as e:
        print(f"[GraphRAG Persistence] Error loading raw text: {e}")
        return ""

def export_entities_csv():
    """Export entities from graph to CSV for dashboard compatibility"""
    try:
        if not hasattr(config, 'G') or config.G is None:
            return
            
        entities_data = []
        seen_entities = set()
        
        # Extract all nodes as entities
        for node in config.G.nodes():
            if node not in seen_entities:
                node_data = config.G.nodes[node]
                entity_info = {
                    "entity_name": node,
                    "entity_type": node_data.get("type", "unknown"),
                    "node_degree": config.G.degree(node),
                    "in_degree": config.G.in_degree(node),
                    "out_degree": config.G.out_degree(node)
                }
                
                # Add node attributes
                for attr, value in node_data.items():
                    entity_info[f"attribute_{attr}"] = value
                
                entities_data.append(entity_info)
                seen_entities.add(node)
        
        # Export to CSV
        df = pd.DataFrame(entities_data)
        df.to_csv(ENTITIES_CSV, index=False)
        
        entities_log_file = "graphrag_entities_log.txt"
        with open(entities_log_file, 'w') as f:
            f.write(f"Total entities: {len(entities_data)}\n")
            f.write(f"Entity types: {df['entity_type'].value_counts().to_dict()}\n")
            f.write(f"Top entities by degree:\n")
            top_entities = df.nlargest(10, 'node_degree')[['entity_name', 'entity_type', 'node_degree']]
            for _, row in top_entities.iterrows():
                f.write(f"{row['entity_name']} ({row['entity_type']}) - {row['node_degree']} connections\n")
        
        print(f"[GraphRAG Persistence] Exported {len(entities_data)} entities to {ENTITIES_CSV}")
        
    except Exception as e:
        print(f"[GraphRAG Persistence] Error exporting entities: {e}")

def log_triples_for_session(triples: List[Tuple[str, str, str, str]], session_info: str):
    """Log triples extracted in current session"""
    try:
        with open(TRIPLES_LOG, 'a') as f:
            f.write(f"\n=== SESSION: {session_info} ===\n")
            f.write(f"Triples extracted: {len(triples)}\n")
            for head, relation, tail, evidence in triples:
                f.write(f"{head} [{relation}] {tail} (Evidence: {evidence[:100]}...)\n")
            f.write("\n")
    except Exception as e:
        print(f"[GraphRAG Persistence] Error logging triples: {e}")

def get_graphrag_stats():
    """Get current GraphRAG statistics"""
    stats = {
        "graph_exists": os.path.exists(GRAPH_FILE),
        "partition_exists": os.path.exists(PARTITION_FILE),
        "context_memory_exists": os.path.exists(CONTEXT_MEMORY_FILE),
        "raw_text_exists": os.path.exists(RAW_TEXT_FILE),
        "entities_csv_exists": os.path.exists(ENTITIES_CSV)
    }
    
    if hasattr(config, 'G') and config.G is not None:
        stats.update({
            "current_nodes": config.G.number_of_nodes(),
            "current_edges": config.G.number_of_edges(),
            "graph_size_mb": os.path.getsize(GRAPH_FILE) / 1024 / 1024 if os.path.exists(GRAPH_FILE) else 0
        })
    
    return stats

def clear_graphrag_data():
    """Clear all saved GraphRAG data (for testing)"""
    files_to_clear = [
        GRAPH_FILE, PARTITION_FILE, CONTEXT_MEMORY_FILE, 
        RAW_TEXT_FILE, TRIPLES_LOG
    ]
    
    for file_path in files_to_clear:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[GraphRAG Persistence] Cleared {file_path}")
    
    print("[GraphRAG Persistence] All GraphRAG data cleared")

if __name__ == "__main__":
    # Test the persistence system
    stats = get_graphrag_stats()
    print("GraphRAG Persistence Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
