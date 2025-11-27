# GraphRAG Studio - Modular Architecture

This is a refactored version of the GraphRAG application, broken down into logical modules for better maintainability and understanding.

## Project Structure

```
├── main.py              # Entry point - launches the application
├── config.py            # Configuration and constants
├── utils.py             # Utility functions (file reading, text processing)
├── extraction.py        # Entity and relation extraction
├── graph.py             # Graph building and community detection
├── clustering.py        # Cluster summarization and analysis
├── visualization.py     # Graph visualization and metrics
├── vector.py            # Vector search (LlamaIndex)
├── chat.py              # Q&A and chat functionality
├── export.py            # RDF and CSV export
├── ui.py                # Gradio interface components
└── README.md            # This file
```

## Module Descriptions

### `main.py`
- Entry point for the application
- Sets up OpenAI API key
- Launches the Gradio interface

### `config.py`
- All global constants and configuration settings
- Optional dependency flags
- Default parameters for graph processing
- Relation policies and entity aliases
- RDF configuration

### `utils.py`
- File reading functions (PDF, DOCX, TXT)
- Text processing utilities (chunking, parsing)
- Mathematical utilities (entropy, Gini coefficient)
- Entity and relation normalization
- RDF utilities

### `extraction.py`
- LLM-based entity and relation extraction
- Rule-based extraction as fallback
- Triple parsing and cleaning
- Main extraction pipeline

### `graph.py`
- NetworkX graph building
- Louvain community detection
- Graph context memory building
- Search options generation
- Graph state management

### `clustering.py`
- Cluster summarization using LLM
- Heuristic cluster analysis
- Cluster information tables
- Individual cluster descriptions

### `visualization.py`
- Graph visualization with matplotlib
- Comprehensive metrics computation
- Entity and relationship tables
- Export-ready data structures

### `vector.py`
- LlamaIndex integration
- Document indexing
- Semantic search
- Vector index management

### `chat.py`
- Question answering system
- Entity and cluster descriptions
- Graph-based queries
- LLM context building
- Chat interface bridge

### `export.py`
- RDF ontology building
- Graph export to Turtle and JSON-LD
- CFO dashboard CSV exports
- Combined text exports

### `ui.py`
- Gradio interface layout
- Component interactions
- Main pipeline orchestration
- UI action handlers

## Key Improvements

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Easier Maintenance**: Changes to one feature don't affect others
3. **Better Testing**: Individual modules can be tested in isolation
4. **Clearer Code**: Each file is focused and easier to understand
5. **Modular Imports**: Only import what you need for specific functionality

## Usage

Run the application with:
```bash
python main.py
```

The application will launch a Gradio interface where you can:
- Upload documents or specify a directory
- Configure graph processing parameters
- Build knowledge graphs from financial documents
- Query the graph through natural language
- Export results in various formats

## Dependencies

The application requires:
- gradio
- networkx
- python-docx (optional, for Word documents)
- PyPDF2 (optional, for PDF documents)
- python-dotenv
- community (python-louvain)
- matplotlib
- pandas
- openai
- rdflib (optional, for RDF export)
- llama-index (optional, for vector search)

## Configuration

All configuration is centralized in `config.py`. Key settings include:
- Extraction parameters (temperature, strictness)
- Graph processing options
- Relation policies
- File type support
- RDF namespaces

## Extending the Application

To add new features:
1. Identify the appropriate module
2. Add new functions following existing patterns
3. Update imports in dependent modules
4. Add UI components in `ui.py` if needed

This modular structure makes it much easier to understand, modify, and extend the GraphRAG application.






