# ContractIQ Dashboard

A comprehensive CFO contract analytics dashboard built with Streamlit.

## Features

- 🧠 **InsightIQ**: Critical financial KPIs and vendor analysis
- 📊 **Portfolio Management**: Strategic planning and renewal pipeline
- 💰 **Cash Flow Analysis**: Financial impact analysis
- 💸 **Cost Control & Risk**: Risk management and optimization
- 📈 **Executive Insights**: Complete CFO analytics report
- 💬 **Talk to Your Contracts**: AI-powered contract Q&A

## Deployment

This dashboard is deployed on Streamlit Cloud.

### Local Development

```bash
cd frontend
streamlit run main_dashboard.py
```

### Requirements

See `requirements.txt` for all dependencies.

## Data Sources

The dashboard works with:
- Contract CSV files
- JSONL CFO insights files
- GraphRAG processed contract data

## Configuration

- Main dashboard: `main_dashboard.py`
- Streamlit config: `.streamlit/config.toml`
- Page modules: `modules/` directory