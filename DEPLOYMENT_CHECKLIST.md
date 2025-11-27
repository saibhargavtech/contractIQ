# Deployment Checklist - Files to Commit

## ✅ Critical Files That MUST Be Committed

### Data Files (Required for App to Work)
- ✅ `dummy_contracts_50.csv` - Demo contract data
- ✅ `demo_contracts.csv` - Demo data
- ✅ `uploaded_contracts.csv` - User uploaded contracts
- ✅ `entities.csv` - Graph entities
- ✅ `relationships.csv` - Graph relationships  
- ✅ `clusters.csv` - Cluster data
- ✅ `new_contracts_only.csv` - New contracts
- ✅ `cfo_contract_insights.jsonl` - CFO insights (generated)
- ✅ `cluster_summaries.json` - Cluster summaries
- ✅ `graph_context_memory.txt` - Graph context for chatbot
- ✅ `cfo_dashboard_export.txt` - Dashboard exports

### Code Files
- ✅ `frontend/main_dashboard.py` - Main entry point
- ✅ `frontend/modules/*.py` - All page modules
- ✅ `frontend/utils.py` - Utility functions
- ✅ `config.py` - Configuration
- ✅ `*.py` - All Python modules needed
- ✅ `requirements.txt` - Dependencies
- ✅ `.streamlit/config.toml` - Streamlit config

### Documentation
- ✅ `README.md` - Project documentation
- ✅ `DEPLOYMENT_QUICK_START.md` - Deployment guide

## ❌ Files That Should NOT Be Committed

### Sensitive Data
- ❌ `.env` - Contains API keys (use Streamlit secrets instead)
- ❌ `.streamlit/secrets.toml` - Local secrets (use Streamlit Cloud secrets)

### Generated/Temporary Files
- ❌ `__pycache__/` - Python cache
- ❌ `*.pyc` - Compiled Python
- ❌ `*.log` - Log files
- ❌ `temp_uploads/` - Temporary uploads
- ❌ `*_backup_*.csv` - Backup files

### IDE/OS Files
- ❌ `.vscode/` - VS Code settings
- ❌ `.idea/` - IntelliJ settings
- ❌ `.DS_Store` - macOS files
- ❌ `Thumbs.db` - Windows files

## 🔍 Verify Before Pushing

Run these commands to check what will be committed:

```bash
# See what files are tracked
git status

# See what files are ignored
git status --ignored

# Verify CSV files are included
git ls-files | grep "\.csv$"

# Verify JSONL files are included
git ls-files | grep "\.jsonl$"

# Verify JSON files are included
git ls-files | grep "\.json$"
```

## 📝 Pre-Deployment Checklist

Before pushing to GitHub:

- [ ] All CSV files are in repository (check `git ls-files *.csv`)
- [ ] All JSONL files are in repository (check `git ls-files *.jsonl`)
- [ ] All JSON files are in repository (check `git ls-files *.json`)
- [ ] Graph context file is included (`graph_context_memory.txt`)
- [ ] `.env` file is NOT committed (check `git ls-files .env`)
- [ ] `requirements.txt` is up to date
- [ ] `.streamlit/config.toml` exists
- [ ] All Python modules are committed

## 🚀 After Deployment

1. Verify app loads without errors
2. Check that demo data is visible
3. Test file upload functionality
4. Verify chatbot has context
5. Check all pages load correctly

## ⚠️ Important Notes

- **Data files are required** - Without CSV/JSONL files, the app won't have data to display
- **Graph context is required** - Without `graph_context_memory.txt`, chatbot won't work properly
- **Never commit `.env`** - Use Streamlit Cloud secrets instead
- **Backup files are excluded** - Only current data files are needed

