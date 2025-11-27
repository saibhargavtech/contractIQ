# Quick Start: Deploy to Streamlit Cloud

## 🚀 5-Minute Deployment

### Step 0: Verify Files Are Included ⚠️ IMPORTANT

**Before pushing, make sure all data files are included:**

```bash
# Check CSV files are tracked
git ls-files | grep "\.csv$"

# Check JSONL files are tracked  
git ls-files | grep "\.jsonl$"

# Check JSON files are tracked
git ls-files | grep "\.json$"

# Verify these critical files exist:
# - dummy_contracts_50.csv
# - uploaded_contracts.csv
# - cfo_contract_insights.jsonl
# - cluster_summaries.json
# - graph_context_memory.txt
```

**Required files that MUST be committed:**
- ✅ All `.csv` files (contract data)
- ✅ All `.jsonl` files (insights)
- ✅ All `.json` files (cluster summaries)
- ✅ `graph_context_memory.txt` (chatbot context)
- ✅ All Python modules
- ✅ `requirements.txt`

**Files that should NOT be committed:**
- ❌ `.env` (use Streamlit secrets)
- ❌ `*_backup_*.csv` (backup files)
- ❌ `__pycache__/` (Python cache)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - **Repository**: Select your repo
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `frontend/main_dashboard.py`
   - **Python version**: 3.10 or higher

### Step 3: Add Secrets

Click **"Advanced settings"** → **"Secrets"** and add:

```toml
OPENAI_API_KEY = "sk-proj-your-actual-key-here"
```

**⚠️ Important:** Replace `your-actual-key-here` with your real OpenAI API key!

### Step 4: Deploy

Click **"Deploy"** and wait 2-3 minutes.

### Step 5: Access Your App

Your app will be live at:
```
https://your-app-name.streamlit.app
```

## ✅ That's It!

Your ContractIQ dashboard is now live on Streamlit Cloud.

## 🔧 Troubleshooting

**App won't start?**
- Check the logs in Streamlit Cloud dashboard
- Verify `requirements.txt` has all dependencies
- Make sure `frontend/main_dashboard.py` exists

**API key not working?**
- Double-check secrets are set correctly
- No quotes needed in TOML format
- Restart the app after adding secrets

**Import errors?**
- All dependencies must be in root `requirements.txt`
- Check file paths are relative, not absolute

## 📝 Next Steps

- Customize your app URL in settings
- Set up custom domain (if needed)
- Monitor usage in Streamlit Cloud dashboard

For detailed instructions, see `streamlit_deployment_guide.md`

