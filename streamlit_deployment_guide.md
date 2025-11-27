# Streamlit Cloud Deployment Guide

This guide will help you deploy ContractIQ to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (free at https://share.streamlit.io)

## Step 1: Prepare Your Repository

### Required Files Structure

```
your-repo/
├── frontend/
│   ├── main_dashboard.py          # Main entry point
│   ├── modules/                   # All page modules
│   ├── utils.py                   # Utility functions
│   └── requirements.txt           # Dependencies (optional, uses root)
├── requirements.txt              # Main dependencies file
├── .streamlit/
│   └── config.toml              # Streamlit configuration
└── (other project files)
```

### Entry Point

Streamlit Cloud will look for `main_dashboard.py` in the root or `frontend/main_dashboard.py`. 

**Option 1: Deploy from root**
- Create a `main.py` in root that imports from `frontend/main_dashboard.py`

**Option 2: Deploy from frontend folder** (Recommended)
- Set the app path in Streamlit Cloud to: `frontend/main_dashboard.py`

## Step 2: Set Up Streamlit Secrets

1. Go to your Streamlit Cloud dashboard
2. Click "New app"
3. Connect your GitHub repository
4. In the "Secrets" section, add:

```toml
OPENAI_API_KEY = "sk-proj-your-actual-api-key-here"
```

**Important:** Never commit your API key to GitHub! Always use Streamlit secrets.

## Step 3: Configure App Settings

### App Path
- Set to: `frontend/main_dashboard.py`

### Python Version
- Use Python 3.9 or higher (recommended: 3.10)

### Branch
- Select your main branch (usually `main` or `master`)

## Step 4: Deploy

1. Click "Deploy"
2. Wait for the build to complete
3. Your app will be live at: `https://your-app-name.streamlit.app`

## Step 5: Verify Deployment

After deployment, check:
- ✅ App loads without errors
- ✅ OpenAI API key works (test the chatbot)
- ✅ File uploads work (Development Centre)
- ✅ All pages are accessible

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Make sure all dependencies are in `requirements.txt`
   - Check that file paths are relative, not absolute

2. **API Key Not Found**
   - Verify secrets are set correctly in Streamlit Cloud
   - Check the secrets format (no quotes needed in TOML)

3. **File Not Found Errors**
   - Ensure data files are committed to the repository
   - Or use Streamlit's file upload feature

4. **Memory Issues**
   - Large files might cause memory issues
   - Consider using Streamlit's caching more aggressively

## Environment Variables vs Secrets

- **Local Development**: Use `.env` file (not committed to Git)
- **Streamlit Cloud**: Use Streamlit secrets (set in dashboard)

The code automatically checks both in this order:
1. Streamlit secrets (for cloud)
2. Environment variables
3. `.env` file (for local)

## Updating Your App

1. Push changes to GitHub
2. Streamlit Cloud will automatically redeploy
3. Or manually trigger redeploy from the dashboard

## Security Notes

- ✅ Never commit `.env` files
- ✅ Never commit API keys
- ✅ Use Streamlit secrets for sensitive data
- ✅ Add `.env` to `.gitignore`

## Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Test locally first
3. Verify all dependencies are listed in `requirements.txt`

