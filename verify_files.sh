#!/bin/bash
# Script to verify all required files are ready for deployment

echo "🔍 Checking required files for Streamlit Cloud deployment..."
echo ""

# Check CSV files
echo "📊 CSV Files:"
required_csv=(
    "dummy_contracts_50.csv"
    "uploaded_contracts.csv"
    "entities.csv"
    "relationships.csv"
    "clusters.csv"
    "frontend/dummy_contracts_50.csv"
    "frontend/uploaded_contracts.csv"
)

for file in "${required_csv[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
    fi
done

echo ""
echo "📄 JSON/JSONL Files:"
required_json=(
    "cfo_contract_insights.jsonl"
    "cluster_summaries.json"
)

for file in "${required_json[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
    fi
done

echo ""
echo "📝 Text Files:"
required_txt=(
    "graph_context_memory.txt"
    "cfo_dashboard_export.txt"
)

for file in "${required_txt[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
    fi
done

echo ""
echo "🐍 Python Files:"
required_py=(
    "frontend/main_dashboard.py"
    "config.py"
    "requirements.txt"
)

for file in "${required_py[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
    fi
done

echo ""
echo "🔒 Security Check:"
if [ -f ".env" ]; then
    echo "  ⚠️  .env file exists - Make sure it's in .gitignore!"
    if git check-ignore -q .env; then
        echo "  ✅ .env is properly ignored"
    else
        echo "  ❌ .env is NOT ignored - FIX THIS!"
    fi
else
    echo "  ✅ No .env file (good for deployment)"
fi

echo ""
echo "📦 Git Status:"
echo "  Files that will be committed:"
git ls-files | grep -E "\.(csv|jsonl|json|txt)$" | head -10

echo ""
echo "✅ Verification complete!"

