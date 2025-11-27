# PowerShell script to verify all required files are ready for deployment

Write-Host "🔍 Checking required files for Streamlit Cloud deployment..." -ForegroundColor Cyan
Write-Host ""

# Check CSV files
Write-Host "📊 CSV Files:" -ForegroundColor Yellow
$required_csv = @(
    "dummy_contracts_50.csv",
    "uploaded_contracts.csv",
    "entities.csv",
    "relationships.csv",
    "clusters.csv",
    "frontend\dummy_contracts_50.csv",
    "frontend\uploaded_contracts.csv"
)

foreach ($file in $required_csv) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (MISSING)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📄 JSON/JSONL Files:" -ForegroundColor Yellow
$required_json = @(
    "cfo_contract_insights.jsonl",
    "cluster_summaries.json"
)

foreach ($file in $required_json) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (MISSING)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📝 Text Files:" -ForegroundColor Yellow
$required_txt = @(
    "graph_context_memory.txt",
    "cfo_dashboard_export.txt"
)

foreach ($file in $required_txt) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (MISSING)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🐍 Python Files:" -ForegroundColor Yellow
$required_py = @(
    "frontend\main_dashboard.py",
    "config.py",
    "requirements.txt"
)

foreach ($file in $required_py) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (MISSING)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🔒 Security Check:" -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  ⚠️  .env file exists - Make sure it's in .gitignore!" -ForegroundColor Yellow
    $ignored = git check-ignore -q .env 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ .env is properly ignored" -ForegroundColor Green
    } else {
        Write-Host "  ❌ .env is NOT ignored - FIX THIS!" -ForegroundColor Red
    }
} else {
    Write-Host "  ✅ No .env file (good for deployment)" -ForegroundColor Green
}

Write-Host ""
Write-Host "📦 Git Status:" -ForegroundColor Yellow
Write-Host "  Files that will be committed:"
git ls-files | Select-String -Pattern "\.(csv|jsonl|json|txt)$" | Select-Object -First 10

Write-Host ""
Write-Host "✅ Verification complete!" -ForegroundColor Green

