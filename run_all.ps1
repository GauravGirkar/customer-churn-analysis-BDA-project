# One-shot: generate data -> Spark pipeline -> dashboard.   Usage:  .\run_all.ps1 [-Customers 100000] [-Tune]
param([int]$Customers = 100000, [switch]$Tune)

python -m churn.generate_data --customers $Customers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Tune) { python -m churn.pipeline --tune } else { python -m churn.pipeline }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

streamlit run dashboard/app.py
