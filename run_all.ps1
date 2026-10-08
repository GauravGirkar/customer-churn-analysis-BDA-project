# One-shot: build tables from real data -> Spark pipeline -> dashboard.
# Usage:  .\run_all.ps1 [-Scale 100000] [-Tune]    (omit -Scale to use all 7,043 real customers)
param([int]$Scale = 0, [switch]$Tune)

if ($Scale -gt 0) { python -m churn.generate_data --scale $Scale } else { python -m churn.generate_data }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Tune) { python -m churn.pipeline --tune } else { python -m churn.pipeline }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

streamlit run dashboard/app.py
