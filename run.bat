@echo off
setlocal
cd /d "%~dp0"
title Churn Prediction - BDA Project

REM ---------------------------------------------------------------
REM  run.bat            -> open the dashboard (builds everything first time only)
REM  run.bat rebuild    -> rebuild tables from the real data + re-run the Spark pipeline, then open dashboard
REM  run.bat tune       -> same as rebuild, with grid search + 3-fold CV (slower)
REM  run.bat 500000     -> rebuild, bootstrapping the real customers up to 500000 rows (scale demo)
REM ---------------------------------------------------------------

set "MODE=%~1"
set "SCALE="
set "TUNE="
set "REBUILD=0"

if /i "%MODE%"=="rebuild" set "REBUILD=1"
if /i "%MODE%"=="tune" (set "REBUILD=1" & set "TUNE=--tune")
echo %MODE%| findstr /r "^[0-9][0-9]*$" >nul && (set "SCALE=--scale %MODE%" & set "REBUILD=1")

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH. Install Python 3.10+ from python.org and tick "Add to PATH".
    goto :fail
)

where java >nul 2>nul
if errorlevel 1 if not defined JAVA_HOME (
    echo [ERROR] Java was not found. Spark needs JDK 17 or 21 - install Temurin/Microsoft OpenJDK 17.
    goto :fail
)

REM --- first-time dependency install (skipped once Spark + Streamlit import fine)
python -c "import pyspark, streamlit, plotly, lightgbm, xgboost, sklearn, pyarrow" >nul 2>nul
if errorlevel 1 (
    echo [1/4] Installing Python dependencies ^(first run only^)...
    python -m pip install -r requirements-pipeline.txt
    if errorlevel 1 goto :fail
)

if not exist "data\raw\customers.parquet" set "REBUILD=1"
if not exist "outputs\scores.parquet" set "REBUILD=1"

if "%REBUILD%"=="1" (
    echo [2/4] Building tables from the real Telco dataset %SCALE% ...
    python -m churn.generate_data %SCALE%
    if errorlevel 1 goto :fail
    echo [3/4] Running the Spark pipeline ^(takes a few minutes^)...
    python -m churn.pipeline %TUNE%
    if errorlevel 1 goto :fail
) else (
    echo Using existing data and model outputs ^(run "run.bat rebuild" to refresh^).
)

echo [4/4] Starting dashboard at http://localhost:8501  ^(Ctrl+C to stop^)
python -m streamlit run dashboard\app.py --server.port 8501 --browser.gatherUsageStats false
goto :eof

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
