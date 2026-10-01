@echo off
setlocal
cd /d "%~dp0"
set "EPIC_CONDA=C:\ML_Progs\Runtimes\Miniforge3\Scripts\conda.exe"
if not exist "%EPIC_CONDA%" (
  echo Miniforge not found. Activate epic-eda and run jupyter lab manually.
  exit /b 1
)
"%EPIC_CONDA%" run --no-capture-output -n epic-eda jupyter lab notebooks/02_eda_logistic_context.ipynb
