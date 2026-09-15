@echo off
setlocal
chcp 65001 >nul

set "SCAFFOLD_SCRIPT=%~dp0src\ml_project\experiment_scaffold.py"
if not exist "%SCAFFOLD_SCRIPT%" (
    echo Experiment scaffolder not found: %SCAFFOLD_SCRIPT%
    exit /b 1
)

set "PYTHON_EXE="
if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
    )
)

if not defined PYTHON_EXE (
    for %%I in (python.exe) do set "PYTHON_EXE=%%~$PATH:I"
)

if not defined PYTHON_EXE (
    echo Python was not found. Activate the project environment first.
    echo Then run:
    echo.
    echo     .\new-experiment.cmd
    exit /b 1
)

set "PYTHONUTF8=1"
"%PYTHON_EXE%" "%SCAFFOLD_SCRIPT%" --project-root "%~dp0." %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
