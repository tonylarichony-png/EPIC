@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0."
set "PYTHONPATH=%PROJECT_ROOT%\src"
set "PYTHON_EXE="
set "PYTHON_LAUNCH_ARGS="
set "CONDA_ENV_NAME="

rem Prefer the already activated environment.
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"

rem Also support conventional project-local virtual environments.
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"

rem Read the Conda environment name declared by this project.
if exist "%PROJECT_ROOT%\environment.yml" for /f "tokens=2 delims=:" %%E in ('findstr /b /c:"name:" "%PROJECT_ROOT%\environment.yml"') do set "CONDA_ENV_NAME=%%E"
if defined CONDA_ENV_NAME for /f "tokens=* delims= " %%E in ("%CONDA_ENV_NAME%") do set "CONDA_ENV_NAME=%%E"

rem Find that environment even when Conda was not activated in this terminal.
if defined CONDA_EXE for %%I in ("%CONDA_EXE%") do set "CONDA_BIN_DIR=%%~dpI"
if defined CONDA_BIN_DIR for %%I in ("%CONDA_BIN_DIR%..") do set "CONDA_BASE=%%~fI"
if not defined PYTHON_EXE if defined CONDA_ENV_NAME if defined CONDA_BASE if exist "%CONDA_BASE%\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=%CONDA_BASE%\envs\%CONDA_ENV_NAME%\python.exe"
if not defined PYTHON_EXE if defined CONDA_ENV_NAME if defined MINIFORGE_HOME if exist "%MINIFORGE_HOME%\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=%MINIFORGE_HOME%\envs\%CONDA_ENV_NAME%\python.exe"
if not defined PYTHON_EXE if defined CONDA_ENV_NAME if defined MAMBA_ROOT_PREFIX if exist "%MAMBA_ROOT_PREFIX%\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=%MAMBA_ROOT_PREFIX%\envs\%CONDA_ENV_NAME%\python.exe"
if not defined PYTHON_EXE if defined CONDA_ENV_NAME if exist "C:\ML_Progs\Runtimes\Miniforge3\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=C:\ML_Progs\Runtimes\Miniforge3\envs\%CONDA_ENV_NAME%\python.exe"
if not defined PYTHON_EXE if defined CONDA_ENV_NAME if exist "%USERPROFILE%\miniforge3\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniforge3\envs\%CONDA_ENV_NAME%\python.exe"

rem Finally fall back to a Python launcher available on PATH.
if not defined PYTHON_EXE where python >nul 2>&1 && set "PYTHON_EXE=python"
if not defined PYTHON_EXE where py >nul 2>&1 && set "PYTHON_EXE=py" && set "PYTHON_LAUNCH_ARGS=-3"

if not defined PYTHON_EXE (
    echo Python environment was not found.
    echo Activate the project environment and run this command again.
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_LAUNCH_ARGS% -c "from ml_project.experiment_state import main; raise SystemExit(main())" --project-root "%PROJECT_ROOT%"
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
