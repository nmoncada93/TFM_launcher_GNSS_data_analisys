@echo off
setlocal

set "ROOT=%~dp0"

if not exist "%ROOT%app\results\" mkdir "%ROOT%app\results"
if not exist "%ROOT%results\" (
    echo Creating results\ link to app\results\ ...
    mklink /J "%ROOT%results" "%ROOT%app\results"
)

cd /d "%ROOT%app"

if exist "%ROOT%app\.venv\Scripts\python.exe" (
    echo Using app\.venv\Scripts\python.exe
    "%ROOT%app\.venv\Scripts\python.exe" "%ROOT%app\web_server.py"
) else if exist "%ROOT%python_portable\python.exe" (
    echo app\.venv not found - using python_portable\python.exe
    "%ROOT%python_portable\python.exe" "%ROOT%bootstrap_web_server.py"
) else (
    echo ERROR: neither app\.venv\Scripts\python.exe nor python_portable\python.exe found.
    echo Cannot start the server.
)

pause
