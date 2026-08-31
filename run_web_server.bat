@echo off
setlocal

set "ROOT=%~dp0"

if exist "%ROOT%app\.venv\Scripts\python.exe" (
    echo Using app\.venv\Scripts\python.exe
    "%ROOT%app\.venv\Scripts\python.exe" "%ROOT%app\web_server.py"
) else if exist "%ROOT%app\python_portable\python.exe" (
    echo app\.venv not found - using app\python_portable\python.exe
    "%ROOT%app\python_portable\python.exe" "%ROOT%app\web_server.py"
) else (
    echo ERROR: neither app\.venv\Scripts\python.exe nor app\python_portable\python.exe found.
    echo Cannot start the server.
)

pause
