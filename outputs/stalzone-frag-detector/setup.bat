@echo off
setlocal
cd /d "%~dp0"

echo Creating Python environment...
python -m venv .venv || goto :error

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --editable ".[dev]" || goto :error

echo.
echo Setup complete. Activate .venv or drag a video onto run_detector.bat.
pause
exit /b 0

:error
echo.
echo Setup failed. Check that Python 3.10 or newer is installed.
pause
exit /b 1
