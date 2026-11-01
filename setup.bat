@echo off
setlocal
cd /d "%~dp0"

echo Creating Python environment...
python -m venv .venv || goto :error

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --editable . || goto :error

echo.
echo Setup complete. Open run_detector.bat to start the application.
pause
exit /b 0

:error
echo.
echo Setup failed. Check that Python 3.10 or newer is installed.
pause
exit /b 1
