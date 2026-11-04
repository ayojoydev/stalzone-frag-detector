@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call setup.bat
    if errorlevel 1 exit /b 1
)

if "%~1"=="" (
    ".venv\Scripts\python.exe" -m szfd --menu
) else (
    ".venv\Scripts\python.exe" -m szfd --menu --input "%~1"
)
set "taskExitCode=%ERRORLEVEL%"

if not "%taskExitCode%"=="0" (
    echo.
    echo Detector exited with code %taskExitCode%.
    pause
)
exit /b %taskExitCode%
