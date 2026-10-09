@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo Drag a video file onto run_detector.bat.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" stalzone_frag_detector.py "%~1" --clips --debug
set "taskExitCode=%ERRORLEVEL%"

echo.
if "%taskExitCode%"=="0" (
    echo Done. Results are next to the source video.
) else (
    echo Detector exited with code %taskExitCode%.
)
pause
exit /b %taskExitCode%
