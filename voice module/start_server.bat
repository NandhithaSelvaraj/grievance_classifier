@echo off
cd /d "%~dp0"
echo Starting Grievance System server...
echo (Keep this window open while testing. Close it when you're done.)
echo.
.\gpu_env\Scripts\uvicorn.exe main:app
pause
