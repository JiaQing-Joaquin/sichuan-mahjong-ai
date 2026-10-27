@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem Prefer the Anaconda environment (has torch); fall back to PATH python.
set "PY=D:\Anaconda3\envs\tensorflow\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\run_tests.py
echo.
echo ==== EXIT_CODE=%ERRORLEVEL% ====
pause
