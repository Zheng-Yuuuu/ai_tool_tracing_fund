@echo off
chcp 65001 >nul
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
if errorlevel 1 pause
