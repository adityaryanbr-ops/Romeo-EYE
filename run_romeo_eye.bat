@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" romeo_eye.py %*
) else (
    start "" pythonw romeo_eye.py %*
)
