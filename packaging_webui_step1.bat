@echo off
REM Batch file to package Python scripts into Windows executables
REM Usage: packaging_webui.bat

REM Package the script into an executable without the onefile option
pyinstaller --additional-hooks-dir=./hooks --noconfirm run_webui.py --clean

REM Clean up build and spec files
REM RMDIR /S /Q build
REM RMDIR /S /Q spec

ECHO Packaging complete.
