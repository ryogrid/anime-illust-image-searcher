@echo off
REM Batch file to package Python scripts into Windows executables
REM Usage: packaging_webui.bat

REM Package the script into an executable without the onefile option
pyinstaller --noconfirm run_webui.spec --clean

REM Clean up build and spec files
REM RMDIR /S /Q build
REM RMDIR /S /Q __pycache__

ECHO Packaging complete.
