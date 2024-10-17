@echo off
REM Batch file to package Python scripts into Windows executables
REM Usage: package_scripts.bat [InputDirectory] [OutputDirectory]

REM Check if two arguments are provided
IF "%~1"=="" (
    ECHO Please provide the input directory containing Python scripts.
    GOTO :EOF
)
IF "%~2"=="" (
    ECHO Please provide the output directory where executables will be stored.
    GOTO :EOF
)

SET input_dir=%~1
SET output_dir=%~2

REM Create the output directory if it doesn't exist
IF NOT EXIST "%output_dir%" (
    MKDIR "%output_dir%"
)

REM Package the script into an executable without the onefile option
pyinstaller --distpath "%output_dir%" --workpath "%output_dir%\build" --specpath "%output_dir%\spec" --noconfirm "%input_dir%\cmd_run.py

REM Clean up build and spec files
RMDIR /S /Q "%output_dir%\build"
RMDIR /S /Q "%output_dir%\spec"

ECHO Packaging complete.
