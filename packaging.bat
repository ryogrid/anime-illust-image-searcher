@echo off

set "SRC_DIR=%~1"
set "TARGET_DIR=%~2"

if "%SRC_DIR%"=="" (
    echo please pass 1st argumnt (source dir).
    goto end
)

if "%TARGET_DIR%"=="" (
    echo please pass 2nd argument (target dir).
    goto end
)

if not exist "%TARGET_DIR%" (
    mkdir "%TARGET_DIR%"
)

REM activate venc
call "%SRC_DIR%\venv\Scripts\activate.bat"

REM packaging .py files on source dir
for %%f in ("%SRC_DIR%\*.py") do (
    echo %%~nxf is packaging...
    pyinstaller --onefile "%%f" --distpath "%TARGET_DIR%"
)

REM deactivate venv
deactivate

:end
