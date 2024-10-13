@echo off

REM 第1引数と第2引数を取得
set "SRC_DIR=%~1"
set "TARGET_DIR=%~2"

REM 引数の確認
if "%SRC_DIR%"=="" (
    echo 第1引数（ソースディレクトリ）を指定してください。
    goto end
)

if "%TARGET_DIR%"=="" (
    echo 第2引数（ターゲットディレクトリ）を指定してください。
    goto end
)

REM ターゲットディレクトリを作成
if not exist "%TARGET_DIR%" (
    mkdir "%TARGET_DIR%"
)

REM venvをアクティベート
call "%SRC_DIR%\venv\Scripts\activate.bat"

REM ソースディレクトリ内の.pyファイルをパッケージ化
for %%f in ("%SRC_DIR%\*.py") do (
    echo %%~nxf をパッケージ化しています...
    pyinstaller --onefile "%%f" --distpath "%TARGET_DIR%"
)

REM venvをディアクティベート
deactivate

:end

