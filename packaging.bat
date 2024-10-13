@echo off

REM ��1�����Ƒ�2�������擾
set "SRC_DIR=%~1"
set "TARGET_DIR=%~2"

REM �����̊m�F
if "%SRC_DIR%"=="" (
    echo ��1�����i�\�[�X�f�B���N�g���j���w�肵�Ă��������B
    goto end
)

if "%TARGET_DIR%"=="" (
    echo ��2�����i�^�[�Q�b�g�f�B���N�g���j���w�肵�Ă��������B
    goto end
)

REM �^�[�Q�b�g�f�B���N�g�����쐬
if not exist "%TARGET_DIR%" (
    mkdir "%TARGET_DIR%"
)

REM venv���A�N�e�B�x�[�g
call "%SRC_DIR%\venv\Scripts\activate.bat"

REM �\�[�X�f�B���N�g������.py�t�@�C�����p�b�P�[�W��
for %%f in ("%SRC_DIR%\*.py") do (
    echo %%~nxf ���p�b�P�[�W�����Ă��܂�...
    pyinstaller --onefile "%%f" --distpath "%TARGET_DIR%"
)

REM venv���f�B�A�N�e�B�x�[�g
deactivate

:end

