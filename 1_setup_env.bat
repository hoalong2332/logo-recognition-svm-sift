@echo off
chcp 65001 >nul
title [BUOC 1] CAI DAT MOI TRUONG CUC BO .VENV

cd /d "%~dp0"

echo =======================================================================
echo          BUOC 1: KHOI TAO VA CAI DAT THU VIEN VAO .VENV (PYTHON 3.12)
echo =======================================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [*] Dang khoi tao moi truong ao .venv bang Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo [!] Khong goi duoc py -3.12, thu khoi tao bang lenh python mac dinh...
        python -m venv .venv
    )
    echo [v] Da tao xong thu muc .venv!
) else (
    echo [v] Da co san thu muc .venv.
)

echo [*] Dang kich hoat .venv va cai dat cac goi thu vien...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo =======================================================================
echo [v] CAI DAT HOAN TAT! Môi truong da duoc khoa dung phien ban.
echo =======================================================================
echo.
pause