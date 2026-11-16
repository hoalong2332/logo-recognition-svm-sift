@echo off
chcp 65001 >nul
title [BUOC 6] KHOI CHAY GIAO DIEN NHAN DIEN LOGO GUI

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [X] Lỗi: Chưa có .venv! Hãy chạy file 1_setup_env.bat trước.
    pause
    exit /b
)

call .venv\Scripts\activate.bat
echo =======================================================================
echo          BƯỚC 6: ĐANG KHỞI ĐỘNG GIAO DIỆN NGƯỜI DÙNG (GUI)...
echo =======================================================================
echo.

python src/gui.py