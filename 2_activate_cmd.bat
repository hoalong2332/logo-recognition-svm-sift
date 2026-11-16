@echo off
chcp 65001 >nul
title [BUOC 2] MO CUA SO DONG LENH DA KICH HOAT (.VENV)

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [X] Lỗi: Chưa tìm thấy .venv! Hãy chạy file 1_setup_env.bat trước.
    pause
    exit /b
)

call .venv\Scripts\activate.bat
echo =======================================================================
echo   ĐÃ KÍCH HOẠT THÀNH CÔNG (.VENV). BẠN ĐANG Ở TRONG DỰ ÁN CỤC BỘ
echo =======================================================================
echo.
cmd /k