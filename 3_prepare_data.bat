@echo off
chcp 65001 >nul
title [BUOC 3] CHUAN HOA ANH VA SINH TAP DU LIEU KIEM THU

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [X] Lỗi: Chưa có .venv! Hãy chạy file 1_setup_env.bat trước.
    pause
    exit /b
)

call .venv\Scripts\activate.bat
echo =======================================================================
echo          BƯỚC 3: ĐANG TIỀN XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU ẢNH...
echo =======================================================================
echo.

python src/preprocess.py

echo.
echo =======================================================================
echo [v] ĐÃ CHUẨN HÓA XONG TOÀN BỘ ẢNH!
echo =======================================================================
echo.
pause