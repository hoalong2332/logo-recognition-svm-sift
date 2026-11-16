@echo off
chcp 65001 >nul
title [BUOC 5] DANH GIA THUC NGHIEM VA VE BIEU DO SO SANH

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [X] Lỗi: Chưa có .venv! Hãy chạy file 1_setup_env.bat trước.
    pause
    exit /b
)

call .venv\Scripts\activate.bat
echo =======================================================================
echo          BƯỚC 5: ĐANG ĐÁNH GIÁ SỐ LIỆU VÀ XUẤT BIỂU ĐỒ BÁO CÁO...
echo =======================================================================
echo.

python src/evaluate_benchmark.py
echo.
python src/visualize_results.py

echo.
echo =======================================================================
echo [v] ĐÃ XUẤT XONG BẢNG SỐ LIỆU VÀ BIỂU ĐỒ TRỰC QUAN TRONG RESULTS/!
echo =======================================================================
echo.
pause