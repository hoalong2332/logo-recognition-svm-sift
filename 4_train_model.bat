@echo off
chcp 65001 >nul
title [BUOC 4] TRICH XUAT DAC TRUNG VA HUAN LUYEN MO HINH SVM

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [X] Lỗi: Chưa có .venv! Hãy chạy file 1_setup_env.bat trước.
    pause
    exit /b
)

call .venv\Scripts\activate.bat
echo =======================================================================
echo          BƯỚC 4: ĐANG TRÍCH XUẤT ĐẶC TRƯNG WAVELET VÀ HUAN LUYEN SVM...
echo =======================================================================
echo.

python src/model.py

echo.
echo =======================================================================
echo [v] HUẤN LUYỆN THÀNH CÔNG! File mô hình đã được lưu tại models/
echo =======================================================================
echo.
pause