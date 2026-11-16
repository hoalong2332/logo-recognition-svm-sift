import os
import re
import sys
import time
import cv2
import numpy as np
import pandas as pd
import joblib

# Thêm đường dẫn để gọi module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature import extract_dtcwt_features
from sift_homography import SIFTHomographyRectifier

def run_benchmark():
    model_path = "models/svm_model.pkl"
    test_dir = "dataset/test_rotated"
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy mô hình: {model_path}")
        return
    
    svm_model = joblib.load(model_path)
    print("[*] Đang khởi tạo bộ nắn phẳng SIFT...")
    rectifier = SIFTHomographyRectifier(templates_dir="dataset/raw_templates")
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)]
    
    if not test_files:
        print(f"❌ Không tìm thấy ảnh trong thư mục {test_dir}!")
        return

    print(f"\n[*] BẮT ĐẦU ĐÁNH GIÁ CHÍNH XÁC TRÊN TOÀN BỘ {len(test_files)} ẢNH KIỂM THỬ...")
    print("=" * 75)

    records = []
    
    for idx, fname in enumerate(test_files, 1):
        # 1. Tách nhãn thực tế và góc xoay từ tên file (vd: DTHU_rot45.png -> label: DTHU, angle: 45)
        match = re.search(r'^(.*?)_rot(\d+)\.', fname)
        if match:
            true_label = match.group(1)
            angle = int(match.group(2))
        else:
            true_label = os.path.splitext(fname)[0]
            angle = 0

        img_path = os.path.join(test_dir, fname)
        img_color = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_color is None:
            continue
            
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # --- PHƯƠNG PHÁP 1: BASELINE (Bài báo gốc - Không có SIFT) ---
        t0 = time.time()
        base_img = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_AREA)
        feat_base = extract_dtcwt_features(base_img).reshape(1, -1)
        pred_base = svm_model.predict(feat_base)[0]
        time_base = (time.time() - t0) * 1000

        # --- PHƯƠNG PHÁP 2: PROPOSED (Cải tiến - Có SIFT + Homography nắn phẳng) ---
        t1 = time.time()
        is_rect, rect_img, _ = rectifier.rectify_image(img_color)
        feat_prop = extract_dtcwt_features(rect_img).reshape(1, -1)
        pred_prop = svm_model.predict(feat_prop)[0]
        time_prop = (time.time() - t1) * 1000

        # Ghi nhận kết quả
        records.append({
            "File": fname,
            "Goc_Xoay": angle,
            "Nhan_Goc": true_label,
            "Baseline_Doan": pred_base,
            "Baseline_Dung": int(pred_base == true_label),
            "Proposed_Doan": pred_prop,
            "Proposed_Dung": int(pred_prop == true_label),
            "Co_Nan_Phang": int(is_rect),
            "Time_Base_ms": time_base,
            "Time_Prop_ms": time_prop
        })

        if idx % 50 == 0 or idx == len(test_files):
            print(f" -> Đã kiểm thử xong: {idx}/{len(test_files)} ảnh...")

    df = pd.DataFrame(records)

    # Xuất kết quả chi tiết từng ảnh ra file CSV làm minh chứng kèm theo
    os.makedirs("results", exist_ok=True)
    csv_path = "results/chi_tiet_thuc_nghiem.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # --- TỔNG HỢP BẢNG THỐNG KÊ KHOA HỌC THEO TỪNG GÓC XOAY ---
    print("\n" + "=" * 75)
    print("      BẢNG SỐ LIỆU MINH CHỨNG THỰC NGHIỆM CHÍNH XÁC (DÙNG CHO BÁO CÁO)")
    print("=" * 75)
    
    summary = []
    angles = sorted(df["Goc_Xoay"].unique())
    
    for ang in angles:
        df_ang = df[df["Goc_Xoay"] == ang]
        total = len(df_ang)
        acc_base = (df_ang["Baseline_Dung"].sum() / total) * 100
        acc_prop = (df_ang["Proposed_Dung"].sum() / total) * 100
        rect_count = df_ang["Co_Nan_Phang"].sum()
        
        summary.append({
            "Góc xoay (°)": f"{ang}°",
            "Số lượng ảnh": total,
            "Baseline (Gốc) (%)": f"{acc_base:.2f}%",
            "Đề xuất (SIFT) (%)": f"{acc_prop:.2f}%",
            "Độ tăng (Δ)": f"+{acc_prop - acc_base:.2f}%",
            "Số ảnh nắn phẳng thành công": f"{rect_count}/{total}"
        })

    # Dòng tổng trung bình
    total_all = len(df)
    total_acc_base = (df["Baseline_Dung"].sum() / total_all) * 100
    total_acc_prop = (df["Proposed_Dung"].sum() / total_all) * 100
    total_rect = df["Co_Nan_Phang"].sum()

    summary.append({
        "Góc xoay (°)": "TRUNG BÌNH CHUNG",
        "Số lượng ảnh": total_all,
        "Baseline (Gốc) (%)": f"{total_acc_base:.2f}%",
        "Đề xuất (SIFT) (%)": f"{total_acc_prop:.2f}%",
        "Độ tăng (Δ)": f"+{total_acc_prop - total_acc_base:.2f}%",
        "Số ảnh nắn phẳng thành công": f"{total_rect}/{total_all}"
    })

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    print("=" * 75)
    print(f"✅ ĐÃ XUẤT MINH CHỨNG ĐẦY ĐỦ RA FILE: {csv_path}")
    print(f"   Thời gian xử lý trung bình: Baseline = {df['Time_Base_ms'].mean():.1f} ms | Proposed = {df['Time_Prop_ms'].mean():.1f} ms")

if __name__ == "__main__":
    run_benchmark()