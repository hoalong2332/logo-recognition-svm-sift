import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thiết lập đường dẫn
csv_path = 'results/chi_tiet_thuc_nghiem.csv'
chart_save_path = 'results/bieu_do_so_sanh.png'

if not os.path.exists(csv_path):
    print(f'❌ Không tìm thấy file dữ liệu: {csv_path}')
    print(
        'Vui lòng chạy file src/evaluate_benchmark.py trước để tạo file CSV!'
    )
    exit()

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 2. Nhóm và tính toán tỷ lệ chính xác (%) theo từng góc xoay
angles = sorted(df['Goc_Xoay'].unique())
baseline_acc = []
proposed_acc = []
rectified_rate = []

print('=' * 75)
print('          BẢNG THỐNG KÊ TỔNG HỢP KẾT QUẢ THỰC NGHIỆM')
print('=' * 75)
print(
    f"{'Góc xoay':<12} | {'Số mẫu':<8} | {'Baseline (Gốc)':<16} | {'Proposed (SIFT)':<16} | {'Độ tăng (Δ)':<12}"
)
print('-' * 75)

for ang in angles:
    df_ang = df[df['Goc_Xoay'] == ang]
    total = len(df_ang)
    acc_base = (df_ang['Baseline_Dung'].sum() / total) * 100
    acc_prop = (df_ang['Proposed_Dung'].sum() / total) * 100
    rect_cnt = (df_ang['Co_Nan_Phang'].sum() / total) * 100

    baseline_acc.append(acc_base)
    proposed_acc.append(acc_prop)
    rectified_rate.append(rect_cnt)

    diff = acc_prop - acc_base
    sign = '+' if diff >= 0 else ''
    print(
        f'{str(ang) + "°":<12} | {total:<8} | {acc_base:>14.2f}% |'
        f' {acc_prop:>14.2f}% | {sign + f"{diff:.2f}%":>11}'
    )

print('-' * 75)
total_samples = len(df)
mean_base = (df['Baseline_Dung'].sum() / total_samples) * 100
mean_prop = (df['Proposed_Dung'].sum() / total_samples) * 100
diff_mean = mean_prop - mean_base
print(
    f"{'TRUNG BÌNH':<12} | {total_samples:<8} | {mean_base:>14.2f}% |"
    f' {mean_prop:>14.2f}% | +{diff_mean:.2f}%'
)
print('=' * 75)

# 3. Vẽ biểu đồ cột trực quan chuẩn ấn phẩm khoa học
plt.figure(figsize=(11, 6), dpi=300)

x = np.arange(len(angles))
width = 0.35

# Vẽ hai cột màu đại diện cho 2 phương pháp
bars1 = plt.bar(
    x - width / 2,
    baseline_acc,
    width,
    label='Baseline (Wavelet + SVM Gốc)',
    color='#EF4444',
    edgecolor='black',
    alpha=0.85,
)
bars2 = plt.bar(
    x + width / 2,
    proposed_acc,
    width,
    label='Proposed (SIFT + Homography + SVM)',
    color='#10B981',
    edgecolor='black',
    alpha=0.9,
)

# Thêm nhãn số phần trăm trên đầu mỗi cột
for bar in bars1:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1.2,
        f'{yval:.1f}%',
        ha='center',
        va='bottom',
        fontsize=9,
        color='#B91C1C',
        fontweight='bold',
    )

for bar in bars2:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1.2,
        f'{yval:.1f}%',
        ha='center',
        va='bottom',
        fontsize=9,
        color='#047857',
        fontweight='bold',
    )

# Cấu hình trục và tiêu đề
plt.xlabel('Góc xoay kiểm thử (Độ)', fontsize=12, fontweight='bold', labelpad=8)
plt.ylabel('Độ chính xác nhận diện (%)', fontsize=12, fontweight='bold')
plt.title(
    'SO SÁNH ĐỘ CHÍNH XÁC NHẬN DIỆN LOGO GIỮA MÔ HÌNH GỐC VÀ MÔ HÌNH CẢI TIẾN',
    fontsize=13,
    fontweight='bold',
    pad=15,
)
plt.xticks(x, [f'{a}°' for a in angles], fontsize=11)
plt.ylim(0, 115)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='upper right', frameon=True, fontsize=10)

# Lưu ảnh biểu đồ
os.makedirs('results', exist_ok=True)
plt.tight_layout()
plt.savefig(chart_save_path)
plt.close()

print(f'\n📊 ĐÃ TẠO XONG BIỂU ĐỒ TRỰC QUAN TẠI: {chart_save_path}')
print('Bạn có thể mở file ảnh trên để chèn trực tiếp vào báo cáo NCKH!')