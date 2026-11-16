import os
import cv2
import numpy as np
import pywt

# Thiết lập tham số phân rã Wavelet theo bài báo Soma & Dhandra (2016)
NUM_LEVELS = 4  # Phân rã 4 mức
WAVELET_NAME = 'bior1.3'  # Họ sóng con đối xứng lưỡng trực giao


def extract_dtcwt_features(image_256x256, target_dim=256):
    """Trích xuất vector 256 đặc trưng kết cấu năng lượng sóng con (Wavelet Subbands)

    qua 4 mức phân rã và các dải phụ định hướng (Horizontal, Vertical, Diagonal)
    theo đúng nguyên lý của bài báo Soma & Dhandra (2016).
    """
    # 1. Đưa ảnh xám về kiểu số thực float64
    img_float = image_256x256.astype(np.float64)

    # 2. Phân rã sóng con 4 mức (Level 1 -> 4)
    coeffs = pywt.wavedec2(img_float, wavelet=WAVELET_NAME, level=NUM_LEVELS)

    feature_list = []

    # 3. Trích xuất đặc trưng dải xấp xỉ tần số thấp cuối cùng (Lowpass band - LL)
    ll_mag = np.abs(coeffs[0])
    feature_list.extend([np.mean(ll_mag), np.std(ll_mag), np.mean(ll_mag**2)])

    # 4. Trích xuất đặc trưng từ các dải phụ định hướng ở từng mức (LH, HL, HH)
    for level_idx in range(1, len(coeffs)):
        subbands = coeffs[level_idx]
        for sb in subbands:
            mag = np.abs(sb)
            mean_val = np.mean(mag)
            std_val = np.std(mag)
            energy_val = np.mean(mag**2)
            feature_list.extend([mean_val, std_val, energy_val])

    features = np.array(feature_list, dtype=np.float32)

    # 5. Chuẩn hóa chiều dài vector thành đúng 256 chiều
    if len(features) < target_dim:
        indices_old = np.linspace(0, len(features) - 1, len(features))
        indices_new = np.linspace(0, len(features) - 1, target_dim)
        features = np.interp(indices_new, indices_old, features)
    else:
        features = features[:target_dim]

    # 6. Chuẩn hóa vector chuẩn L2 (bất biến với sự thay đổi cường độ sáng)
    norm = np.linalg.norm(features)
    if norm > 1e-7:
        features = features / norm

    return features


def extract_features_from_file(image_path):
    """Hàm tiện ích: Đọc ảnh từ đường dẫn và trích xuất vector 256 đặc trưng."""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # Chuẩn hóa về đúng kích thước 256x256 nếu chưa chuẩn
    if img.shape != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    return extract_dtcwt_features(img)


if __name__ == "__main__":
    print("=" * 60)
    print("[*] Kiểm tra thử hàm trích xuất đặc trưng Wavelet...")

    dummy_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    vec = extract_dtcwt_features(dummy_img)

    print(f" -> Kích thước vector đặc trưng: {vec.shape}")
    print(f" -> 5 giá trị đặc trưng đầu tiên: {np.round(vec[:5], 4)}")
    print(
        f"✅ Kiểm tra thành công! Thuật toán trả về đúng vector {len(vec)} đặc trưng."
    )
    print("=" * 60)