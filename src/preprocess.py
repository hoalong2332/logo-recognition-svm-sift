import os
import cv2
import numpy as np

# Kích thước chuẩn hóa ảnh theo bài báo gốc Soma & Dhandra (2016)
STANDARD_SIZE = (256, 256)

# 8 góc xoay kiểm thử theo bài báo
TEST_ROTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def load_and_preprocess_image(image_path, target_size=STANDARD_SIZE):
    """
    Đọc ảnh, chuyển sang ảnh xám (Grayscale) và co/giãn về kích thước chuẩn 256x256.
    """
    # Đọc ảnh với hỗ trợ ký tự UTF-8 trên Windows
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Đưa về kích thước chuẩn 256x256
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    return resized


def rotate_image(image, angle):
    """
    Xoay ảnh quanh tâm với góc quay cho trước, giữ nguyên kích thước ma trận và lấp nền trắng (255).
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Tính toán ma trận biến đổi affine cho phép xoay
    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Thực hiện xoay ảnh, nền xung quanh để màu trắng (255)
    rotated = cv2.warpAffine(
        image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    return rotated


def generate_test_dataset(
    raw_dir="dataset/raw_templates", output_dir="dataset/test_rotated"
):
    """
    Tự động quét toàn bộ ảnh mẫu gốc, chuẩn hóa và sinh 8 góc xoay kiểm thử.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    template_files = [
        f
        for f in os.listdir(raw_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    print("=" * 60)
    print(f"[*] Bắt đầu tiền xử lý và sinh tập dữ liệu kiểm thử...")
    print(f"[*] Tìm thấy {len(template_files)} logo mẫu trong thư mục: {raw_dir}")
    print("=" * 60)

    total_generated = 0

    for file_name in template_files:
        base_name, _ = os.path.splitext(file_name)
        input_path = os.path.join(raw_dir, file_name)

        try:
            # 1. Đọc và chuẩn hóa ảnh mẫu
            gray_img = load_and_preprocess_image(input_path)

            # 2. Xoay ảnh ở 8 góc kiểm thử
            for angle in TEST_ROTATION_ANGLES:
                rot_img = rotate_image(gray_img, angle)

                # Quy tắc đặt tên: <TênLogo>_rot<Góc>.png (ví dụ: DTHU_rot45.png)
                out_name = f"{base_name}_rot{angle}.png"
                out_path = os.path.join(output_dir, out_name)

                # Lưu ảnh hỗ trợ đường dẫn tiếng Việt
                is_success, buffer = cv2.imencode(".png", rot_img)
                if is_success:
                    with open(out_path, "wb") as f:
                        f.write(buffer)
                    total_generated += 1

            print(f" -> Đã xử lý {base_name}: sinh thành công 8 góc xoay.")
        except Exception as e:
            print(f" ⚠️ Lỗi khi xử lý {file_name}: {e}")

    print("=" * 60)
    print(f"✅ HOÀN THÀNH TIỀN XỬ LÝ!")
    print(
        f"✅ Đã tạo ra tổng cộng {total_generated} ảnh kiểm thử trong thư mục: {output_dir}"
    )
    print("=" * 60)


if __name__ == "__main__":
    generate_test_dataset()