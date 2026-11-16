# Hệ thống nhận diện Logo bất biến phép xoay (SIFT + Homography + Wavelet + SVM)

Dự án nghiên cứu khoa học phát triển hệ thống nhận diện logo các trường đại học tại Việt Nam, khắc phục hiện tượng suy giảm độ chính xác khi ảnh bị xoay góc hoặc biến dạng phối cảnh.

## 🚀 Tính năng chính
- **Khối tiền xử lý**: Định vị logo bằng SIFT, lọc ngoại lai RANSAC và nắn phẳng Homography về chuẩn 256x256.
- **Khối phân loại**: Trích xuất 256 đặc trưng sóng con (Wavelet) và phân lớp bằng SVM (RBF Kernel).
- **Giao diện người dùng (GUI)**: Trực quan hóa ảnh trước/sau nắn phẳng và chuyển đổi giữa mô hình Baseline và Proposed.

## 🛠️ Hướng dẫn cài đặt & Khởi chạy nhanh
1. Chạy file `1_setup_env.bat` để tự động tạo môi trường ảo `.venv` và cài đặt thư viện từ `requirements.txt`.
2. Chạy `4_train_model.bat` để huấn luyện mô hình.
3. Chạy `6_run_app.bat` để mở giao diện nhận diện logo.

## 📊 Kết quả thực nghiệm
- **Baseline (Wavelet + SVM)**: 37.23% (vỡ phổ ở các góc xoay xéo).
- **Proposed (SIFT Homography + Wavelet + SVM)**: 81.12% (tăng trưởng +43.88%).	