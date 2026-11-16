import os
import sys
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import joblib

# Thêm đường dẫn thư mục hiện tại để gọi các module đã viết
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature import extract_dtcwt_features
from sift_homography import SIFTHomographyRectifier

class LogoRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG NHẬN DIỆN LOGO ĐẠI HỌC BẰNG WAVELET & SVM (HỖ TRỢ NẮN PHẲNG SIFT)")
        self.root.geometry("1100x720")
        self.root.configure(bg="#F4F6F9")

        # Đường dẫn tài nguyên
        self.model_path = "models/svm_model.pkl"
        self.templates_dir = "dataset/raw_templates"

        # Tải mô hình và module nắn phẳng
        self._load_resources()

        # Biến trạng thái
        self.current_image_path = None
        self.current_cv_image = None
        self.use_sift_var = tk.BooleanVar(value=True)

        # Xây dựng giao diện
        self._build_ui()

    def _load_resources(self):
        """Tải mô hình SVM và khởi tạo bộ nắn phẳng SIFT Homography."""
        if not os.path.exists(self.model_path):
            messagebox.showerror("Lỗi", f"Không tìm thấy mô hình: {self.model_path}\nHãy chạy huấn luyện file src/model.py trước!")
            self.svm_model = None
        else:
            self.svm_model = joblib.load(self.model_path)

        print("[*] Đang khởi tạo bộ nắn phẳng SIFT...")
        self.rectifier = SIFTHomographyRectifier(templates_dir=self.templates_dir)

    def _build_ui(self):
        # 1. Tiêu đề chính
        header_frame = tk.Frame(self.root, bg="#1E3A8A", height=60)
        header_frame.pack(fill=tk.X)
        title_label = tk.Label(
            header_frame,
            text="HỆ THỐNG NHẬN DIỆN LOGO BẰNG THUẬT TOÁN WAVELET VÀ SVM",
            font=("Arial", 16, "bold"),
            fg="#FFFFFF",
            bg="#1E3A8A",
            pady=12
        )
        title_label.pack()

        # 2. Thanh điều khiển
        control_frame = tk.Frame(self.root, bg="#FFFFFF", relief=tk.RAISED, bd=1, pady=10)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        btn_load = tk.Button(
            control_frame,
            text="📂 Chọn ảnh Logo kiểm thử...",
            font=("Arial", 11, "bold"),
            bg="#2563EB",
            fg="white",
            padx=15,
            pady=5,
            command=self.load_image
        )
        btn_load.pack(side=tk.LEFT, padx=15)

        # Hộp chọn chế độ: Gốc hay Cải tiến
        chk_sift = tk.Checkbutton(
            control_frame,
            text="Bật module nắn phẳng 3D (SIFT + Homography RANSAC - Đề xuất cải tiến)",
            font=("Arial", 11),
            variable=self.use_sift_var,
            bg="#FFFFFF",
            activebackground="#FFFFFF"
        )
        chk_sift.pack(side=tk.LEFT, padx=15)

        btn_predict = tk.Button(
            control_frame,
            text="🔍 Tiến hành Nhận diện",
            font=("Arial", 11, "bold"),
            bg="#059669",
            fg="white",
            padx=20,
            pady=5,
            command=self.predict_logo
        )
        btn_predict.pack(side=tk.RIGHT, padx=15)

        # 3. Khu vực hiển thị 3 khung hình
        display_frame = tk.Frame(self.root, bg="#F4F6F9")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # Khung 1: Ảnh gốc
        self.frame_input, self.lbl_img_input = self._create_image_card(display_frame, "1. Ảnh kiểm thử đầu vào")
        self.frame_input.grid(row=0, column=0, padx=10, sticky="nsew")

        # Khung 2: Ảnh sau tiền xử lý / nắn phẳng
        self.frame_rect, self.lbl_img_rect = self._create_image_card(display_frame, "2. Ảnh tiền xử lý / Nắn phẳng")
        self.frame_rect.grid(row=0, column=1, padx=10, sticky="nsew")

        # Khung 3: Ảnh mẫu khớp từ Database
        self.frame_match, self.lbl_img_match = self._create_image_card(display_frame, "3. Mẫu khớp trong Database")
        self.frame_match.grid(row=0, column=2, padx=10, sticky="nsew")

        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.columnconfigure(2, weight=1)

        # 4. Bảng kết quả dự đoán
        result_box = tk.LabelFrame(self.root, text="KẾT QUẢ PHÂN TÍCH NHẬN DIỆN", font=("Arial", 11, "bold"), bg="#FFFFFF", fg="#1E3A8A", padx=15, pady=10)
        result_box.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_result_name = tk.Label(result_box, text="Logo nhận diện: [Chưa có dữ liệu]", font=("Arial", 13, "bold"), fg="#DC2626", bg="#FFFFFF")
        self.lbl_result_name.pack(anchor="w", pady=2)

        self.lbl_confidence = tk.Label(result_box, text="Độ tin cậy SVM: --", font=("Arial", 11), bg="#FFFFFF")
        self.lbl_confidence.pack(anchor="w", pady=2)

        self.lbl_details = tk.Label(result_box, text="Trạng thái xử lý: Sẵn sàng | Thời gian: --", font=("Arial", 10, "italic"), fg="#4B5563", bg="#FFFFFF")
        self.lbl_details.pack(anchor="w", pady=2)

    def _create_image_card(self, parent, title):
        card = tk.LabelFrame(parent, text=title, font=("Arial", 10, "bold"), bg="#FFFFFF", fg="#374151")
        lbl = tk.Label(card, text="Chưa tải ảnh\n(256x256)", bg="#E5E7EB", fg="#6B7280", font=("Arial", 10))
        lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        return card, lbl

    def _display_cv_image_on_label(self, cv_img, label_widget):
        """Chuyển đổi ảnh OpenCV sang định dạng hiển thị trong Tkinter (250x250)."""
        if cv_img is None:
            return
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if len(cv_img.shape) == 3 else cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (250, 250), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(img_resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        label_widget.configure(image=tk_img, text="")
        label_widget.image = tk_img

    def load_image(self):
        """Mở hộp thoại chọn ảnh kiểm thử."""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh Logo",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not file_path:
            return

        self.current_image_path = file_path
        # Đọc ảnh hỗ trợ tiếng Việt
        self.current_cv_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if self.current_cv_image is None:
            messagebox.showerror("Lỗi", "Không thể đọc tệp ảnh đã chọn!")
            return

        # Hiển thị vào Khung 1
        self._display_cv_image_on_label(self.current_cv_image, self.lbl_img_input)
        # Xóa các khung hiển thị cũ
        self.lbl_img_rect.configure(image="", text="Đang chờ nhận diện...")
        self.lbl_img_match.configure(image="", text="Đang chờ nhận diện...")
        self.lbl_result_name.configure(text="Logo nhận diện: [Đã nạp ảnh mới]", fg="#2563EB")
        self.lbl_confidence.configure(text="Độ tin cậy SVM: --")
        self.lbl_details.configure(text=f"Tệp: {os.path.basename(file_path)}")

    def predict_logo(self):
        """Thực hiện chu trình nhận diện: Nắn phẳng -> Wavelet -> SVM."""
        if self.current_cv_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh logo đầu vào trước!")
            return
        if self.svm_model is None:
            messagebox.showerror("Lỗi", "Mô hình SVM chưa được nạp!")
            return

        start_time = time.time()
        is_rectified = False
        processed_img = None

        # GIAI ĐOẠN 1: TIỀN XỬ LÝ / NẮN PHẲNG PHỐI CẢNH 3D
        if self.use_sift_var.get():
            is_rectified, processed_img, best_sift_label = self.rectifier.rectify_image(self.current_cv_image)
            status_text = "Đã nắn phẳng 3D bằng SIFT + Homography" if is_rectified else "Không phát hiện góc nghiêng (giữ nguyên kích thước 256x256)"
        else:
            # Chế độ bài báo gốc: Chỉ co giãn về 256x256 trên ảnh xám
            gray = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2GRAY)
            processed_img = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
            status_text = "Chế độ bài báo gốc (Baseline): Không nắn phẳng"

        # Hiển thị Khung 2: Ảnh sau tiền xử lý
        self._display_cv_image_on_label(processed_img, self.lbl_img_rect)

        # GIAI ĐOẠN 2: TRÍCH XUẤT 256 ĐẶC TRƯNG WAVELET
        feature_vec = extract_dtcwt_features(processed_img)
        feature_vec = feature_vec.reshape(1, -1)

        # GIAI ĐOẠN 3: PHÂN LOẠI BẰNG SVM
        predicted_label = self.svm_model.predict(feature_vec)[0]
        probabilities = self.svm_model.predict_proba(feature_vec)[0]
        max_prob = np.max(probabilities) * 100

        elapsed_time = (time.time() - start_time) * 1000

        # GIAI ĐOẠN 4: HIỂN THỊ KẾT QUẢ VÀ ẢNH MẪU ĐỐI CHỨNG
        self.lbl_result_name.configure(text=f"Logo nhận diện: {predicted_label}", fg="#059669")
        self.lbl_confidence.configure(text=f"Độ tin cậy (Confidence): {max_prob:.2f}% | Thuật toán: SVM RBF")
        self.lbl_details.configure(text=f"Trạng thái: {status_text} | Thời gian xử lý: {elapsed_time:.1f} ms")

        # Tìm ảnh mẫu chuẩn trong database để hiển thị Khung 3
        template_img_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.webp'):
            candidate = os.path.join(self.templates_dir, f"{predicted_label}{ext}")
            if os.path.exists(candidate):
                template_img_path = candidate
                break

        if template_img_path:
            match_cv = cv2.imdecode(np.fromfile(template_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self._display_cv_image_on_label(match_cv, self.lbl_img_match)

if __name__ == "__main__":
    root = tk.Tk()
    app = LogoRecognitionApp(root)
    root.mainloop()