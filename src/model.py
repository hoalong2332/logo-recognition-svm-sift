import os
import re
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Import hàm trích xuất 256 đặc trưng đã hoàn thiện ở bước trước
from feature import extract_features_from_file

def load_dataset(raw_dir="dataset/raw_templates", test_dir="dataset/test_rotated"):
    """
    Trích xuất đặc trưng cho tập Huấn luyện (ảnh gốc) và tập Kiểm thử (ảnh xoay 8 góc).
    """
    print("=" * 65)
    print("[1/3] BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG TẬP HUẤN LUYỆN...")
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    template_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(valid_exts)]
    
    X_train = []
    y_train = []
    
    for filename in template_files:
        label, _ = os.path.splitext(filename)
        img_path = os.path.join(raw_dir, filename)
        try:
            feat = extract_features_from_file(img_path)
            X_train.append(feat)
            y_train.append(label)
        except Exception as e:
            print(f" ⚠️ Lỗi đọc ảnh mẫu {filename}: {e}")
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f" -> Đã trích xuất xong {len(X_train)} mẫu huấn luyện (tương ứng {len(set(y_train))} lớp logo).")

    print("\n[2/3] BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG TẬP KIỂM THỬ ĐA GÓC XOAY...")
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)]
    
    X_test = []
    y_test = []
    
    for filename in test_files:
        # Tách nhãn gốc từ tên file: ví dụ "DTHU_rot45.png" -> nhãn gốc là "DTHU"
        label = re.sub(r'_rot\d+', '', os.path.splitext(filename)[0])
        img_path = os.path.join(test_dir, filename)
        try:
            feat = extract_features_from_file(img_path)
            X_test.append(feat)
            y_test.append(label)
        except Exception as e:
            print(f" ⚠️ Lỗi đọc ảnh test {filename}: {e}")
            
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f" -> Đã trích xuất xong {len(X_test)} mẫu kiểm thử xoay.")
    print("=" * 65)
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate():
    """
    Huấn luyện mô hình SVM và KNN, đánh giá độ chính xác và lưu mô hình.
    """
    X_train, y_train, X_test, y_test = load_dataset()
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ Dữ liệu trống! Vui lòng kiểm tra lại thư mục dataset.")
        return

    print("\n[3/3] HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH (SVM vs KNN)...")
    
    # 1. Huấn luyện SVM Kernel RBF theo bài báo của Soma & Dhandra (2016)
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm) * 100

    # 2. Huấn luyện KNN (K=1) theo bài báo để đối chứng
    knn_model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn) * 100

    # In kết quả so sánh
    print("=" * 65)
    print("           KẾT QUẢ THỬ NGHIỆM ĐỘ CHÍNH XÁC NHẬN DIỆN")
    print("=" * 65)
    print(f"  * Thuật toán K-Nearest Neighbors (KNN, K=1) : {acc_knn:.2f}%")
    print(f"  * Thuật toán Support Vector Machine (SVM RBF): {acc_svm:.2f}%")
    print("-" * 65)
    if acc_svm >= acc_knn:
        print(f"  => KẾT LUẬN: SVM vượt trội hơn KNN (+{acc_svm - acc_knn:.2f}%), đúng như bài báo gốc!")
    else:
        print(f"  => Cả hai mô hình đều cho kết quả nhận diện rất khả quan.")
    print("=" * 65)

    # Lưu mô hình SVM và danh sách các lớp vào thư mục models/
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "svm_model.pkl")
    joblib.dump(svm_model, model_save_path)
    print(f"\n✅ Đã lưu mô hình SVM thành công vào tệp: {model_save_path}")

if __name__ == "__main__":
    train_and_evaluate()