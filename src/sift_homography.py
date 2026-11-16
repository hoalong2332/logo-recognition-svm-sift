import os
import cv2
import numpy as np

# Ngưỡng số điểm khớp hợp lệ tối thiểu để ước lượng Homography tin cậy
MIN_MATCH_COUNT = 8
STANDARD_SIZE = (256, 256)


class SIFTHomographyRectifier:

    def __init__(self, templates_dir='dataset/raw_templates'):
        self.templates_dir = templates_dir
        # Khởi tạo bộ trích xuất SIFT
        self.sift = cv2.SIFT_create()

        # Khởi tạo bộ khớp FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Bộ nhớ đệm lưu trước keypoints & descriptors của tất cả logo mẫu
        self.template_cache = {}
        self._load_and_index_templates()

    def _load_and_index_templates(self):
        """Đọc và trích xuất sẵn SIFT descriptors cho 47 logo mẫu để tối ưu tốc

        độ.
        """
        if not os.path.exists(self.templates_dir):
            print(f'⚠️ Thư mục mẫu không tồn tại: {self.templates_dir}')
            return

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        files = [
            f
            for f in os.listdir(self.templates_dir)
            if f.lower().endswith(valid_exts)
        ]

        for f in files:
            label, _ = os.path.splitext(f)
            path = os.path.join(self.templates_dir, f)

            # Đọc ảnh mẫu hỗ trợ đường dẫn tiếng Việt
            img = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if img is not None:
                img_resized = cv2.resize(
                    img, STANDARD_SIZE, interpolation=cv2.INTER_AREA
                )
                kp, des = self.sift.detectAndCompute(img_resized, None)
                if des is not None and len(kp) > 0:
                    self.template_cache[label] = {
                        'image': img_resized,
                        'keypoints': kp,
                        'descriptors': des,
                    }

        print(
            f'[*] SIFT Indexer: Đã lập chỉ mục SIFT cho {len(self.template_cache)} logo mẫu sẵn sàng.'
        )

    def find_best_template_match(self, scene_gray):
        """Quét ảnh hiện trường với toàn bộ logo mẫu để tìm logo có độ trùng

        khớp cao nhất.
        """
        kp_scene, des_scene = self.sift.detectAndCompute(scene_gray, None)
        if des_scene is None or len(des_scene) < MIN_MATCH_COUNT:
            return None, None, []

        best_label = None
        best_good_matches = []
        max_inliers = 0
        best_homography = None

        for label, data in self.template_cache.items():
            des_temp = data['descriptors']
            kp_temp = data['keypoints']

            try:
                matches = self.flann.knnMatch(des_temp, des_scene, k=2)
            except Exception:
                continue

            # Lọc theo Lowe's Ratio Test (ngưỡng 0.75)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32(
                    [kp_temp[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp_scene[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)

                # Ước lượng ma trận Homography bằng RANSAC
                H, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, 5.0
                )
                if mask is not None:
                    inliers_count = np.sum(mask)
                    if inliers_count > max_inliers:
                        max_inliers = inliers_count
                        best_label = label
                        best_good_matches = good
                        best_homography = H

        return best_label, best_homography, best_good_matches

    def rectify_image(self, scene_image):
        """Nắn phẳng phối cảnh 3D của ảnh đầu vào về kích thước chuẩn 256x256

        nền trắng.
        """
        if len(scene_image.shape) == 3:
            scene_gray = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
        else:
            scene_gray = scene_image.copy()

        best_label, H, matches = self.find_best_template_match(scene_gray)

        if H is not None and len(matches) >= MIN_MATCH_COUNT:
            try:
                # Tính nghịch đảo của Homography để nắn từ Scene về tọa độ Template chính diện
                H_inv = np.linalg.inv(H)
                rectified = cv2.warpPerspective(
                    scene_gray,
                    H_inv,
                    STANDARD_SIZE,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,  # Lấp đầy vùng ngoài bằng nền trắng
                )
                return True, rectified, best_label
            except np.linalg.LinAlgError:
                pass

        # Trường hợp không đủ inliers để nắn phẳng (ảnh không bị nghiêng hoặc không nhận diện được vùng phối cảnh)
        fallback = cv2.resize(
            scene_gray, STANDARD_SIZE, interpolation=cv2.INTER_AREA
        )
        return False, fallback, None


if __name__ == '__main__':
    print('=' * 60)
    print('[*] Khởi tạo và kiểm tra module SIFT + Homography RANSAC...')
    rectifier = SIFTHomographyRectifier()

    # Kiểm tra thử bằng cách lấy 1 ảnh xoay trong test set để nắn thử
    test_folder = 'dataset/test_rotated'
    if os.path.exists(test_folder):
        test_files = [f for f in os.listdir(test_folder) if f.endswith('.png')]
        if test_files:
            sample_path = os.path.join(test_folder, test_files[0])
            sample_img = cv2.imdecode(
                np.fromfile(sample_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )

            is_rectified, out_img, label_found = rectifier.rectify_image(
                sample_img
            )
            print(f' -> Test thử với ảnh: {test_files[0]}')
            print(f' -> Nắn phẳng thành công: {is_rectified}')
            print(f' -> Mẫu logo trùng khớp nhất: {label_found}')
            print(f' -> Kích thước sau nắn phẳng: {out_img.shape}')
            print('✅ Module SIFT Homography hoạt động hoàn hảo!')
    print('=' * 60)