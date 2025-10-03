import pandas as pd
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from joblib import dump, load

# ==== Cấu hình ====
base_dir = r"D:\CNTT(PTPM)\NhapMonHocMay\Nhom15_NhanDienBienBaoGiaoThong"
train_csv_path = os.path.join(base_dir, "Train.csv")
test_csv_path = os.path.join(base_dir, "Test.csv")
resize_shape = (48, 48)
bins_color = 8

# ==== 1. Khám phá dữ liệu (EDA) ====
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Phân phối nhãn
plt.figure(figsize=(12,6))
train_df['ClassId'].value_counts().sort_index().plot(kind='bar')
plt.title("Phân phối số lượng ảnh theo từng lớp (Train)")
plt.xlabel("ClassId")
plt.ylabel("Số lượng ảnh")
plt.show()

# Hiển thị một vài ảnh mẫu từ mỗi lớp
classes = train_df['ClassId'].unique()
plt.figure(figsize=(15, 15))
for i, c in enumerate(classes[:25]):  # hiển thị 25 lớp đầu tiên
    sample_path = os.path.join(base_dir, train_df[train_df['ClassId']==c].iloc[0]['Path'])
    img = cv2.imread(sample_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(5, 5, i+1)
    plt.imshow(img)
    plt.title(f"Class {c}")
    plt.axis("off")
plt.show()

# ==== 2. Hàm tăng cường dữ liệu ====
def augment_image(img):
    aug_images = []

    # Xoay ±15 độ
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, Sh))
    aug_images.append(rotated)

    # Lật ngang
    flipped = cv2.flip(img, 1)
    aug_images.append(flipped)

    # Thay đổi độ sáng/độ tương phản
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.randint(-20, 20)    # brightness
    bright_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    aug_images.append(bright_contrast)

    return aug_images

# ==== Hàm tạo vector đặc trưng HOG + màu ====
def feature_vector(img):
    # HOG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       transform_sqrt=True)

    # Color Histogram (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins_color], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins_color], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins_color], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])

    return np.concatenate([hog_features, color_features])

# ==== Hàm trích xuất đặc trưng ====
def extract_features(df, augment=False):
    X = []
    y = []
    for idx, row in df.iterrows():
        img_path = os.path.join(base_dir, row['Path'])
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Crop ROI
        x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])
        roi = img[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, resize_shape)

        # Đặc trưng từ ảnh gốc
        features = feature_vector(roi_resized)
        X.append(features)
        if 'ClassId' in df.columns:
            y.append(int(row['ClassId']))

        # Nếu augment=True thì tạo thêm ảnh biến đổi
        if augment:
            aug_imgs = augment_image(roi_resized)
            for aug in aug_imgs:
                features_aug = feature_vector(aug)
                X.append(features_aug)
                if 'ClassId' in df.columns:
                    y.append(int(row['ClassId']))

        if idx % 1000 == 0:
            print(f"Đã xử lý {idx}/{len(df)} ảnh")

    return np.array(X), np.array(y)

# ==== 3. Trích xuất đặc trưng + Huấn luyện + Đánh giá ====
print("Đang trích xuất đặc trưng Train (có tăng cường dữ liệu)...")
X_train, y_train = extract_features(train_df, augment=True)

print("Đang trích xuất đặc trưng Test...")
X_test, y_test = extract_features(test_df, augment=False)

print("Đang huấn luyện mô hình SVM...")
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy trên tập Test: {acc:.4f}")

# ==== 4. Lưu mô hình ra file ====
model_path = os.path.join(base_dir, "svm_model.joblib")
dump(svm_model, model_path)
print(f"Đã lưu mô hình vào: {model_path}")
