import streamlit as st
import numpy as np
import cv2
import os
from skimage.feature import hog
from joblib import load
from labels import class_names  # import dictionary ánh xạ ClassId -> tên biển báo

# ==== Cấu hình ====
BASE_DIR = r"D:\CNTT(PTPM)\NhapMonHocMay\Nhom15_NhanDienBienBaoGiaoThong"
#Mô hình đã được huấn luyện
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.joblib")
RESIZE_SHAPE = (48, 48)
BINS_COLOR = 8

# ==== Load mô hình ====
@st.cache_resource
def load_model(model_path):
    return load(model_path)

# ==== Tiền xử lý ánh sáng bằng CLAHE ====
def enhance_contrast(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# ==== Tự động phát hiện ROI dựa trên màu đỏ ====
def detect_roi(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Ngưỡng màu đỏ (2 dải do đỏ nằm ở đầu và cuối thang Hue)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Tìm contour lớn nhất
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        roi = img_bgr[y:y+h, x:x+w]
        if roi.size > 0:
            return roi
    return img_bgr  # fallback: dùng toàn bộ ảnh

# ==== Tạo vector đặc trưng HOG + Color Histogram ====
def feature_vector(img_bgr):
    img_resized = cv2.resize(img_bgr, RESIZE_SHAPE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm="L2-Hys",
                       transform_sqrt=True)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [BINS_COLOR], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [BINS_COLOR], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [BINS_COLOR], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    return np.concatenate([hog_features, color_features])

# ==== Giao diện Streamlit ====
st.title("Nhận Diện Biển Báo Giao Thông")
st.write("Nhóm 15: Đỗ Quang Minh và Đinh Ngọc Chính")
st.write("Ảnh sẽ được tự động tiền xử lý và phát hiện ROI biển báo.")

# Load mô hình
if not os.path.exists(MODEL_PATH):
    st.error(f"Không tìm thấy mô hình tại: {MODEL_PATH}")
    st.stop()
model = load_model(MODEL_PATH)
st.success("Đã tải mô hình SVM.")

# Upload ảnh
uploaded_file = st.file_uploader("Tải ảnh biển báo (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Không thể đọc ảnh.")
        st.stop()

    # Hiển thị ảnh gốc
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)

    # Tiền xử lý ánh sáng
    img_enhanced = enhance_contrast(img_bgr)
    st.image(cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB), caption="Ảnh sau tiền xử lý ánh sáng", use_container_width=True)

    # Tự động phát hiện ROI
    roi_img = detect_roi(img_enhanced)
    st.image(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB), caption="ROI biển báo đã phát hiện", use_container_width=True)

    # Nút dự đoán
    if st.button("Dự đoán"):
        fv = feature_vector(roi_img).reshape(1, -1)
        pred_class = model.predict(fv)[0]
        st.subheader("Kết quả dự đoán")
        #Hiển Thị Class ID: st.write(f"ClassId: {int(pred_class)}")
        st.write(f"Tên biển báo: **{class_names[int(pred_class)]}**")
