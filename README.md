# Nhận Diện Biển Báo Giao Thông

_Đề tài áp dụng các thuật toán Học Máy (Machine Learning), cụ thể là SVM, kết hợp với trích xuất đặc trưng truyền thống (HOG + Color Histogram) để nhận diện 43 loại biển báo giao thông và triển khai ứng dụng web bằng Streamlit._



---



### 🌟 **Giới thiệu**  

- **Phân loại biển báo:** Huấn luyện mô hình Support Vector Machine (SVM) để phân loại 43 lớp biển báo giao thông khác nhau.

- **Trích xuất đặc trưng**: Sử dụng kết hợp Histogram of Oriented Gradients (HOG) và Color Histogram (HSV) để tạo vector đặc trưng cho mỗi biển báo.

- **Ứng dụng:** Xây dựng giao diện web (Streamlit) cho phép người dùng tải ảnh lên, tự động phát hiện vùng quan tâm (ROI), và dự đoán loại biển báo.



---



### 🏗️ **Hệ thống**  

#### 📂 **Cấu trúc dự án**  

📦 Project  

├── 📂 data # Chứa dữ liệu ảnh và file CSV (Train.csv, Test.csv)

│ HuanLuyenMoHinh.py # Mã nguồn huấn luyện mô hình SVM và trích xuất đặc trưng.

│ UngDungDuDoan.py # Mã nguồn ứng dụng web Streamlit để dự đoán.

│ labels.py # Dictionary ánh xạ ClassId sang tên biển báo.

│ svm_model.joblib # Mô hình SVM đã được huấn luyện và lưu lại.

│ requirements.txt # Danh sách thư viện cần thiết.  



---



### 🛠️ **Công nghệ sử dụng**  



#### 🖥️ **Phần mềm**  

- **Python (OpenCV, scikit-learn, joblib):** Xử lý ảnh, trích xuất đặc trưng, huấn luyện mô hình SVM.

- **Streamlit:** Framework mã nguồn mở để xây dựng giao diện ứng dụng web tương tác.

- **Pandas, Matplotlib, Seaborn:** Xử lý dữ liệu và trực quan hóa (EDA, so sánh thuật toán, Confusion Matrix).



---



### 🧮 **Thuật toán & Phương pháp**

1. **Tiền xử lý dữ liệu và Trích xuất ROI (Region of Interest):**

   - Dựa trên mô hình YOLO (`yolov8n.pt`) để phát hiện bounding box và phân loại đối tượng (xe hơi, xe tải, ...).
     
   - Đọc dữ liệu từ `Train.csv` và `Test.csv`.
     
   - Cắt ROI: Dựa trên tọa độ ROI có sẵn trong file CSV (trong huấn luyện) hoặc tự động phát hiện ROI dựa trên màu đỏ (trong ứng dụng Streamlit).
     
   - Tăng cường dữ liệu (Data Augmentation): Áp dụng xoay (±15 ∘), lật ngang, thay đổi độ sáng/tương phản để tăng kích thước và đa dạng tập huấn luyện (`HuanLuyenMoHinh.py`).
     
   - Tiền xử lý ánh sáng: Sử dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) trong ứng dụng để cải thiện độ tương phản (`UngDungDuDoan.py`).



2. **Trích xuất đặc trưng (Feature Extraction):**

   - HOG: Trích xuất đặc trưng hình dạng (gradient) từ ảnh xám.
     
   - Color Histogram (HSV): Trích xuất đặc trưng màu sắc từ kênh Hue, Saturation, Value.
     
   - Vector đặc trưng: Nối (concatenate) vector HOG và 3 kênh Color Histogram.



3. **Huấn luyện mô hình:**

   - Sử dụng mô hình Support Vector Machine (SVM) với kernel RBF (Radial Basis Function).
     
   - Áp dụng StandardScaler (chuẩn hóa dữ liệu) trong pipeline trước khi đưa vào SVM.
     
   - Các thuật toán khác được so sánh: KNN, Decision Tree, Random Forest, Logistic Regression (CacThuatToan.py).



4. **Triển khai ứng dụng (Streamlit):**

   - Tải mô hình đã huấn luyện (svm_model.joblib).
     
   - Ảnh đầu vào được áp dụng CLAHE → Phát hiện ROI → Trích xuất đặc trưng → Dự đoán bởi mô hình SVM.
     
   - Hiển thị kết quả dự đoán (tên biển báo) thông qua dictionary class_names (labels.py).



---



### 🚀 **Kết quả và Đánh giá**

  - **Độ chính xác (Accuracy) trên tập Test:** Mô hình **SVM** đạt độ chính xác cao nhất (được ghi nhận trong quá trình huấn luyện: $Accuracy_{Test} \approx 0.96$ - Tùy thuộc vào tham số $C$ và $\gamma$).
    
  - **Giao diện:** Ứng dụng Streamlit hiển thị ảnh gốc, ảnh sau CLAHE, ảnh ROI đã phát hiện, và kết quả dự đoán cuối cùng.



-----



### 🔧 **Hướng dẫn cài đặt và chạy**

1️⃣ **Cài đặt môi trường:**

```bash
pip install pandas numpy opencv-python scikit-learn scikit-image joblib streamlit matplotlib seaborn
```

2️⃣ **Huấn luyện mô hình:**

```bash
python HuanLuyenMoHinh.py
# Kết quả là file svm_model.joblib được lưu.
```

3️⃣ **Chạy ứng dụng web Streamlit:**

```bash
streamlit run UngDungDuDoan.py
```

4️⃣ **Kết quả:**

  - Giao diện web được mở trên trình duyệt, cho phép người dùng tải lên ảnh để nhận diện.



-----



### 🤝 **Đóng góp nhóm**


| Họ và Tên | Vai trò |
|---|---|
| Đỗ Quang Minh | Phát triển thuật toán trích xuất đặc trưng (HOG+Color) và Huấn luyện mô hình SVM. Xây dựng ứng dụng Streamlit. |
| Đinh Ngọc Chính | Tiền xử lý dữ liệu, Data Augmentation. So sánh hiệu năng các thuật toán Học Máy. |



---



© 2025 NHÓM 15, NHẬN DIỆN BIỂN BÁO GIAO THÔNG, NHẬP MÔN HỌC MÁY, TRƯỜNG ĐẠI HỌC ĐẠI NAM
