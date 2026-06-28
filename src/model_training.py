import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def run_preprocessing_pipeline(X_train, X_test, config):
    """
    Thực hiện tiền xử lý thực tế trên tập Train và Test độc lập 
    để đảm bảo nguyên tắc chống rò rỉ dữ liệu (Data Leakage).
    """
    num_cols = config['num_cols']
    cat_cols = config['cat_cols']
    impute_strategy = config['impute_strategy']
    scaler_strategy = config['scaler_strategy']

    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()

    # 1. Xử lý giá trị khuyết thiếu (Imputation)
    if num_cols and impute_strategy:
        imputer = SimpleImputer(strategy=impute_strategy)
        # Chỉ TÍNH TOÁN toán học (fit) trên tập Train và áp dụng (transform) lên cả hai
        X_train_proc[num_cols] = imputer.fit_transform(X_train_proc[num_cols])
        X_test_proc[num_cols] = imputer.transform(X_test_proc[num_cols])

    # 2. Mã hóa biến phân loại (One-Hot Encoding)
    if cat_cols:
        X_train_proc = pd.get_dummies(X_train_proc, columns=cat_cols, drop_first=True)
        X_test_proc = pd.get_dummies(X_test_proc, columns=cat_cols, drop_first=True)
        # Đồng bộ cấu trúc cột giữa 2 tập dữ liệu đề phòng trường hợp lệch nhóm dữ liệu chữ
        X_train_proc, X_test_proc = X_train_proc.align(X_test_proc, join='left', axis=1, fill_value=0)

    # 3. Chuẩn hóa dữ liệu (Scaling)
    if num_cols and scaler_strategy != "None":
        scaler = StandardScaler() if scaler_strategy == "StandardScaler" else MinMaxScaler()
        # Tuyệt đối không fit trên tập Test nhằm giữ tính khách quan tuyệt đối cho mô hình
        X_train_proc[num_cols] = scaler.fit_transform(X_train_proc[num_cols])
        X_test_proc[num_cols] = scaler.transform(X_test_proc[num_cols])

    return X_train_proc, X_test_proc

def render_model_training_ui():
    st.header("3. Model Training")

    # Kiểm tra điều kiện ràng buộc đầu vào của hệ thống luồng
    if 'X' not in st.session_state or 'preprocessing_config' not in st.session_state:
        st.warning("Vui lòng hoàn thành và xác nhận cấu hình tại bước 'Preprocessing' trước.")
        return False

    X = st.session_state['X']
    y = st.session_state['y']
    config = st.session_state['preprocessing_config']

    st.subheader("Cấu hình tham số huấn luyện")

    # 1. Thiết lập phân chia tập dữ liệu dữ liệu qua thanh trượt
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Tỷ lệ tập kiểm thử (Test Size Ratio):", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    with col2:
        random_state = st.number_input("Random State (Giữ tính nhất quán kết quả):", min_value=0, value=42, step=1)

    # 2. Lựa chọn thuật toán Machine Learning
    model_type = st.selectbox("Chọn thuật toán phân loại (Algorithm):", ["Random Forest", "Logistic Regression"])

    hyperparameters = {}
    st.markdown("**Điều chỉnh Siêu tham số (Hyperparameters):**")
    
    if model_type == "Random Forest":
        hyperparameters['n_estimators'] = st.slider("Số lượng cây (n_estimators):", min_value=10, max_value=200, value=100, step=10)
        hyperparameters['max_depth'] = st.slider("Độ sâu tối đa (max_depth):", min_value=1, max_value=20, value=5, step=1)
    elif model_type == "Logistic Regression":
        c_val = st.select_slider("Hệ số phạt C (Inverse of regularization strength):", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        hyperparameters['C'] = c_val

    # 3. Kích hoạt tiến trình huấn luyện
    if st.button("Khởi chạy huấn luyện mô hình"):
        with st.spinner("Hệ thống đang phân chia dữ liệu, tiền xử lý và huấn luyện..."):
            
            # Thực hiện chia nhỏ dữ liệu thô từ session_state
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Đưa qua bộ tiền xử lý an toàn độc lập
            X_train_proc, X_test_proc = run_preprocessing_pipeline(X_train, X_test, config)

            # Khởi tạo mô hình tương ứng với tham số đã chọn
            if model_type == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=hyperparameters['n_estimators'],
                    max_depth=hyperparameters['max_depth'],
                    random_state=random_state
                )
            elif model_type == "Logistic Regression":
                model = LogisticRegression(C=hyperparameters['C'], random_state=random_state, max_iter=1000)

            # Thực hiện tối ưu hóa/huấn luyện thuật toán
            model.fit(X_train_proc, y_train)

            # Lưu trữ toàn bộ kết quả cần thiết cho giai đoạn Đánh giá (Evaluation) kế tiếp
            st.session_state['trained_model'] = model
            st.session_state['X_test_proc'] = X_test_proc
            st.session_state['y_test'] = y_test
            st.session_state['model_type'] = model_type

            st.success("Mô hình học máy đã được huấn luyện thành công!")
            st.info("Hệ thống đã sẵn sàng dữ liệu kiểm thử. Vui lòng chọn bước tiếp theo trên thanh điều hướng.")
            return True

    return False