import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def render_preprocessing_ui():
    st.header("2. Preprocessing Configuration")
    
    # Kiểm tra điều kiện tiên quyết: Phải có dữ liệu thô từ bước Ingestion
    if 'raw_data' not in st.session_state:
        st.warning("Vui lòng hoàn thành bước 'Data Ingestion' và tải lên dữ liệu trước.")
        return False

    df = st.session_state['raw_data']
    columns = df.columns.tolist()

    # 1. Lựa chọn các biến đặc trưng và mục tiêu
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Chọn cột mục tiêu (Target - y):", columns)
    with col2:
        available_features = [col for col in columns if col != target_col]
        feature_cols = st.multiselect("Chọn các cột đặc trưng (Features - X):", available_features, default=available_features)

    if not feature_cols:
        st.error("Vui lòng chọn ít nhất một cột đặc trưng để tiếp tục.")
        return False

    X = df[feature_cols]
    y = df[target_col]

    # Phân tách tự động các kiểu dữ liệu để xử lý riêng biệt
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    st.subheader("Thiết lập xử lý luồng dữ liệu")

    # 2. Cấu hình xử lý giá trị thiếu (Imputation)
    impute_strategy = None
    if num_cols and X[num_cols].isnull().sum().sum() > 0:
        st.info("Phát hiện giá trị thiếu (Missing Values) trong tập dữ liệu số.")
        impute_strategy = st.selectbox(
            "Phương pháp điền giá trị thiếu (Numerical Imputation):",
            ["mean", "median", "most_frequent"]
        )
    else:
        st.success("Các cột số hiện tại không chứa giá trị khuyết thiếu.")

    # 3. Cấu hình chuẩn hóa thang đo (Scaling)
    scaler_strategy = st.selectbox(
        "Phương pháp chuẩn hóa đặc trưng số (Feature Scaling):",
        ["None", "StandardScaler", "MinMaxScaler"]
    )

    # 4. Xác nhận lưu cấu hình vào pipeline
    if st.button("Áp dụng cấu hình và xem trước dữ liệu"):
        # Tạo bản sao cục bộ để hiển thị Preview cho người dùng
        X_preview = X.copy()

        # Thực hiện Imputation tạm thời cho giao diện Preview
        if num_cols and impute_strategy:
            imputer = SimpleImputer(strategy=impute_strategy)
            X_preview[num_cols] = imputer.fit_transform(X_preview[num_cols])

        # Thực hiện Mã hóa One-Hot Encoding cho các biến phân loại
        if cat_cols:
            X_preview = pd.get_dummies(X_preview, columns=cat_cols, drop_first=True)

        # Thực hiện Scaling tạm thời cho giao diện Preview
        if num_cols and scaler_strategy != "None":
            scaler = StandardScaler() if scaler_strategy == "StandardScaler" else MinMaxScaler()
            X_preview[num_cols] = scaler.fit_transform(X_preview[num_cols])

        # Lưu toàn bộ cấu hình và dữ liệu đã phân tách vào session_state
        st.session_state['preprocessing_config'] = {
            'target_col': target_col,
            'feature_cols': feature_cols,
            'impute_strategy': impute_strategy,
            'scaler_strategy': scaler_strategy,
            'num_cols': num_cols,
            'cat_cols': cat_cols
        }
        st.session_state['X'] = X
        st.session_state['y'] = y
        
        st.success("Đã cấu hình và lưu thông số Preprocessing thành công!")
        st.write("Bản xem trước dữ liệu sau biến đổi (X_processed Preview):")
        st.dataframe(X_preview.head())
        return True

    return False