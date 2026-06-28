import streamlit as st
import matplotlib.pyplot as plt
import numpy as np # Bổ sung thư viện numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix

def render_evaluation_ui():
    st.header("4. Model Evaluation")

    # Kiểm tra điều kiện đầu vào
    if 'trained_model' not in st.session_state or 'X_test_proc' not in st.session_state:
        st.warning("Vui lòng hoàn thành quá trình 'Model Training' trước khi xem đánh giá.")
        return False

    # Lấy dữ liệu từ bộ nhớ trạng thái
    model = st.session_state['trained_model']
    X_test = st.session_state['X_test_proc']
    y_test = st.session_state['y_test']
    model_type = st.session_state['model_type']

    st.subheader(f"Kết quả đánh giá trên tập kiểm thử - {model_type}")

    with st.spinner("Đang tính toán các chỉ số..."):
        # 1. Thực hiện dự đoán trên tập dữ liệu chưa từng thấy (Test set)
        y_pred = model.predict(X_test)

        # 2. Tính toán các độ đo hiệu suất cốt lõi cho bài toán phân loại
        acc = accuracy_score(y_test, y_pred)
        # Sử dụng average='weighted' để xử lý cả bài toán phân loại nhị phân và đa lớp
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # 3. Trực quan hóa các chỉ số bằng thành phần st.metric
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")

        st.markdown("---")

        # 4. Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix)
        st.subheader("Confusion Matrix (Ma trận nhầm lẫn)")
        st.write("Biểu đồ thể hiện chi tiết số lượng dự đoán đúng và sai cho từng nhãn.")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # FIX: Trích xuất các nhãn thực tế tồn tại trong y_test và y_pred để tránh lỗi bất đồng bộ
        unique_labels = np.unique(np.concatenate((y_test, y_pred)))
        
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        
        # Xoay nhãn trục x nếu số lượng class quá nhiều
        disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
        
        # Hiển thị biểu đồ lên giao diện (đã xóa dòng lặp thừa)
        st.pyplot(fig)

        st.success("Hoàn tất toàn bộ quy trình Machine Learning Pipeline!")
        return True