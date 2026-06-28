import streamlit as st
from src.data_ingestion import render_data_ingestion_ui
from src.preprocessing import render_preprocessing_ui

def main():
    st.set_page_config(page_title="ML Training Pipeline", layout="wide")
    st.title("⚙️ Machine Learning Training Pipeline Dashboard")

    st.sidebar.title("Điều hướng quy trình")
    steps = ["Data Ingestion", "Preprocessing", "Model Training", "Evaluation"]
    
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = "Data Ingestion"

    selected_step = st.sidebar.radio(
        "Chọn giai đoạn:", 
        steps, 
        index=steps.index(st.session_state['current_step'])
    )

    # Điều khiển luồng hoạt động logic qua các bước chuyển đổi
    if selected_step == "Data Ingestion":
        success = render_data_ingestion_ui()
        if success:
            st.session_state['current_step'] = "Preprocessing"
            
    elif selected_step == "Preprocessing":
        success = render_preprocessing_ui()
        if success:
            st.session_state['current_step'] = "Model Training"
            
    elif selected_step == "Model Training":
        st.info("Module Huấn luyện mô hình đang đợi cấu hình từ bước tiền xử lý.")
    elif selected_step == "Evaluation":
        st.info("Module Đánh giá đang được phát triển.")

if __name__ == "__main__":
    main()