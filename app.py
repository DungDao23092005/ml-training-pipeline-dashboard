import streamlit as st
from src.data_ingestion import render_data_ingestion_ui

def main():
    st.set_page_config(page_title="ML Training Pipeline", layout="wide")
    st.title("⚙️ Machine Learning Training Pipeline Dashboard")

    # Thiết lập thanh điều hướng
    st.sidebar.title("Điều hướng quy trình")
    steps = ["Data Ingestion", "Preprocessing", "Model Training", "Evaluation"]
    
    # Quản lý trạng thái bước hiện tại
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = "Data Ingestion"

    selected_step = st.sidebar.radio(
        "Chọn giai đoạn:", 
        steps, 
        index=steps.index(st.session_state['current_step'])
    )

    # Điều khiển luồng hoạt động
    if selected_step == "Data Ingestion":
        render_data_ingestion_ui()
    elif selected_step == "Preprocessing":
        st.info("Module Tiền xử lý đang được phát triển.")
    elif selected_step == "Model Training":
        st.info("Module Huấn luyện đang được phát triển.")
    elif selected_step == "Evaluation":
        st.info("Module Đánh giá đang được phát triển.")

if __name__ == "__main__":
    main()