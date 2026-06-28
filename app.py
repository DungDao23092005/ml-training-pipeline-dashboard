import streamlit as st
from src.data_ingestion import render_data_ingestion_ui
from src.preprocessing import render_preprocessing_ui
from src.model_training import render_model_training_ui
from src.evaluation import render_evaluation_ui # Import module đánh giá

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

    # Điều hướng logic tích hợp toàn bộ hệ thống
    if selected_step == "Data Ingestion":
        success = render_data_ingestion_ui()
        if success:
            st.session_state['current_step'] = "Preprocessing"
            
    elif selected_step == "Preprocessing":
        success = render_preprocessing_ui()
        if success:
            st.session_state['current_step'] = "Model Training"
            
    elif selected_step == "Model Training":
        success = render_model_training_ui()
        if success:
            st.session_state['current_step'] = "Evaluation"
            
    elif selected_step == "Evaluation":
        render_evaluation_ui()

if __name__ == "__main__":
    main()