import pandas as pd
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Lỗi định dạng tệp: {e}")
        return None

def render_data_ingestion_ui():
    st.header("1. Data Ingestion")
    st.write("Vui lòng tải lên tập dữ liệu định dạng CSV để bắt đầu quy trình.")
    
    uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("Tải dữ liệu thành công!")
            st.write("Bản xem trước dữ liệu (Data Preview):")
            st.dataframe(df.head())
            # Lưu trữ dataframe vào session_state để sử dụng cho các giai đoạn sau
            st.session_state['raw_data'] = df
            return True
    return False