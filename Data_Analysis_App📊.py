## Perfect
import streamlit as st

# data overview
from Codes.data_overview.load_data import load_data
from Codes.data_overview.data_overview_tab import show_data_overview

# data cleaning
from Codes.data_cleaning.data_clean_tab import clean_data_interface

# data visualization
from Codes.data_visualization.visualization_tab import display_visualizations

# model building
from Codes.model_building.model_tab import select_and_run_model


st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

# Center the page title using markdown
st.markdown("<h3 style='text-align: center;'> Data Analysis 📊 </h3>", unsafe_allow_html=True)

# read the style.css file
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add file uploader and file selector to the sidebar
uploaded_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv', 'xlsx', 'xls'])

try:
    if uploaded_file is not None:
        data = load_data(uploaded_file)

        if data is not None:

            with st.expander("View Original Data", expanded=False):
                original_data = st.write(data)

            # Tabs for different sections
            overview, cleaning, visualization, model = st.tabs(['Data Overview', 'Data Cleaning', 'Visualizations', 'Model Building'])

            with overview:
                show_data_overview(data)

            with cleaning:
                clean_data_interface(data)

            with visualization:
                display_visualizations(data)

            with model:
                select_and_run_model(data)

    else:
        st.info('Please upload a file to get started.')
    
except Exception as e:
    st.info(f":warning: An error occurred: {e}")
