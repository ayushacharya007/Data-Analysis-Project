import streamlit as st
import pandas as pd
from collections import Counter

@st.cache_data
def data_informations(data) -> None:
    '''
    Display detailed information about the dataset.
    
    Args:
        - data: DataFrame to display information for.
    '''
    try:
        with st.container():
            _, stat, _, type_, _ = st.columns([0.3, 3, 0.5, 2, 0.3])
            
            with stat:
                st.markdown("<div class='stat-box'><div class='stat-title'>Dataset Statistics</div>", unsafe_allow_html=True)

                # Calculate statistics
                missing_cells = data.isnull().sum().sum()
                missing_cells_percentage = (missing_cells / (data.shape[0] * data.shape[1])) * 100
                duplicate_rows = data.duplicated().sum()
                duplicate_rows_percentage = (duplicate_rows / data.shape[0]) * 100

                # Create list of statistics
                statistics = [
                    ["Number of variables", data.shape[1]],
                    ["Number of observations", data.shape[0]],
                    ["Missing cells", missing_cells],
                    ["Missing cells (%)", f"{missing_cells_percentage:.2f}%"],
                    ["Duplicate rows", duplicate_rows],
                    ["Duplicate rows (%)", f"{duplicate_rows_percentage:.2f}%"]
                ]

                for stat in statistics:
                    st.markdown(
                        f'<table class="stat-table"><tr><td>{stat[0]}:</td><td>{stat[1]}</td></tr></table>',
                        unsafe_allow_html=True
                    )

            with type_:
                st.markdown("<div class='stat-box'><div class='stat-title'>Variable Types</div>", unsafe_allow_html=True)
                
                variable_types = [str(data[col].dtype) for col in data.columns]
                variable_types_counts = dict(Counter(variable_types))
                variable_types_df = pd.DataFrame.from_dict(variable_types_counts, orient='index', columns=["Count"]).reset_index()
                variable_types_df.columns = ["Type", "Count"]

                # Convert DataFrame to HTML and add CSS classes
                html = variable_types_df.to_html(index=False, classes=['stat-table'])
                st.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        st.info(f":warning: An error occurred: {e}")