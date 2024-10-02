import streamlit as st
from Codes.data_cleaning.missing_values import handle_missing_values

from Codes.data_cleaning.duplicated_values import handle_duplicated_values

from Codes.data_cleaning.download_clean import download_clean_file

def clean_data_interface(data):
    '''
    Interface to clean the data.
    
    Args:
        - data: DataFrame containing the data.
        
    Returns:
        - data: DataFrame containing the cleaned data.
    '''
    try:
        if data is None:
            st.info(":warning: Please select or upload a file.")
            return None

        else:
            with st.expander('Show Modified/Cleaned Data', expanded=False):
                show_data = st.empty()

            # Create a radio button to select the operation
            missing, duplicates, download = st.tabs(
                ['Missing Values', 'Duplicated Values', 'Download CSV']
            )

            with missing:
                data = handle_missing_values(data)
                show_data.dataframe(data)

            with duplicates:
                data = handle_duplicated_values(data)
                show_data.dataframe(data)

            with download:
                download_clean_file(data)

            return data

    except Exception as e:
        st.error(f'An error occurred: {e}')
        return None
