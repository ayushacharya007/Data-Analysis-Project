import streamlit as st
import pandas as pd

def download_clean_file(data):
    '''
    Download the cleaned data as a CSV file.

    Args:
        - data: DataFrame containing the cleaned data.

    Returns:
        - download: Button to download the cleaned data.
    '''
    try:
        if data is not None:
            # Convert DataFrame to CSV, then encode to UTF-8 bytes
            csv = data.to_csv(index=False).encode('utf-8')
            download = st.download_button(
                label='Download Cleaned CSV',
                data=csv,
                file_name='cleaned_data.csv',
                mime='text/csv',
                use_container_width=False,
                type='primary'
            )
            if download:
                # Write a message to the user when the data is downloaded
                st.success('Data downloaded successfully.')
        else:
            st.warning("Data is not available for download.")
    except Exception as e:
        st.error(f'An error occurred: {e}')