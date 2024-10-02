import streamlit as st
import pandas as pd

# Load data from file
def load_data(file) -> pd.DataFrame:
    '''
    Load data from the given file object.
    
    Args:
        - file: File object to load data from.

    Returns:
        - data: DataFrame containing the loaded data. Returns None if failed to load.
    '''

    if file is not None:
        
        try:
            csv_encoding = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            
            if file.name.endswith('.csv'):
                for encoding in csv_encoding:
                    try:
                        data = pd.read_csv(file, encoding=encoding)
                        return data  # Return data immediately if read successfully
                    except Exception:
                        pass  # Ignore the exception and try the next encoding

                st.error("Failed to load the CSV file with all tried encodings.")
                return None
            
            elif file.name.endswith(('.xls', '.xlsx')):
                try:
                    data = pd.read_excel(file)
                    return data  # Return data immediately if read successfully
                except Exception as e:
                    st.error(f"Failed to load the Excel file: {e}")
                    return None
            
            else:
                st.error("Unsupported file format")
                return None
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
        
        
        
    return None  # Return None if file is None
