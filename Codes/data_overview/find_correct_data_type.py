import streamlit as st
import pandas as pd

def correct_data_types(data) -> pd.DataFrame:
    '''
    Correct the data types of the DataFrame.
    
    Args:
        - data: DataFrame to correct data types for.
        
    Returns:
        - data: DataFrame with corrected data types.
    '''
    try:
        if data is not None:
            
            st.write(":blue[**Original Data Types**] :")
            st.dataframe(data.dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column'}))

            columns = st.multiselect(':blue[**Select columns to convert**] :', data.columns)

            if columns:
                for column in columns:
                    data_type = st.selectbox(f':blue[**Select data type for column : :green[{column}]**]', ['None', 'int', 'float', 'object', 'datetime'], key=f'dtype_{column}')

                    if data_type != 'None':
                        try:
                            if data_type == 'int':
                                 data[column] = pd.to_numeric(data[column], errors='coerce').astype(int)

                            elif data_type == 'float':
                                data[column] = pd.to_numeric(data[column], errors='coerce').astype(float)

                            elif data_type == 'object':
                                data[column] = data[column].astype(str)
                                
                            elif data_type == 'datetime':
                                data[column] = pd.to_datetime(data[column], errors='coerce')

                            st.write(":blue[**Converted Data Types**] :")
                            st.dataframe(data.dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column'}))

                            with st.expander('**Show converted data**'):
                                st.write(data)

                        except Exception as e:
                            st.info(f':warning: Error converting column ":blue[{column}]" to ":green[{data_type}]"')
                    else:
                        st.info(':warning: Please select data type to convert')

            else:
                st.info(':warning: Please select columns to convert')

        return data

    except Exception as e:
        st.info(f":warning: An error occurred: {e}")