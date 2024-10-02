import pandas as pd
import streamlit as st

def handle_missing_values(data) -> pd.DataFrame:
    '''
    Handle missing values in the data.
    
    Args:
        - data: DataFrame containing the data.
        
    Returns:
        - data: DataFrame with missing values handled.
    '''
    def show_missing_values(data) -> None:
        ''' 
        Display missing values in the data.

        Args:
            - data: DataFrame containing the data.
        '''
        try:
            missing_values = data.isnull().sum()
            missing_percentage = (missing_values / data.shape[0]) * 100
            missing_info = pd.DataFrame({
                'Missing Values': missing_values,
                'Percentage %': missing_percentage
            }, index=data.columns)
            missing_info.index.name = 'Columns'
            
            with st.expander('Show Missing Values', expanded=False):
                st.dataframe(missing_info)
        
        except Exception as e:
            st.info(f":warning: An error occurred: {e}")

    def drop_columns(data) -> pd.DataFrame:
        '''
        Drop columns from the data.

        Args:
            - data: DataFrame containing the data.

        Returns:
            - data: DataFrame with columns dropped.
        '''
        try:
            if 'drop_columns' not in st.session_state:
                st.session_state.drop_columns = False

            if cols_drop.button('Drop the column', use_container_width=True, type = 'primary'):
                st.session_state.drop_columns = not st.session_state.drop_columns

            if st.session_state.drop_columns:
                selected_columns = st.multiselect(label = '', options = data.columns, key='column_drop')
                if selected_columns:
                    data = data.drop(selected_columns, axis=1)
                    st.info(':white_check_mark: Selected columns dropped successfully.')
                else:
                    st.info(':warning: Please select column(s) to drop.')
                
                st.write('---')
            
            return data

        except Exception as e:
            st.info(f':warning: An error occurred: {e}')

    def drop_data(data) -> pd.DataFrame:
        '''
        Drop rows with missing values from the data.

        Args:
            - data: DataFrame containing the data.

        Returns:
            - data: DataFrame with rows dropped.
        '''
        try:
            if 'drop_data' not in st.session_state:
                st.session_state.drop_data = False

            if data_drop.button('Drop the data', use_container_width=True, type = 'primary'):
                st.session_state.drop_data = not st.session_state.drop_data

            if st.session_state.drop_data:
                null_counts = data.isnull().sum()
                columns_with_nulls = null_counts[null_counts > 0]
                if columns_with_nulls.empty:
                    st.info(':white_check_mark: No null values to drop.')
                else:
                    data = data.dropna()
                    formatted_columns = ', '.join([f'{col} ({count})' for col, count in columns_with_nulls.items()])
                    st.success(f':white_check_mark: Rows with null values have been removed successfully from columns: {formatted_columns}')
                st.write('---')

            return data

        except Exception as e:
            st.info(f':warning: An error occurred: {e}')

    def replace_values(data) -> pd.DataFrame:
        '''
        Replace missing values in the data.
        
        Args:
            - data: DataFrame containing the data.
            
        Returns:
            - data: DataFrame with missing values replaced.
        '''
        try:
            if 'fill_data' not in st.session_state:
                st.session_state.fill_data = False

            if data_fill.button('Fill values', use_container_width=True, type = 'primary'):
                st.session_state.fill_data = not st.session_state.fill_data

            if st.session_state.fill_data:
                columns = st.multiselect(label = '', options = data.columns)
                if columns:
                    for column in columns:
                        if data[column].isnull().any():
                            selected_method = st.selectbox(label = ':blue[**Select filling methods :**]', options = ['None', 'Custom Value', 'Mean', 'Mode', 'Median'],key=f'select_{column}')
                            if selected_method == 'None':
                                st.info(f':warning: Please select a filling method for column {column}')
                            elif selected_method == 'Custom Value':
                                user_input = st.text_input(f':blue[**Enter custom value for column :**] :green[**{column}**]', key=f'input_{column}')
                                if user_input:
                                    try:
                                        if data[column].dtype in ['float64', 'int64']:
                                            user_input = float(user_input)
                                        elif data[column].dtype == 'datetime64[ns]':
                                            user_input = pd.to_datetime(user_input)
                                        elif data[column].dtype == 'object':
                                            user_input = str(user_input)
                                        else:
                                            raise ValueError("Unsupported data type")

                                        data[column] = data[column].fillna(user_input)
                                        st.success(f':white_check_mark: Missing values have been filled successfully in column {column}')
                                    except ValueError as ve:
                                        st.info(f':warning: An error occurred: {ve}')
                                    except Exception as e:
                                        st.info(f':warning: An error occurred: {e}')
                                else:
                                    st.info(':warning: Please enter a value to fill the missing values.')

                            elif selected_method in ['Mean', 'Mode', 'Median']:
                                try:
                                    if selected_method == 'Mean':
                                        if data[column].dtype in ['int64', 'float64']:
                                            mean = data[column].mean()
                                            data[column] = data[column].fillna(mean)
                                            st.success(f':white_check_mark: Successfully filled missing values with mean value {mean}')
                                        else:
                                            st.error(f':warning: Cannot calculate mean for column {column} with {data[column].dtype} data type.')

                                    elif selected_method == 'Mode':
                                        mode = data[column].mode()[0]
                                        data[column] = data[column].fillna(mode)
                                        st.success(f':white_check_mark: Successfully filled missing values with mode value {mode}')

                                    elif selected_method == 'Median':
                                        if data[column].dtype in ['int64', 'float64']:
                                            median = data[column].median()
                                            data[column] = data[column].fillna(median)
                                            st.success(f':white_check_mark: Successfully filled missing values with median value {median}')
                                        else:
                                            st.info(f':warning: Cannot calculate median for column {column} with {data[column].dtype} data type.')
                                except Exception as e:
                                    st.info(f':warning: An error occurred: {e}')
                        else:
                            st.info(f':white_check_mark: No missing values to fill in column {column}')
                else:
                    st.info(':warning: Please select columns to fill missing values.')
            return data

        except Exception as e:
            st.info(f':warning: An error occurred: {e}')

    try: 
        if data is None:
            st.info(':warning: Please select or upload a file.')
            return None

        else:
            show_missing_values(data)
            cols_drop, data_drop, data_fill = st.columns(3, gap='large', vertical_alignment='center')
            data = drop_columns(data)
            data = drop_data(data)
            data = replace_values(data)
            return data
        
    except Exception as e:
        st.info(f":warning: An error occurred: {e}")

   
