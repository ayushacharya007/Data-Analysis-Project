import pandas as pd
import streamlit as st


def handle_duplicated_values(data) -> pd.DataFrame:
    '''
    Handle duplicated values in the data.
    
    Args:
        - data: DataFrame containing the data.
        
    Returns:
        - data: DataFrame with duplicated values removed.
    '''

    def check_duplicates(data) -> None:
        '''
        Check for duplicated values in the data.
        
        Args:
            - data: DataFrame containing the data.
            
        Returns:
            - None
        '''
        try:
            duplicated_rows = data[data.duplicated(keep=False)]
            duplicated_rows_sorted = duplicated_rows.sort_values(by=data.columns.tolist())

            with st.expander('Check Duplicated Rows (Whole)', expanded=False):
                if not duplicated_rows.empty:
                    st.write(f':blue[**Total duplicated rows**] : :green[{duplicated_rows.shape[0]}]')
                    st.write(duplicated_rows_sorted)
                else:
                    st.info(':white_check_mark: No duplicated rows found in the data.')
                
            with st.expander('Check Duplicated Values (Selected Columns)', expanded=False):

                selected_columns = st.multiselect(label = '', options= data.columns, key='duplicate_check')
                if selected_columns:
                    duplicated_rows_count = data.duplicated(subset=selected_columns, keep=False).sum()
                    columns_str = ', '.join(selected_columns)
                    st.write(f"**Total duplicated rows in :green[{columns_str}] column is: :blue[{duplicated_rows_count}]**")
                    duplicated_rows = data[data.duplicated(subset=selected_columns, keep=False)]
                    duplicated_rows_sorted = duplicated_rows.sort_values(by=selected_columns)
                    st.write(duplicated_rows_sorted[selected_columns])
                else:
                    st.info(':warning: Please select columns to check for duplicated values.')

        except Exception as e:
            st.info(f":warning: An error occurred: {e}")

    def remove_duplicates_whole(data) -> pd.DataFrame:
        '''
        Remove duplicated values from the whole data.
        
        Args:
            - data: DataFrame containing the data.
            
        Returns:
            - data: DataFrame with duplicated values removed.
        '''
        try:
            if 'remove_duplicates_whole' not in st.session_state:
                st.session_state.remove_duplicates_whole = False

            if whole_remove.button('Remove Duplicates (Whole)', use_container_width=True, type='primary'):
                st.session_state.remove_duplicates_whole = not st.session_state.remove_duplicates_whole

            if st.session_state.remove_duplicates_whole:
                if data.duplicated().any():
                    data = data.drop_duplicates().reset_index(drop=True)
                    st.success(f':white_check_mark: The length of the data frame is :green[{len(data)}] after removing the duplicates.')
                else:
                    st.info(':white_check_mark: No duplicated values to remove.')
            return data

        except Exception as e:
            st.info(f":warning: An error occurred: {e}")


    def remove_duplicates_selected(data) -> pd.DataFrame:
        '''
        Remove duplicated values from selected columns.
        
        Args:
            - data: DataFrame containing the data.
            
        Returns:
            - data: DataFrame with duplicated values removed.
        '''
        try:
            if 'remove_duplicates_selected' not in st.session_state:
                st.session_state.remove_duplicates_selected = False

            if cols_remove.button('Remove Duplicates (Selected Columns)', use_container_width=True, type='primary'):
                st.session_state.remove_duplicates_selected = not st.session_state.remove_duplicates_selected

            if st.session_state.remove_duplicates_selected:
                selected_columns = st.multiselect(label = '', options = data.columns, key='selected_duplicates')
                
                if selected_columns:
                    if data.duplicated(subset=selected_columns).any():
                        data = data.drop_duplicates(subset=selected_columns).reset_index(drop=True)
                        st.info(f':white_check_mark: The length of the data frame is :green[{len(data)}] after removing the duplicates for selected column(s).')
                    else:
                        st.info(':white_check_mark: No duplicated values to remove for selected columns.')
                
                else:
                    st.info(':warning: Please select column(s) to remove duplicated values.')

            return data

        except Exception as e:
            st.info(f":warning: An error occurred: {e}")

    try:
        if data is None:
            st.info(":warning: Please select or upload a file.")
            return None

        else:
            check_duplicates(data)

            whole_remove, cols_remove = st.columns(2, gap='large', vertical_alignment='center')
            data = remove_duplicates_whole(data)
            data = remove_duplicates_selected(data)

        return data
    
    except Exception as e:
        st.info(f":warning: An error occurred: {e}")
