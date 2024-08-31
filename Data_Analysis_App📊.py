## Perfect
import os
import pandas as pd
import numpy as np
import streamlit as st
from dateutil.parser import parse
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud 
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import time
import re
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
# try:
#     import spacy
#     nlp = spacy.load('en_core_web_sm')
# except Exception as e:
#     os.system('python -m spacy download en_core_web_sm')
#     import spacy
#     nlp = spacy.load('en_core_web_sm')
# nltk.download('wordnet')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer


st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

# Center the page title using markdown
st.markdown("<h3 style='text-align: center;'> Data Analysis 📊 </h3>", unsafe_allow_html=True)

# display the data at the top container

# read the style.css file
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# @st.cache_data
def load_data(file):
    if file is not None:
        try:
            csv_encoding = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            if file.name.endswith('.csv'):
                for encoding in csv_encoding:
                    try:
                        data = pd.read_csv(file, encoding=encoding)
                        break
                    except Exception as e:
                        pass
            else:
                st.error("Unsupported file format")
                return None
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
        return data


def correct_data_types(data):
    if data is not None:
        conversion_successful = True

        # Display data types
        og_data_types = pd.DataFrame({
            'Column ': data.columns,
            'Data Types': [data[col].dtype for col in data.columns]
        })

        st.write(":blue[**Original Data Types**] :")
        st.dataframe(og_data_types.style)

        # Let the user select the columns to convert 
        columns = st.multiselect(':blue[**Select columns to convert**] :', data.columns)

        if columns:
            for column in columns:
                data_type = st.selectbox(f':blue[**Select data type for column : :green[{column}]**]', ['None', 'int', 'float', 'object', 'datetime'], key=f'dtype_{column}')
                if data_type != 'None':
                    try:
                        if data_type == 'int':
                            data[column] = data[column].astype(int)
                        elif data_type == 'float':
                            data[column] = data[column].astype(float)
                        elif data_type == 'object':
                            data[column] = data[column].astype(str)
                        elif data_type == 'datetime':
                            data[column] = pd.to_datetime(data[column], errors='coerce')
                    except Exception as e:
                        conversion_successful = False
                        st.error(f'Error converting column ":blue[{column}]" to ":green[{data_type}]"')
                else:
                    st.info('Please select data type to convert')
        else:
            st.warning('Please select columns to convert')

        if conversion_successful and columns and data_type != 'None':
            # Create a DataFrame to show original and new data types
            dtype_df = pd.DataFrame({
                'Column ': columns,
                'New DataType': [data[col].dtype for col in columns]
            })

            # Display the DataFrame in a tabular format
            st.write(":blue[**Converted Data Types**] :")
            st.dataframe(dtype_df)
        
            with st.expander('**Show converted data**'):
                st.write(data)

    return data

# Function to generate alerts
@st.cache_data
def generate_alerts(data):
    alerts = ""
    if data is not None:
        # 1. Check for high correlation
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr().abs()
        high_corr_pairs = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns if i != j and corr_matrix.loc[i, j] > 0.8]
        for i, j in high_corr_pairs:
            alerts += f"<p class='alert high-corr-alert'>{i} is highly correlated with {j}</span><span>High correlation</span></p>"

        # 2. Check for imbalance
        for col in data.columns:
            imbalance_ratio = data[col].value_counts(normalize=True).max()
            if imbalance_ratio > 0.88:
                alerts += f"<p class='alert imbalance-alert'>{col} is highly imbalanced ({imbalance_ratio * 100:.1f}%)<span>Imbalanced</span></p>"

        # 3. Check for missing values
        missing_percent = data.isnull().mean() * 100
        for col, percent in missing_percent.items():
            if percent > 45:
                alerts += f"<p class='alert missing-alert'>{col} has {percent:.1f}% missing values<span>Missing</span></p>"

        # 4. Check for unique values
        for col in data.columns:
            if data[col].nunique() / len(data) > 0.6:
                alerts += f"<p class='alert unique-alert'>{col} has high unique values<span>Unique</span></p>"
    return alerts

@st.cache_data
def data_informations(data):
    with st.container():
         _,stat, _,type,_= st.columns([0.3,3,0.5,2,0.3])
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
                st.markdown(f'<table class="stat-table"><tr><td>{stat[0]}:</td><td>{stat[1]}</td></tr></table>', unsafe_allow_html=True)

            with type:
                st.markdown("<div class='stat-box'><div class='stat-title'>Variable Types</div>", unsafe_allow_html=True)
                variable_types = [str(data[col].dtype) for col in data.columns]
                variable_types_counts = dict(Counter(variable_types))
                variable_types_df = pd.DataFrame.from_dict(variable_types_counts, orient='index', columns=["Count"]).reset_index()
                variable_types_df.columns = ["Type", "Count"]

                # Convert DataFrame to HTML and add CSS classes
                html = variable_types_df.to_html(index=False, classes=['stat-table'])

                st.markdown(html, unsafe_allow_html=True)

@st.cache_data
def data_overview(data):
    with st.container():
        for col in data.columns:
            with st.container():
                _,infos, _, viz,_ = st.columns([0.3,2.5, 0.5, 2,0.3], vertical_alignment='center')
                with infos:
                    st.markdown(f"<div class='info-box'><div class='info-column-title' style='text-align:left; margin-bottom:10px;'>{col}</div>", unsafe_allow_html=True)        
                    # Get the data type of the column
                    col_type = str(data[col].dtype)
                    # Create list of statistics
                    informations = [
                                ["Type", col_type],
                                ["Distinct Values", data[col].nunique()],
                                ["Distinct Percentage", f"{data[col].nunique() / len(data) * 100:.2f}%"],
                                ["Missing Values", data[col].isnull().sum()],
                                ["Missing Percentage", f"{data[col].isnull().mean() * 100:.2f}%"]
                            ]
                    for info in informations:
                        st.markdown(f'<table class="info-table"><tr><td>{info[0]}</td><td>{info[1]}</td></tr></table>', unsafe_allow_html=True)
                    st.write('')
                    st.write('')
                with viz:
                    # Display the information in a table
                    col_type = str(data[col].dtype)
                    # Numeric columns
                    if col_type in ['int64', 'float64']:
                        if data[col].nunique() <= 10 and data[col].nunique() > 3:
                            value_counts = data[col].value_counts().sort_values(ascending=False)
                            top_values = value_counts[:3]
                            others = pd.Series(value_counts[3:].sum(), index=['Others'])
                            plot_data = pd.concat([top_values, others]).reset_index()
                            plot_data.columns = [col, 'count']
                            fig = px.bar(plot_data, y=col, x='count', height=350, width=500)
                            fig.update_xaxes(title_text='')
                            fig.update_yaxes(title_text='')
                            st.plotly_chart(fig)

                        elif data[col].nunique() <= 3:
                            fig = px.bar(data, x=col, height=350, width=500)
                            fig.update_xaxes(title_text='')
                            fig.update_yaxes(title_text='')
                            st.plotly_chart(fig)
                        else:
                            fig = px.histogram(data, x=col, height=350, width=500)
                            fig.update_xaxes(title_text='')
                            fig.update_yaxes(title_text='')
                            fig.update_xaxes(showgrid=False)
                            fig.update_yaxes(showgrid=False)
                            st.plotly_chart(fig)

                    # Object columns
                    elif col_type == 'object':
                        if data[col].nunique() < 10 and data[col].nunique() > 3:
                            value_counts = data[col].value_counts().sort_values(ascending=False)
                            top_values = value_counts[:3]
                            others = pd.Series(value_counts[3:].sum(), index=['Others'])
                            plot_data = pd.concat([top_values, others]).reset_index()
                            plot_data.columns = [col, 'count']
                            fig = px.bar(plot_data, y=col, x='count', height=350, width=500)
                            fig.update_xaxes(title_text='')
                            fig.update_yaxes(title_text='')
                            st.plotly_chart(fig)

                        elif data[col].nunique() <= 3:
                            fig = px.histogram(data, y=col, height=350, width=500)
                            fig.update_xaxes(title_text='')
                            fig.update_yaxes(title_text='')
                            fig.update_xaxes(showgrid=False)
                            fig.update_yaxes(showgrid=False)
                            st.plotly_chart(fig)
                        else:
                            try:
                                # Generate the word cloud
                                wordcloud = WordCloud(width=500, height=300, background_color='white', contour_color='black', contour_width=2).generate(' '.join(data[col].dropna().astype(str)))

                                # Create a matplotlib figure and axis
                                fig, ax = plt.subplots()
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis("off")

                                # Display the figure in Streamlit
                                st.pyplot(fig)
                            except ValueError as e:
                                st.error(f"Could not generate word cloud for column '{col}': {str(e)}")

                    # Datetime columns
                    elif col_type == 'datetime64[ns]':
                        fig = px.histogram(data, x=col, height=350, width=500)
                        fig.update_xaxes(title_text='')
                        fig.update_yaxes(title_text='')
                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)
                        st.plotly_chart(fig)

# Function to show data overview
def show_data_overview(data):
    if data is not None:
        if 'report_profile' not in st.session_state:
            st.session_state.report_profile = False

        if st.button('Generate Report', use_container_width=True):
            st.session_state.report_profile = not st.session_state.report_profile

        if st.session_state.report_profile:
            with st.spinner('Report is being generated...'):
                # Simulate a delay to show the spinner for 5 seconds
                time.sleep(1)
                
                with st.container():
                    overview_tab, alerts_tab, fix_data_type = st.tabs(["Overview", "Alerts", "Fix Data Types"])

                    with fix_data_type:
                        with st.container():
                            correct_data_types(data)

                    with overview_tab:
                        st.write(':blue[**Data Overview**] :')  
                        with st.container():
                            data_informations(data)
                        
                        with st.container():
                            for i in range(1):
                                st.write('')
                            st.write(':blue[**Columns Overview**] :')
                            data_overview(data)

                    with alerts_tab:
                        st.write(':blue[**Data Alerts**] :')
                        with st.container():
                            _, alert, _ = st.columns([0.2, 3, 0.3])
                            with alert:
                                with st.container():
                                    # Generate alerts
                                    alerts = generate_alerts(data)
                                    st.markdown(alerts, unsafe_allow_html=True)
                        with st.container():
                            for i in range(1):
                                st.write('')
                            st.write(':blue[**Columns Overview**] :')
                            data_overview(data)
    else:
        st.warning("Please select or upload a file.")
        

def show_missing(data):
    missing_values = data.isnull().sum()
    missing_percentage = missing_values / data.shape[0] * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values.values, 
        'Percentage %': missing_percentage.values
    }, index=data.columns)
    missing_info.index.name = 'Columns'  # Set the index title
    with st.expander('Show Missing Values', expanded=False):
        st.dataframe(missing_info)

def drop_columns(data):

    if 'drop_columns' not in st.session_state:
        st.session_state.drop_columns = False

    if st.button('Drop the column'):
        st.session_state.drop_columns = not st.session_state.drop_columns

    if st.session_state.drop_columns:
        selected_columns = st.multiselect(':blue[**Select the columns you want to drop**] :', data.columns)
        if selected_columns:
            data = data.drop(selected_columns, axis=1)
            st.success('Selected columns dropped successfully.')
        else:
            st.info('No columns selected for dropping.')
    return data

def drop_data(data):

    if 'drop_data' not in st.session_state:
        st.session_state.drop_data = False

    if st.button('Drop the data'):
        st.session_state.drop_data = not st.session_state.drop_data

    if st.session_state.drop_data:
        null_counts = data.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if columns_with_nulls.empty:
            st.info('No null values to drop.')
        else:
            data = data.dropna()
            formatted_columns = ', '.join([f'{col} ({count})' for col, count in columns_with_nulls.items()])
            st.success(f'Rows with null values have been removed successfully from columns: :blue[{formatted_columns}]')
    return data

# def fill_data(data):

#     if 'fill_data' not in st.session_state:
#         st.session_state.fill_data = False

#     if st.button('Fill data'):
#         st.session_state.fill_data = not st.session_state.fill_data

#     if st.session_state.fill_data:
#         # Let the user select the columns to fill
#         columns = st.multiselect(':blue[**Select columns to fill**] :', data.columns)

#         if columns:
#             for column in columns:
#                 if data[column].isnull().any():
#                     user_input = st.text_input(label=f'Enter value for column {column}', key=f'input_{column}')
#                     if user_input:
#                         data_type = st.selectbox(f':blue[**Select data type for column : :green[{column}]**]', ['None', 'int', 'float', 'object', 'datetime'], key=f'dtype_{column}')
#                         if data_type != 'None':
#                             try:
#                                 if data_type == 'int':
#                                     data[column] = data[column].fillna(int(user_input)).astype(int)
#                                 elif data_type == 'float':
#                                     data[column] = data[column].fillna(float(user_input)).astype(float)
#                                 elif data_type == 'object':
#                                     data[column] = data[column].fillna(str(user_input)).astype(str)
#                                 elif data_type == 'datetime':
#                                     data[column] = data[column].fillna(pd.to_datetime(user_input, errors='coerce')).astype('datetime64[ns]')
#                                 st.write(f'Missing values have been filled successfully in column: {column} with data type: {data_type}')
#                             except Exception as e:
#                                 st.error(f"Error converting column {column} to {data_type}: {e}")
#                         else:
#                             st.info('Please select a data type to convert')
#                 else:
#                     st.info(f'The column "{column}" does not have any null values.')

#             show_data.write(data)
#         else:
#             st.warning('Please select columns to fill')

#         st.write(data)
#     return data

def replace_values(data):
    if 'replace_data' not in st.session_state:
        st.session_state.replace_data = False

    if st.button('Replace values'):
        st.session_state.replace_data = not st.session_state.replace_data

    if st.session_state.replace_data:
        st.info('Only use mean and median with numerical data type.')
        columns = st.multiselect(':blue[**Select columns to fill**] :', data.columns)

        if columns:
            for column in columns:
                selected_method = st.selectbox(f':blue[**Select filling method for column**] : :green[{column}]', ['None', 'Mean', 'Mode', 'Median', 'Custom Value'], key=f'method_{column}')

                if selected_method == 'None':
                    st.warning(f'Please select a filling method for column :blue[{column}]')
                elif data[column].isnull().any():
                    if selected_method == 'Custom Value':
                        user_input = st.text_input(label=f':blue[**Enter value for column**] : :green[{column}]', key=f'input_{column}')
                        if user_input:
                            try:
                                data[column] = data[column].fillna(user_input)
                                st.success(f'Missing values have been filled successfully in column ":blue[{column}]"')
                            except Exception as e:
                                st.error(f'Error converting column ":blue[{column}]" to ":green[{data[column].dtype}]": {e}')
                    elif selected_method in ['Mean', 'Mode', 'Median']:
                        try:
                            if selected_method == 'Mean':
                                if data[column].dtype in ['int64', 'float64']:
                                    mean = data[column].mean()
                                    st.write(f'**The mean for the ":blue[{column}]" column is :green[{mean}]**.')
                                    data[column] = data[column].fillna(mean)
                                else:
                                    st.error(f'Cannot calculate mean for column ":blue[{column}]" with ":green[{data[column].dtype}]" data type.')
                            elif selected_method == 'Mode':
                                mode = data[column].mode()[0]
                                st.write(f'The mode for the ":blue[{column}]" column is :green[{mode}].')
                                data[column] = data[column].fillna(mode)
                            elif selected_method == 'Median':
                                if data[column].dtype in ['int64', 'float64']:
                                    median = data[column].median()
                                    st.write(f'The median for the ":blue[{column}]" column is :green[{median}].')
                                    data[column] = data[column].fillna(median)
                                else:
                                    st.error(f'Cannot calculate median for column ":blue[{column}]" with ":green[{data[column].dtype}]" data type.')
                        except Exception as e:
                            st.error(f'An error occurred: {e}')
                else:
                    st.warning(f'No Null Values to write in ":blue[{column}]"')    
            # show_data.write(data)
        else:
            st.warning('Please select columns to fill')
    return data

# def encode_data(data):
#     if st.session_state.show_missing:
#         if 'handle_missing' not in st.session_state:
#             st.session_state.handle_missing = False

#         if st.button('Encoding', use_container_width=True):
#             st.session_state.handle_missing = not st.session_state.handle_missing

#         if st.session_state.handle_missing:
#             st.write('Select the columns you want to encode')
#             selected_columns = st.multiselect('Columns', data.columns, key='encoding')
#             st.write('Selected Columns:')
#             selected_columns
#             encoding_type = st.selectbox('Select encoding type', ['One-Hot Encoding', 'Label Encoding'])
#             if encoding_type == 'One-Hot Encoding':
#                 data = pd.get_dummies(data, columns=selected_columns)
#             elif encoding_type == 'Label Encoding':
#                 for col in selected_columns:
#                     le = LabelEncoder()
#                     data[col] = le.fit_transform(data[col])
#             st.write(data)
#     return data

def missing_values(data):
    show_missing(data)
    with st.expander('Select methods to handle null', expanded=False):
        data = drop_columns(data)
        data = drop_data(data)
        data = replace_values(data)
    # data = encode_data(data)
    return data

def check_duplicates(data):
    # count the total duplicated rows
    duplicated_rows = data.duplicated().sum()
    # show the duplicated rows
    duplicated_rows = data[data.duplicated(keep=False)]
    duplicated_rows_sorted = duplicated_rows.sort_values(by=data.columns.tolist())
    with st.expander('Show Duplicated Rows', expanded=False):
        if duplicated_rows.shape[0] > 0:
            st.write(f':blue[**Total duplicated rows**] : :green[{duplicated_rows.shape[0]}]')
            st.write(duplicated_rows_sorted)
            st.write('')
        else:
            st.info('No duplicated rows found. The dataset is clean.')

        for i in range(2):
            st.write('')
    
        selected_columns = st.multiselect(':blue[**Select the columns you want to see the duplicated values**] :', data.columns, key='duplicate_check')

        if selected_columns:  # Ensure there's at least one column selected
            # Adjusted to include keep=False to mark all duplicates as True
            duplicated_rows_count = data.duplicated(subset=selected_columns, keep=False).sum()
            # Join the selected_columns list into a string separated by commas
            columns_str = ', '.join(selected_columns)
            st.write(f"**Total duplicated rows in :green[{columns_str}] column is: :blue[{duplicated_rows_count}]**")
            # Adjusted to include keep=False to get all duplicated rows
            duplicated_rows = data[data.duplicated(subset=selected_columns, keep=False)]
            # Sort the duplicated rows by the selected columns to display them one above the other
            duplicated_rows_sorted = duplicated_rows.sort_values(by=selected_columns)
            # Display only the selected columns
            
            st.write(duplicated_rows_sorted[selected_columns])
        else:
            st.info('Please select columns to check for duplicated values.')
    
    return data

def remove_duplicates_whole(data):

# create a button to remove the duplicated rows
    if 'remove_duplicates_whole' not in st.session_state:
        st.session_state.remove_duplicates_whole = False

    if st.button('Remove Duplicates (Whole)'):
        st.session_state.remove_duplicates_whole = not st.session_state.remove_duplicates_whole

    if st.session_state.remove_duplicates_whole:
        # check if there are duplicated rows
        if data.duplicated().any():
            # remove the duplicated rows
            data = data.drop_duplicates().reset_index(drop=True)
            # show_data.write(data)
            st.info(f'Duplicates have been removed successfully. The length of the data frame is :green[{len(data)}] after removing the duplicates.')
        else:
            st.info('No duplicated values to remove.')
    return data

def remove_duplicates_selected(data):
    if 'remove_duplicates_selected' not in st.session_state:
            st.session_state.remove_duplicates_selected = False

    if st.button('Remove Duplicates (Selected Columns)'):
        st.session_state.remove_duplicates_selected = not st.session_state.remove_duplicates_selected

    if st.session_state.remove_duplicates_selected:
        # check if there are duplicated rows based on selected columns
        selected_columns = st.multiselect(':blue[**Select the columns to remove duplicates**] :', data.columns)
        if selected_columns:
            if data.duplicated(subset=selected_columns).any():
                # remove the duplicated rows based on selected columns
                data = data.drop_duplicates(subset=selected_columns).reset_index(drop=True)
                # show_data.write(data)
                st.info(f'Duplicates have been removed successfully for selected columns. The length of the data frame is :green[{len(data)}] after removing the duplicates for selected columns.')
            else:
                st.info('No duplicated values to remove for selected columns.')
                
    return data


def duplicated_values(data):
    check_duplicates(data)
    with st.expander('Select methods to handle duplicates', expanded=False):
        data = remove_duplicates_whole(data)
        data = remove_duplicates_selected(data)
    return data

def display_and_download_cleaned_data(data):
    if data is not None:
        # Apply data cleaning steps
        with st.expander('Show Modified/Cleaned Data', expanded=False):
            show_data = st.empty()
        missing, duplicated, download_csv=st.tabs(['Missing Values', 'Duplicated Values', 'Download Cleaned Data'])
        
        with  st.container():
            with missing:  
                data = missing_values(data)
                show_data.write(data)
                
            with duplicated:
                data = duplicated_values(data)
                show_data.write(data)
            
            with download_csv:
                if data is not None:
                    # Convert DataFrame to CSV, then encode to UTF-8 bytes
                    csv = data.to_csv(index=False).encode('utf-8')
                    download= st.download_button(label='Download Cleaned CSV', data=csv, file_name='cleaned_data.csv', mime='text/csv', use_container_width=True)
                    if download:
                        # write a message to the user when the data is downloaded
                        st.write(':green[**Downladed the cleaned data successfully.**]')
                else:
                    st.warning("Data is not available for download.")
    else:
        st.warning("Please select or upload a file.")


@st.cache_data
def get_available_plot_types(x_dtype, y_dtype, x_axis, y_axis):
    # Check if both x and y are the same column
    if x_axis == y_axis:
        return ['Count Plot']
    elif x_dtype in ['float64', 'int64'] and y_dtype in ['float64', 'int64']:
        return ['Scatter Plot', 'Line Graph', 'Heat Map', 'Bubble Chart']
    elif (x_dtype in ['float64', 'int64'] and y_dtype == object) or (x_dtype == object and y_dtype in ['float64', 'int64']):
        return ['Bar Chart (Sum)', 'Bar Chart (Average)', 'Bar Chart (Count Distinct)', 'Stacked Bar Chart', 'Box Plot', 'Pie Chart']
    elif (x_dtype == 'datetime64[ns]' and y_dtype in ['float64', 'int64']) or (x_dtype in ['float64', 'int64'] and y_dtype == 'datetime64[ns]'):
        return ['Line Chart (Sum)', 'Line Chart (Average)', 'Area Chart (Sum)', 'Area Chart (Average)']
    elif (x_dtype == 'datetime64[ns]' and y_dtype == 'object') or (x_dtype == 'object' and y_dtype == 'datetime64[ns]'):
        return ['Line Chart (Count Distinct)']
    elif x_dtype == object and y_dtype == object:
        return ['Pie Chart(Count)', 'Stacked Bar Chart', 'Count Plot']
    else:
        return ['None']
    
# Function to create visualizations
def create_visualization(data, x_axis, y_axis, plot_type):
    fig = None
    if plot_type == 'Scatter Plot':
        fig = px.scatter(data, x=x_axis, y=y_axis)
    elif plot_type == 'Line Graph':
        fig = px.line(data, x=x_axis, y=y_axis)
    elif plot_type == 'Heat Map':
        fig = px.density_heatmap(data, x=x_axis, y=y_axis)
    elif plot_type == 'Bubble Chart':
        fig = px.scatter(data, x=x_axis, y=y_axis)
    elif plot_type == 'Bar Chart (Sum)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.bar(data.groupby(y_axis)[x_axis].sum().reset_index(), x=y_axis, y=x_axis, color=y_axis, barmode='group')
        else:
            fig = px.bar(data.groupby(x_axis)[y_axis].sum().reset_index(), x=x_axis, y=y_axis, color=x_axis, barmode='group')
        fig.update_layout(barmode='group', xaxis={'categoryorder':'total descending'})
    elif plot_type == 'Bar Chart (Average)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.bar(data.groupby(y_axis)[x_axis].mean().reset_index(), x=y_axis, y=x_axis, color=y_axis, barmode='group')
        else:
            fig = px.bar(data.groupby(x_axis)[y_axis].mean().reset_index(), x=x_axis, y=y_axis, color=x_axis, barmode='group')
        fig.update_layout(barmode='group', xaxis={'categoryorder':'total descending'})
    elif plot_type == 'Bar Chart (Count Distinct)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.bar(data.groupby(y_axis)[x_axis].nunique().reset_index(), x=y_axis, y=x_axis, color=y_axis, barmode='group')
        else:
            fig = px.bar(data.groupby(x_axis)[y_axis].nunique().reset_index(), x=x_axis, y=y_axis, color=x_axis, barmode='group')
        fig.update_layout(barmode='group', xaxis={'categoryorder':'total descending'})
    elif plot_type == 'Stacked Bar Chart':
        fig = px.bar(data.groupby([x_axis, y_axis]).size().reset_index(), x=x_axis, y=0, color=y_axis)
        fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    elif plot_type == 'Box Plot':
        fig = px.box(data, x=x_axis, y=y_axis)
    elif plot_type == 'Line Chart (Sum)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.line(data.groupby(y_axis)[x_axis].sum().reset_index(), x=y_axis, y=x_axis)
        else:
            fig = px.line(data.groupby(pd.Grouper(key=x_axis, freq='D'))[y_axis].sum().reset_index(), x=x_axis, y=y_axis)
    elif plot_type == 'Line Chart (Average)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.line(data.groupby(y_axis)[x_axis].mean().reset_index(), x=y_axis, y=x_axis)
        else:
            fig = px.line(data.groupby(pd.Grouper(key=x_axis, freq='D'))[y_axis].mean().reset_index(), x=x_axis, y=y_axis)
    elif plot_type == 'Area Chart (Sum)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.area(data.groupby(y_axis)[x_axis].sum().reset_index(), x=y_axis, y=x_axis)
        else:
            fig = px.area(data.groupby(pd.Grouper(key=x_axis, freq='D'))[y_axis].sum().reset_index(), x=x_axis, y=y_axis)
    elif plot_type == 'Area Chart (Average)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.area(data.groupby(y_axis)[x_axis].mean().reset_index(), x=y_axis, y=x_axis)
        else:
            fig = px.area(data.groupby(pd.Grouper(key=x_axis, freq='D'))[y_axis].mean().reset_index(), x=x_axis, y=y_axis)
    elif plot_type == 'Line Chart (Count Distinct)':
        if data[x_axis].dtype == 'object':
            fig = px.line(data.groupby(y_axis)[x_axis].nunique().reset_index(), x=y_axis, y=x_axis)
        else:
            fig = px.line(data.groupby(pd.Grouper(key=x_axis, freq='D'))[y_axis].nunique().reset_index(), x=x_axis, y=y_axis)
    elif plot_type == 'Pie Chart':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.pie(data.groupby(y_axis)[x_axis].sum().reset_index(), values=x_axis, names=y_axis)
        else:
            fig = px.pie(data.groupby(x_axis)[y_axis].count().reset_index(), values=y_axis, names=x_axis)
    elif plot_type == 'Pie Chart(Count)':
        fig = px.pie(data.groupby(x_axis).size().reset_index(), values=0, names=x_axis)
    elif plot_type == 'Count Plot':
        fig = px.histogram(data, x=x_axis)
        fig.update_layout(barmode='group', xaxis={'categoryorder':'total descending'})
    elif plot_type == 'Bar Chart':
        fig = px.bar(data.groupby(x_axis)[y_axis].count().reset_index(), x=x_axis, y=y_axis, color=x_axis, barmode='group')
        fig.update_layout(barmode='group', xaxis={'categoryorder':'total descending'})
    return fig

# Function to handle data visualization
def display_visualizations(data):
    if data is not None:
        # display_dataset(data)
        # display_columns(data)

        if data is not None:
            x_axis = st.selectbox(':blue[**Select the x-axis value**] :', [None] + data.columns.to_list())
            y_axis = st.selectbox(':blue[**Select the y-axis value**] :', [None] + data.columns.to_list())

            if x_axis is not None and y_axis is not None:
                x_axis_dtype = data[x_axis].dtype
                y_axis_dtype = data[y_axis].dtype
                available_plot_types = get_available_plot_types(x_axis_dtype, y_axis_dtype, x_axis, y_axis)

                plot_type = st.selectbox(':blue[**Select the type of plot**] :', available_plot_types)

                if plot_type != 'None':
                    fig = create_visualization(data, x_axis, y_axis, plot_type)
                    if fig is not None:
                        st.write(fig)
                    else:
                        st.warning("No visualization available for the selected options.")
                else:
                    st.warning("Please select a valid plot type.")
            else:
                st.warning("Please select both x-axis and y-axis values.")
        else:
            st.warning("No data available for visualization.")
    else:
        st.warning("Please select or upload a file.")

def train_classification_model(pipe, X_train, y_train, X_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_pred

def train_regression_model(pipe, X_train, y_train, X_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_pred

def train_nlp_model(pipe, X_train, y_train, X_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_pred

# Define the function to preprocess text data
def preprocess_text_data(data, feature_column, target_column):
    # Ensure the feature column is treated as a string and handle NaNs
    data = data.dropna(subset=[feature_column, target_column])
    feature = data[feature_column].astype(str).fillna('')

    # Clean the feature column
    feature = feature.str.replace('[^a-zA-Z]', ' ', regex=True)
    feature = feature.str.lower()
    # lemmatize the words and get rid of stop words using spacy
    # not working now skip
    feature = feature.apply(lambda x: ' '.join([word for word in x.split()]))
    # stop_words = set(stopwords.words('english'))
    # feature = feature.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    # lemmatizer = WordNetLemmatizer()
    # feature = feature.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    # if feature.str.strip().eq('').all():
    #     st.error('The feature column is empty after preprocessing. Please check the data and try again.')

    # Encode the target column if it's a string
    if isinstance(target_column, str):
        y = data[target_column]
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
    else:
        y = target_column

    return feature, y


def model_building(data):
    if data is not None:
        # with st.expander('Orignal Data', expanded=False):
        #     data

        problem_type = st.radio(':blue[**Select the problem type**] : ', ['None','Classification', 'Regression', 'Sentiment Analysis'],horizontal=True)

        if problem_type != 'None':
            if problem_type == 'Classification':
                models = st.radio(':blue[**Select the models**] :', ['None','Random Forest', 'Logistic Regression', 'K-Nearest Neighbors'], horizontal=False)
                if models != 'None':

                    target_column = st.selectbox(':blue[**Select the target column**] :', [None] + list(data.columns))
                    features = st.multiselect(':blue[**Select the features**] : ', list(data.columns))

                    if target_column is not None and features:
                        X = data[features]
                        y = data[target_column]

                        # Handle NaN values in the target column
                        if y.isnull().any():
                            st.warning('The target column contains NaN values. Rows with NaN values is dropped.')
                            data = data.dropna(subset=[target_column])
                            X = data[features]
                            y = data[target_column]
                        
                        else:
                            pass

                        try:
                            if not X.empty and not y.empty:
                                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                                categorical_features = X.select_dtypes(include=['object']).columns
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                                _, class_train, class_test, _ = st.columns([0.1, 2, 2, 0.1])
                                with st.container():
                                    with class_train:
                                        st.write(':blue[**Train Data**] :')
                                        st.write(X_train)

                                    with class_test:
                                        st.write(':blue[**Test Data**] :')
                                        st.write(X_test)

                            preprocessor_classification = ColumnTransformer(
                                transformers=[
                                    ('cat', Pipeline([
                                        ('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                                    ]), categorical_features),
                                    ('num', Pipeline([
                                        ('imputer', SimpleImputer(strategy='mean')),
                                        ('scaler', StandardScaler())
                                    ]), numeric_features)
                                ],
                                remainder='passthrough'  # Keep the rest of the columns as they are
                            )

                            rf_pipeline = Pipeline([
                                ('preprocessor', preprocessor_classification),
                                ('classifier', RandomForestClassifier())
                            ])

                            lr_pipeline = Pipeline([
                                ('preprocessor', preprocessor_classification),
                                ('classifier', LogisticRegression())
                            ])

                            knn_pipeline = Pipeline([
                                ('preprocessor', preprocessor_classification),
                                ('classifier', KNeighborsClassifier())
                            ])
                            
                            label_encoder = LabelEncoder()
                            y_train = label_encoder.fit_transform(y_train)
                            y_test = label_encoder.transform(y_test)

                            if models == 'Random Forest':
                                y_pred = train_classification_model(rf_pipeline, X_train, y_train, X_test)

                            elif models == 'Logistic Regression':
                                y_pred = train_classification_model(lr_pipeline, X_train, y_train, X_test)

                            elif models == 'K-Nearest Neighbors':
                                y_pred = train_classification_model(knn_pipeline, X_train, y_train, X_test)

                            st.write(':blue[**Accuracy**] :', accuracy_score(y_test, y_pred))
                            st.write(':blue[**Confusion Matrix**] :', confusion_matrix(y_test, y_pred))
                            report = classification_report(y_test, y_pred, output_dict=True)
                            st.write(':blue[**Classification Report**] :')
                            st.dataframe(pd.DataFrame(report).transpose())

                        except ValueError as e:
                            st.error(f"Error: Cannot preprocess the data: {e}")

                    else:
                        st.warning('Please select the target column and features.')
                else: 
                    st.warning('Please select a model.')

            elif problem_type == 'Regression':
                models = st.radio(':blue[**Select the models**] :', ['None','Random Forest', 'Linear Regression', 'SVR'])
                if models != 'None':
                    target_column = st.selectbox(':blue[**Select the target column**] : ', [None] + list(data.columns))
                    features = st.multiselect(':blue[**Select the features**] : ', list(data.columns))
                    if target_column is not None and features:
                        X = data[features]
                        y = data[target_column]

                        # Handle NaN values in the target column
                        if y.isnull().any():
                            st.warning('The target column contains NaN values. Rows with NaN values is dropped.')
                            data = data.dropna(subset=[target_column])
                            X = data[features]
                            y = data[target_column]
                        
                        else:
                            pass

                        try:
                            if not X.empty and not y.empty:
                                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                                categorical_features = X.select_dtypes(include=['object']).columns
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                                _, reg_train, reg_test, _ = st.columns([0.1, 2, 2, 0.1])
                                with st.container():
                                    with reg_train:
                                        st.write(':blue[**Train Data**] :')
                                        st.write(X_train)

                                    with reg_test:
                                        st.write(':blue[**Test Data**] :')
                                        st.write(X_test)
                            
                                # Define the ColumnTransformer with imputers, encoders, and scalers
                                preprocessor_regressor = ColumnTransformer(
                                    transformers=[
                                        ('cat', Pipeline([
                                            ('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                                        ]), categorical_features),
                                        ('num', Pipeline([
                                            ('imputer', SimpleImputer(strategy='mean')),
                                            ('scaler', StandardScaler())
                                        ]), numeric_features)
                                    ],
                                    remainder='passthrough'  # Keep the rest of the columns as they are
                                )

                                # Create a pipeline for the regression model
                                rf_pipeline = Pipeline([
                                    ('preprocessor', preprocessor_regressor),
                                    ('regressor', RandomForestRegressor())
                                ])

                                lr_pipeline = Pipeline([
                                    ('preprocessor', preprocessor_regressor),
                                    ('regressor', LinearRegression())
                                ])

                                svm_pipeline = Pipeline([
                                    ('preprocessor', preprocessor_regressor),
                                    ('regressor', SVR())
                                ])

                                if models == 'Random Forest':
                                    y_pred = train_regression_model(rf_pipeline, X_train, y_train, X_test)

                                elif models == 'Linear Regression':
                                    y_pred = train_regression_model(lr_pipeline, X_train, y_train, X_test)

                                elif models == 'SVR':
                                    y_pred = train_regression_model(svm_pipeline, X_train, y_train, X_test)
                                
                                st.write(':blue[**Mean Squared Error**] :', mean_squared_error(y_test, y_pred))
                                st.write(':blue[**R2 Score**] :', r2_score(y_test, y_pred))
                            
                        except ValueError as e:
                            st.error(f"Error: Cannot preprocess the data: {e}")

                    else:
                        st.warning('Please select the target column and features.')

                else:
                    st.warning('Please select a model.')

            elif problem_type == 'Sentiment Analysis':
                models = st.radio(':blue[**Select the models**] :', ['None', 'Logistic Regression','Random Forest', 'Multinomial Naive Bayes'])
                if models != 'None':
                    target_column = st.selectbox(':blue[**Select the target column**] :', [None] + list(data.columns))
                    if target_column != None:
                        feature = st.selectbox(':blue[**Select the feature column**] :', [None] + list(data.columns))
                        if feature != None and data[feature].dtype == 'object':
                            try:
                                X, y = preprocess_text_data(data, feature, target_column)

                                if X.size > 0 and y.size > 0:
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                                    _, nlp_train, nlp_test, _ = st.columns([0.1, 2, 2, 0.1])
                                    with nlp_train:
                                        st.write(':blue[**Train Data**] :')
                                        st.write(X_train)

                                    with nlp_test:
                                        st.write(':blue[**Test Data**] :')
                                        st.write(X_test)
                                    
                                else:
                                    pass

                                # create a pipeline for the nlp model
                                lr_pipeline = Pipeline([
                                    ('preprocessor', TfidfVectorizer()),
                                    ('classifier', LogisticRegression())
                                ])

                                rf_pipeline = Pipeline([
                                    ('preprocessor', TfidfVectorizer()),
                                    ('classifier', RandomForestClassifier())
                                ])

                                naive_bayes_pipeline = Pipeline([
                                    ('preprocessor', TfidfVectorizer()),
                                    ('classifier', MultinomialNB())
                                ])

                                if models == 'Logistic Regression':
                                    y_pred = train_nlp_model(lr_pipeline, X_train, y_train, X_test)

                                elif models == 'Random Forest':
                                    y_pred = train_nlp_model(rf_pipeline, X_train, y_train, X_test)

                                elif models == 'Multinomial Naive Bayes':
                                    y_pred = train_nlp_model(naive_bayes_pipeline, X_train, y_train, X_test)

                                st.write(':blue[**Accuracy**] :', accuracy_score(y_test, y_pred))
                                # show the real and predicted values in a dataframe
                                st.write(':blue[**Real and Predicted Values**] :')
                                values = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
                                st.write(values)
                                st.write(':blue[**Confusion Matrix**] :', confusion_matrix(y_test, y_pred))
                                st.write(':blue[**Classification Report**] :')
                                report = classification_report(y_test, y_pred, output_dict=True)
                                st.dataframe(pd.DataFrame(report).transpose())

                            except ValueError as e:
                                st.error(f"Error: Cannot preprocess the text data: {e}")
                            
                        else:
                            st.info('Please select a text feature.')
                    else:
                        st.warning('Please select a target column.')
                else:
                    st.warning('Please select a model.')
            
            # save the model
            save_model = joblib.dump(models, 'model.pkl')

            # Read the saved model file in binary mode
            with open('model.pkl', 'rb') as file:
                model_data = file.read()

            # Let the user download the model
            if model_data is not None:
                st.download_button(label='Download Model', data=model_data, file_name='model.pkl', mime='application/octet-stream')
        else:
            st.warning('Please select a problem type to do the analysis.')

    else:
        st.warning('Please select or upload a file.')

# Add file uploader and file selector to the sidebar
uploaded_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv'])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    with st.expander("View Original Data", expanded=False):
        original_data = st.write(data)

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(['Data Overview', 'Data Cleaning', 'Visualizations', 'Model Building'])

    with tab1:
        show_data_overview(data)

    with tab2:
        display_and_download_cleaned_data(data)

    with tab3:
        display_visualizations(data)
    with tab4:
        model_building(data)
else:
    st.warning('Please upload a file to get started.')
        
