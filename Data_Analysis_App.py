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
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import time
import re
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
# from modelling.linear_regression import perform_linear_regression


st.set_page_config(page_title="Write AI Data Analysis", page_icon="📊", layout="wide")

# read the style.css file
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Center the page title using markdown
st.markdown("<h1>Data Analysis APP</h1>", unsafe_allow_html=True)

# Create a sidebar radio button for file selection
selected_option = st.sidebar.radio('Select a file for analysis', ['Upload a new file'])


        
@st.cache_data
def load_data(file):
    if file is not None:
        # try different encoding types to read the file and try csv as well as excel
        try:
            csv_encoding = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            if file.name.endswith('.csv'):
                # try all the encoding types to read the csv file
                for encoding in csv_encoding:
                    try:
                        data = pd.read_csv(file, encoding=encoding)
                        # data = correct_data_types(data)
                        return data
                    except Exception as e:
                        pass
            elif file.name.endswith('.xlsx'):
                data = pd.read_excel(file)
                # data = correct_data_types(data)
                return data
            else:
                st.error('Please upload a valid file. Only CSV and Excel files are allowed.')
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
    return None

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

        # let the user select the columns to convert 
        columns = st.multiselect(':blue[**Select columns to convert**] :', data.columns)

        if columns:
            for column in columns:
                data_type = st.selectbox(f':blue[**Select data type for column : :green[{column}]**]', ['None','int', 'float', 'object', 'datetime'])
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
                        st.error(f"**Error converting column :blue[{column}] to :green[{data_type}]** :")

            if conversion_successful:
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
            
                # # Save the modified data for other analysis
                # st.session_state.modified_data = data
        else:
            st.warning('Please select columns to convert')
    else:
        st.warning('Please upload a file')

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
                                st.info(f"Could not generate word cloud for column '{col}': {str(e)}")

                    # Datetime columns
                    elif col_type == 'datetime64[ns]':
                        fig = px.histogram(data, x=col, height=350, width=500)
                        fig.update_xaxes(title_text='')
                        fig.update_yaxes(title_text='')
                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)
                        st.plotly_chart(fig)
    return col,col_type

# Function to show data overview
def show_data_overview(data):
    if data is not None:
        # display the data at the top container
        with st.expander("View Original Data", expanded=False):
            st.write(data)
        if 'report_profile' not in st.session_state:
            st.session_state.report_profile = False
        if st.button('Generate Report', use_container_width=True):
            st.session_state.report_profile = not st.session_state.report_profile

        if st.session_state.report_profile:
            with st.spinner('Generating Report for Your Data...'):
                # lets put the time.sleep(7) to show the spinner for 5 seconds
                time.sleep(0)
                st.write('')

                with st.container():
                    fix_data_type, overview_tab, alerts_tab = st.tabs(["Fix Data Types", "Overview", "Alerts"])

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
        st.warning("**Please select or upload a file.**")
        

def show_missing(data):
    if 'show_missing' not in st.session_state:
        st.session_state.show_missing = False
    if st.button('Missing Values', use_container_width=True):
        st.session_state.show_missing = not st.session_state.show_missing

    if st.session_state.show_missing:
        # show the missing values and percentage in a dataframe
        missing_values = data.isnull().sum()
        missing_percentage = missing_values / data.shape[0] * 100
        missing_info = pd.DataFrame({
            'Missing Values': missing_values.values, 
            'Percentage %': missing_percentage.values
        }, index=data.columns)
        missing_info.index.name = 'Columns'  # Set the index title
        st.dataframe(missing_info)


# def drop_columns(data):
#     if st.session_state.show_missing:
#         if 'columns_dropped' not in st.session_state:
#             st.session_state.columns_dropped = False
#         if st.button('Drop Columns', use_container_width=True):
#             st.session_state.columns_dropped = not st.session_state.columns_dropped

#         if st.session_state.columns_dropped:
#             # let the user choose which columns to drop
#             # st.write('')
#             selected_columns = st.multiselect('Select the columns you want to drop', data.columns)
#             data = data.drop(selected_columns, axis=1)
#             st.write(data)
#     return data


def handle_nulls(data):
    if st.session_state.show_missing:
        if 'handle_missing' not in st.session_state:
            st.session_state.handle_missing = False

        if st.button('Handle Missing', use_container_width=False):
            st.session_state.handle_missing = not st.session_state.handle_missing
        for i in range(1):
            st.write('') 
        

        if st.session_state.handle_missing:
            handle_missing_values = st.selectbox(':blue[**Select methods to handel null values**] :', ['None', 'Drop the column','Drop the data', 'Input missing data', 'Replace values'])

            for i in range(1):
                st.write('') 
            # st.write(f'You have selected: :red[{handle_missing_values}]')
            if handle_missing_values == 'None':
                pass

            elif handle_missing_values == 'Drop the column':
                selected_columns = st.multiselect(':blue[**Select the columns you want to drop**] :', data.columns)
                if selected_columns:
                    data = data.drop(selected_columns, axis=1)
                    st.write(data)
                    # st.write(f':blue[**Dropped columns**] : :green[{", ".join(selected_columns)}]')
                    st.write(':green[**Selected columns dropped successfully.**]')
                    # Update the dataset after dropping columns
                    data = data.drop(selected_columns, axis=1)
                # else:
                #     st.info('No columns selected for dropping.')

            elif handle_missing_values == 'Drop the data':
                null_counts = data.isnull().sum()
                columns_with_nulls = null_counts[null_counts > 0]
                data = data.dropna()
                
                # Format the columns with null counts
                formatted_columns = ', '.join([f'{col} ({count})' for col, count in columns_with_nulls.items()])
                data
                # Show the remaining length of the data frame
                st.write(f':green[**Rows with null values have been removed successfully from columns**] : :blue[{formatted_columns}].')

            elif handle_missing_values == 'Input missing data':
                # Separate the slider from the container
                num_columns = st.slider(':blue[**Number of columns**] :', 1, len(data.columns))

                for i in range(num_columns):
                    # Ensure data.columns is converted to a list before concatenation
                    columns_with_none = ["None"] + list(data.columns)

                    # Create the selectbox with "None" as the default selection
                    col_name = st.selectbox(f':blue[**Select column {i+1}**] :', columns_with_none, index=0, key=f'col_{i}')

                    # Check if a column is selected and handle missing values
                    if col_name == "None":
                        st.info("**Select proper columns to fill the values.**")
                    else:
                        if data[col_name].isnull().any():
                            user_input = st.text_input(label=f'**Enter value for column : :blue[{col_name}]**', key=f'input_{i}')
                            if user_input:
                                data[col_name] = data[col_name].fillna(user_input)
                                st.write(f':green[**Missing values have been filled successfully in column**] : :blue[{col_name}]')
                        else:
                            st.info(f'The column ":green[{col_name}]" does not have any null values.')

                # Display the updated dataset
                st.write(data)

            elif handle_missing_values == 'Replace values':
                # let user to choose which column they want to replace the values using mean, mode and median
                st.info('Only use mean and median with numerical data type.')
                selected_method = st.radio(':blue[**Select any one method**] :', ['Mean', 'Mode', 'Median'])
                num_columns = st.slider(':blue[**Number of columns**] :', 1, len(data.columns))

                for i in range(num_columns):
                    columns_with_none = ["None"] + list(data.columns)
                    col_name = st.selectbox(f':blue[**Select column {i+1}**] :', columns_with_none, key=f'col_{i}')
                    if  col_name == "None":
                        st.info("**Select proper columns to fill the values.**")

                    elif data[col_name].isnull().any():
                        try:
                            if selected_method == 'Mean':
                                if data[col_name].dtype in ['int64', 'float64']:
                                    mean = data[col_name].mean()
                                    st.write(f'**The mean for the :green[{col_name}] column is :blue[{mean}].**')
                                    data[col_name] = data[col_name].fillna(mean)
                                else:
                                    st.error(f'**Cannot calculate mean for column ":blue[{col_name}]" with data type {data[col_name].dtype}.**')
                            elif selected_method == 'Mode':
                                mode = data[col_name].mode()[0]
                                st.write(f'**The mode for the :green[{col_name}] column is :blue[{mode}].**')
                                data[col_name] = data[col_name].fillna(mode)
                            elif selected_method == 'Median':
                                if data[col_name].dtype in ['int64', 'float64']:
                                    median = data[col_name].median()
                                    st.write(f'**The median for the :green[{col_name}] column is :blue[{median}].**')
                                    data[col_name] = data[col_name].fillna(median)
                                else:
                                    st.error(f'**Cannot calculate median for column "{col_name}" with {data[col_name].dtype} data type.**')
                        except Exception as e:
                            st.error(f'**An error occurred: {e}**')
                    else:
                        st.warning(f'**No Null Values to write in "{col_name}".**')    
                st.write(data)
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
    data = handle_nulls(data)
    # data = encode_data(data)
    return data

def check_duplicates(data):
    if 'duplicated' not in st.session_state:
        st.session_state.duplicated = False

    if st.button('Check Duplicate Values', use_container_width=True):
        st.session_state.duplicated = not st.session_state.duplicated

    if st.session_state.duplicated:
        # count the total duplicated rows
        duplicated_rows = data.duplicated().sum()
        st.write(f'Total duplicated rows: :green[{duplicated_rows}]')
        # show the duplicated rows
        duplicated_rows = data[data.duplicated(keep=False)]
        duplicated_rows_sorted = duplicated_rows.sort_values(by=data.columns.tolist())
        st.write(duplicated_rows_sorted)
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

        st.write(':blue[**Select method to remove duplicates**] :')
    
    return data

def remove_duplicates_whole(data):
    
    if st.session_state.duplicated:
    # create a button to remove the duplicated rows
        if 'remove_duplicates_whole' not in st.session_state:
            st.session_state.remove_duplicates_whole = False

        if st.button('Remove Duplicates for Whole Data'):
            st.session_state.remove_duplicates_whole = not st.session_state.remove_duplicates_whole

        if st.session_state.remove_duplicates_whole:
            # check if there are duplicated rows
            if data.duplicated().any():
                # remove the duplicated rows
                data = data.drop_duplicates()
                data
                st.info(f'**Duplicates have been removed successfully. The length of the data frame is :green[{len(data)}] after removing the duplicates.**')
            else:
                st.info('**No duplicated values to remove.**')
    return data

def remove_duplicates_selected(data):
    if st.session_state.duplicated:
        if 'remove_duplicates_selected' not in st.session_state:
                st.session_state.remove_duplicates_selected = False

        if st.button('Remove Duplicates for Selected Columns'):
            st.session_state.remove_duplicates_selected = not st.session_state.remove_duplicates_selected

        if st.session_state.remove_duplicates_selected:
            # check if there are duplicated rows based on selected columns
            selected_columns = st.multiselect(':blue[**Select the columns to remove duplicates**] :', data.columns)
            if selected_columns:
                if data.duplicated(subset=selected_columns).any():
                    # remove the duplicated rows based on selected columns
                    data = data.drop_duplicates(subset=selected_columns)
                    st.write(data)
                    st.info(f'**Duplicates have been removed successfully for selected columns. The length of the data frame is :green[{len(data)}] after removing the duplicates for selected columns.**')
                else:
                    st.info('**No duplicated values to remove for selected columns.**')
            else:
                st.warning('**Please select at least one column to check for duplicates.**')
    return data


def duplicated_values(data):
    check_duplicates(data)
    data = remove_duplicates_whole(data)
    data = remove_duplicates_selected(data)
    return data

def display_and_download_cleaned_data(data):
    if data is not None:
        # Apply data cleaning steps
        data = missing_values(data)
        data = duplicated_values(data)
        
        if data is not None:
            # Convert DataFrame to CSV, then encode to UTF-8 bytes
            csv = data.to_csv(index=False).encode('utf-8')
            download= st.download_button(label='Download Cleaned CSV', data=csv, file_name='cleaned_data.csv', mime='text/csv', use_container_width=True)
            if download:
                # write a message to the user when the data is downloaded
                st.write(':green[**Downladed the cleaned data successfully.**]')
        else:
            st.warning("**Data is not available for download.**")
    else:
        st.warning("Please select or upload a file.")

def display_dataset(data):
    st.header('Dataset:')
    st.write(data)

def display_columns(data):
    if data is not None:
        st.write('Columns in the dataset:')
        st.write(data.columns.to_list())
    else:
        st.warning("No data available to display columns.")

@st.cache_data
def get_available_plot_types(x_dtype, y_dtype):
    # Check if both x and y are the same column
    if x_dtype in ['float64', 'int64'] and y_dtype in ['float64', 'int64']:
        return ['Scatter Plot', 'Line Graph', 'Heat Map', 'Bubble Chart']
    elif (x_dtype in ['float64', 'int64'] and y_dtype == object) or (x_dtype == object and y_dtype in ['float64', 'int64']):
        return ['Bar Chart (Sum)', 'Bar Chart (Average)', 'Bar Chart (Count Distinct)', 'Stacked Bar Chart', 'Box Plot', 'Pie Chart']
    elif (x_dtype == 'datetime64[ns]' and y_dtype in ['float64', 'int64']) or (x_dtype in ['float64', 'int64'] and y_dtype == 'datetime64[ns]'):
        return ['Line Chart (Sum)', 'Line Chart (Average)', 'Area Chart (Sum)', 'Area Chart (Average)']
    elif (x_dtype == 'datetime64[ns]' and y_dtype == 'object') or (x_dtype == 'object' and y_dtype == 'datetime64[ns]'):
        return ['Line Chart (Count Distinct)']
    elif x_dtype == object and y_dtype == object:
        return ['Pie Chart(Count)', 'Bar Chart', 'Stacked Bar Chart', 'Count Plot']
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
            fig = px.bar(data.groupby(y_axis)[x_axis].sum().reset_index(), x=y_axis, y=x_axis)
        else:
            fig = px.bar(data.groupby(x_axis)[y_axis].sum().reset_index(), x=x_axis, y=y_axis)
    elif plot_type == 'Bar Chart (Average)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.bar(data.groupby(y_axis)[x_axis].mean().reset_index(), x=y_axis, y=x_axis)
        else:
            fig = px.bar(data.groupby(x_axis)[y_axis].mean().reset_index(), x=x_axis, y=y_axis)
    elif plot_type == 'Bar Chart (Count Distinct)':
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.bar(data.groupby(y_axis)[x_axis].nunique().reset_index(), x=y_axis, y=x_axis)
        else:
            fig = px.bar(data.groupby(x_axis)[y_axis].nunique().reset_index(), x=x_axis, y=y_axis)
    elif plot_type == 'Stacked Bar Chart':
        # group by x_axis and y_axis and count the values
        fig = px.bar(data.groupby([x_axis, y_axis]).size().reset_index(), x=x_axis, y=0, color=y_axis)
        
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
        # if one axis is numeric, group by the other axis and sum the values and if bothe are object, group by the x_axis and count the values
        if data[x_axis].dtype in ['float64', 'int64']:
            fig = px.pie(data.groupby(y_axis)[x_axis].sum().reset_index(), values=x_axis, names=y_axis)
        else:
            fig = px.pie(data.groupby(x_axis)[y_axis].count().reset_index(), values=y_axis, names=x_axis)
    elif plot_type == 'Pie Chart(Count)':
        fig = px.pie(data.groupby(x_axis).size().reset_index(), values=0, names=x_axis)


    elif plot_type == 'Count Plot':
        fig = px.histogram(data, x=x_axis)
    elif plot_type == 'Bar Chart':
        fig = px.bar(data.groupby(x_axis)[y_axis].count().reset_index(), x=x_axis, y=y_axis)
    return fig

# Function to handle data visualization
def display_visualizations(data):
    if data is not None:
        display_dataset(data)
        # display_columns(data)

        if data is not None:
            x_axis = st.selectbox('Select the x-axis value:', [None] + data.columns.to_list())
            y_axis = st.selectbox('Select the y-axis value:', [None] + data.columns.to_list())

            if x_axis is not None and y_axis is not None:
                x_axis_dtype = data[x_axis].dtype
                y_axis_dtype = data[y_axis].dtype
                available_plot_types = get_available_plot_types(x_axis_dtype, y_axis_dtype)

                plot_type = st.selectbox('Select the type of plot:', available_plot_types)

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

def preprocess_text_data(data, feature, target_column):
    feature = data[feature].str.replace('[^a-zA-Z]', ' ')
    feature = feature.str.lower()
    stop_words = stopwords.words('english')
    feature = feature.apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    feature = feature.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))


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
        with st.expander('Orignal Data', expanded=False):
            data
        # data = data.dropna()
        st.header('Model Creation')

        problem_type = st.radio('Select the models type', ['None','Classification', 'Regression', 'Sentiment Analysis'], horizontal=True)

        st.success(f'The selected model type is: {problem_type}')

        if problem_type != 'None':
            if problem_type == 'Classification':
                models = st.radio('Select the models', ['None','Random Forest', 'Logistic Regression', 'Decision Tree'], horizontal=False)
                if models != 'None':
                    st.success(f'The selected model type is: {models}')

                    target_column = st.selectbox('Select the target column', [None] + list(data.columns))
                    features = st.multiselect('Select the features', list(data.columns))

                    if target_column is not None and features:
                        X = data[features]
                        y = data[target_column]

                        if not X.empty and not y.empty:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            _, class_train, class_test, _ = st.columns([1, 2, 2, 1])
                            with class_train:
                                st.write('Train Data')
                                st.write(X_train)

                            with class_test:
                                st.write('Test Data')
                                st.write(X_test)

                        # lets create a pipeline for the classification model
                        rf_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('classifier', RandomForestClassifier())
                        ])

                        lr_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('classifier', LogisticRegression())
                        ])

                        # svm_pipeline = Pipeline([
                        #     ('encoder', OneHotEncoder(), features),
                        #     ('classifier', SVC())
                        # ])

                        dt_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('classifier', DecisionTreeClassifier())
                        ])

                        # knn_pipeline = Pipeline([
                        #     ('encoder', OneHotEncoder(), features),
                        #     ('classifier', KNeighborsClassifier())
                        # ])

                        label_encoder = LabelEncoder()
                        y_train = label_encoder.fit_transform(y_train)
                        y_test = label_encoder.transform(y_test)

                        if models == 'Random Forest':
                            y_pred = train_classification_model(rf_pipeline, X_train, y_train, X_test)

                        elif models == 'Logistic Regression':
                            y_pred = train_classification_model(lr_pipeline, X_train, y_train, X_test)

                        # elif models == 'Support Vector Machine':
                        #     y_pred = train_classification_model(svm_pipeline, X_train, y_train, X_test)

                        elif models == 'Decision Tree':
                            y_pred = train_classification_model(dt_pipeline, X_train, y_train, X_test)

                        # elif models == 'K-Nearest Neighbors':
                        #     y_pred = train_classification_model(knn_pipeline, X_train, y_train, X_test)

                        st.write('Accuracy:', accuracy_score(y_test, y_pred))
                        st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred))
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.write('Classification Report:')
                        st.dataframe(pd.DataFrame(report).transpose())

                    else:
                        st.warning('Please select the target column.')
                else: 
                    st.warning('Please select a model.')

            elif problem_type == 'Regression':
                models = st.radio('Select the models', ['None','Random Forest', 'Linear Regression', 'Decision Tree'])
                if models != 'None':
                    target_column = st.selectbox('Select the target column', [None] + list(data.columns))
                    features = st.multiselect('Select the features', list(data.columns))
                    if target_column is not None and features:
                        X = data[features]
                        y = data[target_column]

                        if not X.empty and not y.empty:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        else:
                            st.warning('Please select the target column.')


                        reg_train, reg_test = st.columns(2)
                        with reg_train:
                            st.write('Train Data')
                            st.write(X_train)

                        with reg_test:
                            st.write('Test Data')
                            st.write(X_test)
                        
                        # Define the ColumnTransformer to apply OneHotEncoder to the specified features
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('encoder', OneHotEncoder(), features)
                            ],
                            remainder='passthrough'  # Keep the rest of the columns as they are
                        )

                        # create a pipeline for the regression model
                        rf_pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor())
                        ])

                        lr_pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', LinearRegression())
                        ])

                        # svm_pipeline = Pipeline([
                        #     ('preprocessor', preprocessor),
                        #     ('regressor', SVR())
                        # ])

                        dt_pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', DecisionTreeRegressor())
                        ])

                        # knn_pipeline = Pipeline([
                        #     ('preprocessor', preprocessor),
                        #     ('regressor', KNeighborsRegressor())
                        # ])

                        if models == 'Random Forest':
                            y_pred = train_regression_model(rf_pipeline, X_train, y_train, X_test)

                        elif models == 'Linear Regression':
                            y_pred = train_regression_model(lr_pipeline, X_train, y_train, X_test)

                        # elif models == 'Support Vector Machine':
                        #     y_pred = train_regression_model(svm_pipeline, X_train, y_train, X_test)

                        elif models == 'Decision Tree':
                            y_pred = train_regression_model(dt_pipeline, X_train, y_train, X_test)

                        # elif models == 'K-Nearest Neighbors':
                        #     y_pred = train_regression_model(knn_pipeline, X_train, y_train, X_test)

                        st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
                        st.write('R2 Score:', r2_score(y_test, y_pred))

                    else:
                        st.warning('Please select the target column.')
                else:
                    st.warning('Please select a model.')

            elif problem_type == 'Sentiment Analysis':
                models = st.radio('Select the models', ['None', 'Logistic Regression','Random Forest', 'Multinomial Naive Bayes'])
                if models != 'None':
                    target_column = st.selectbox('Select the target column', [None] + list(data.columns))
                    feature = st.selectbox('Select the features', [None] + list(data.columns))
                    if target_column != None and feature != None:
                        X, y = preprocess_text_data(data, feature, target_column)

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        nlp_train, nlp_test = st.columns(2) 
                        with nlp_train:
                            st.write('Train Data')
                            st.write(X_train)

                        with nlp_test:
                            st.write('Test Data')
                            st.write(X_test)

                        # create a pipeline for the nlp model
                        lr_pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', LogisticRegression())
                        ])

                        # svm_pipeline = Pipeline([
                        #     ('vectorizer', TfidfVectorizer()),
                        #     ('classifier', SVC())
                        # ])

                        rf_pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', RandomForestClassifier())
                        ])

                        # dt_pipeline = Pipeline([
                        #     ('vectorizer', TfidfVectorizer()),
                        #     ('classifier', DecisionTreeClassifier())
                        # ])

                        # knn_pipeline = Pipeline([
                        #     ('vectorizer', TfidfVectorizer()),
                        #     ('classifier', KNeighborsClassifier())
                        # ])
                        naive_bayes_pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', MultinomialNB())
                        ])

                        if models == 'Logistic Regression':
                            y_pred = train_nlp_model(lr_pipeline, X_train, y_train, X_test, y_test)

                        # elif models == 'Support Vector Machine':
                        #     y_pred = train_nlp_model(svm_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Random Forest':
                            y_pred = train_nlp_model(rf_pipeline, X_train, y_train, X_test, y_test)

                        # elif models == 'Decision Tree':
                        #     y_pred = train_nlp_model(dt_pipeline, X_train, y_train, X_test, y_test)

                        # elif models == 'K-Nearest Neighbors':
                        #     y_pred = train_nlp_model(knn_pipeline, X_train, y_train, X_test, y_test)
                        elif models == 'Multinomial Naive Bayes':
                            y_pred = train_nlp_model(naive_bayes_pipeline, X_train, y_train, X_test, y_test)

                        st.write('Accuracy:', accuracy_score(y_test, y_pred))
                        st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred))
                        st.write('Classification Report:')
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())

                                # save the model
                        save_model = joblib.dump(models, 'model.pkl')

                        # Read the saved model file in binary mode
                        with open('model.pkl', 'rb') as file:
                            model_data = file.read()

                        # Let the user download the model
                        if model_data is not None:
                            st.download_button(label='Download Model', data=model_data, file_name='model.pkl', mime='application/octet-stream')
                        
                    else:
                        st.warning('Please select the target column.')
                else:
                    st.warning('Please select a model.')
        else:
            st.warning('Please select something to do the analysis.')

    else:
        st.warning('Please select or upload a file.')



# Add file uploader and file selector to the sidebar
if selected_option == 'Upload a new file':
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv", "xlsx"])
    data = load_data(uploaded_file)
else:
    selected_option= None
    st.warning('Please select a file to upload.')


# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(['Data Overview', 'Data Cleaning', 'Visualizations', 'Model Building'])

with tab1:
    show_data_overview(data)

with tab2:
    st.write(':blue[**Data Cleaning**] :')
    display_and_download_cleaned_data(data)

with tab3:
    display_visualizations(data)
with tab4:
    model_building(data)
    
