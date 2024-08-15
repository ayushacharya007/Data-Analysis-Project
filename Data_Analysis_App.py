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
from modelling.linear_regression import perform_linear_regression


st.set_page_config(page_title="Write AI Data Analysis", page_icon="📊", layout="wide")

# read the style.css file
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Center the page title using markdown
st.markdown("<h1>Data Analysis APP</h1>", unsafe_allow_html=True)

# Create a sidebar radio button for file selection
selected_option = st.sidebar.radio('Select a file for analysis', ['Upload a new file'])

def correct_data_types(data):
    return data
        
@st.cache_data
def load_data(file):
    if file is not None:
        data = pd.read_csv(file)
        data = correct_data_types(data)
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
                                st.info(f"Could not generate word cloud for column '{col}': {str(e)}")

                    # Datetime columns
                    elif col_type == 'datetime64[ns]':
                        fig = px.histogram(data, x=col, height=350, width=500)
                        fig.update_xaxes(title_text='')
                        fig.update_yaxes(title_text='')
                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)
                        st.plotly_chart(fig)
    # return col,col_type

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
                st.markdown("<h3>Overview:</h3>", unsafe_allow_html=True)

                with st.container():
                    overview_tab, alerts_tab = st.tabs(["Overview", "Alerts"])
                    with overview_tab:
                        _,stat, _,type,_= st.columns([0.3,3,0.5,2,0.3])
                        with stat:
                            statistics(data)
                            
                        with type:
                            data_types(data)
                    st.write('')
                    st.write('')

                    st.markdown("<h3>Column Information:</h3>", unsafe_allow_html=True)
                    st.write('')
                    st.write('')
                                
                    with alerts_tab:
                        with st.container():
                            _, alert, _ = st.columns([0.2, 3, 0.3])
                            with alert:
                                st.write('')
                                # Generate alerts
                                st.markdown("<h5 style='color: dodgerblue;'>Alerts</h5>", unsafe_allow_html=True)
                                alerts = generate_alerts(data)
                                st.markdown(alerts, unsafe_allow_html=True)

                with st.container():
                    data_overview(data)
    else:
        st.warning("Please select or upload a file.")
        

def show_missing(data):
    if 'show_missing' not in st.session_state:
        st.session_state.show_missing = False
    if st.button('Missing Values', use_container_width=True):
        st.session_state.show_missing = not st.session_state.show_missing

    if st.session_state.show_missing:
        # show the missing values and percentage in a dataframe
        missing_values = data.isnull().sum()
        missing_percentage = missing_values / data.shape[0] * 100
        missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage %': missing_percentage})
        st.write(missing_info)


def drop_columns(data):
    if st.session_state.show_missing:
        if 'columns_dropped' not in st.session_state:
            st.session_state.columns_dropped = False
        if st.button('Drop Columns', use_container_width=True):
            st.session_state.columns_dropped = not st.session_state.columns_dropped

        if st.session_state.columns_dropped:
            # let the user choose which columns to drop
            st.write('Select the columns you want to drop')
            selected_columns = st.multiselect('Columns', data.columns)
            st.write('Selected Columns:')
            selected_columns
            data = data.drop(selected_columns, axis=1)
            st.write(data)
    return data


def handle_nulls(data):
    if st.session_state.show_missing:
        handle_missing_values = st.selectbox('Select methods to handel null values', ['None', 'Drop the data', 'Input missing data', 'Replace values'])
        st.write(f'You have selected: :red[{handle_missing_values}]')
        if handle_missing_values == 'None':
            pass

        elif handle_missing_values == 'Drop the data':
            null_counts = data.isnull().sum()
            columns_with_nulls = null_counts[null_counts > 0]
            data = data.dropna()
            st.success(f'Rows with null values have been removed successfully from columns [{columns_with_nulls}].')
            data
            # show the remaining length of the data frame
            st.info(f'The length of the data frame is {len(data)} after removing the nulls.')

        elif handle_missing_values == 'Input missing data':
            # let user to choose how many columns they want tyo select to imput the data using slider
            st.write('Select the number of columns you want to input the missing data')
            num_columns = st.slider('Number of columns', 1, len(data.columns))
            for i in range(num_columns):
                col_name = st.selectbox(f'Select column {i+1}:', data.columns, key=f'col_{i}')
                if data[col_name].isnull().any():
                    user_input = st.text_input(f'Enter value for column ({col_name}):', key=f'input_{i}')
                    data[col_name] = data[col_name].fillna(user_input)
                else:
                    st.warning(f'No Null Values to write in "{col_name}".')
            data

        elif handle_missing_values == 'Replace values':
            # let user to choose which column they want to replace the values using mean, mode and median
            st.info('Only use mean and medain with numerical data type.')
            selected_method = st.radio('Select any one method:', ['Mean', 'Mode', 'Median'])
            num_columns = st.slider('Number of columns', 1, len(data.columns))

            for i in range(num_columns):
                col_name = st.selectbox(f'Select column {i+1}:', data.columns, key=f'col_{i}')
                if data[col_name].isnull().any():
                    if selected_method == 'Mean':
                        mean = data[col_name].mean()
                        st.write(f'The mean for the :red[{col_name}] column is :blue[{mean}].')
                        data[col_name] = data[col_name].fillna(mean)
                    elif selected_method == 'Mode':
                        mode = data[col_name].mode()[0]
                        st.write(f'The mode for the :red[{col_name}] column is :blue[{mode}].')
                        data[col_name] = data[col_name].fillna(mode)
                    elif selected_method == 'Median':
                        median = data[col_name].median()
                        st.median(f'The mean for the :red[{col_name}] column is :blue[{median}].')
                        data[col_name] = data[col_name].fillna(median)
                else:
                    st.warning(f'No Null Values to write in "{col_name}".')
            data
    return data

def encode_data(data):
    if st.session_state.show_missing:
        if 'encoded_data' not in st.session_state:
            st.session_state.encoded_data = False

        if st.button('Encoding', use_container_width=True):
            st.session_state.encoded_data = not st.session_state.encoded_data

        if st.session_state.encoded_data:
            st.write('Select the columns you want to encode')
            selected_columns = st.multiselect('Columns', data.columns, key='encoding')
            st.write('Selected Columns:')
            selected_columns
            encoding_type = st.selectbox('Select encoding type', ['One-Hot Encoding', 'Label Encoding'])
            if encoding_type == 'One-Hot Encoding':
                data = pd.get_dummies(data, columns=selected_columns)
            elif encoding_type == 'Label Encoding':
                for col in selected_columns:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
            st.write(data)
    return data

def missing_values(data):
    show_missing(data)

    data = drop_columns(data)
    data = handle_nulls(data)
    data = encode_data(data)
    return data

def check_duplicates(data):
    if 'duplicated' not in st.session_state:
        st.session_state.duplicated = False

    if st.button('Check Duplicate Values'):
        st.session_state.duplicated = not st.session_state.duplicated

    if st.session_state.duplicated:
    # count the total duplicated rows
        duplicated_rows = data.duplicated().sum()
        f'Total duplicated rows: :green[{duplicated_rows}]'
    # show the duplicated rows
        duplicated_rows = data[data.duplicated()]
        st.write(duplicated_rows)
        st.write('')
    # let user to see the duplicatde data column wise
        st.write('Select the columns you want to see the duplicated values')
        selected_columns = st.multiselect('Columns', data.columns, key='duplicate_check')
        if selected_columns:  # Ensure there's at least one column selected
        # Adjusted to include keep=False to mark all duplicates as True
            duplicated_rows_count = data.duplicated(subset=selected_columns, keep=False).sum()
        # Join the selected_columns list into a string separated by commas
            columns_str = ', '.join(selected_columns)
            st.write(f"Total duplicated rows in :red[{columns_str}] column is: :green[{duplicated_rows_count}]")
        # Adjusted to include keep=False to get all duplicated rows
            duplicated_rows = data[data.duplicated(subset=selected_columns, keep=False)]
            st.write(duplicated_rows)

def remove_duplicates(data):
    if st.session_state.duplicated:
    # create a button to remove the duplicated rows
        if st.button('Remove Duplicates'):
        # check if there are duplicated rows
            if data.duplicated().any():
            # remove the duplicated rows
                data = data.drop_duplicates()
                st.success('Duplicates have been removed successfully.')
                st.write(data)
                st.info(f'The length of the data frame is {len(data)} after removing the duplicates.')
            else:
                st.info('No duplicated values to remove.')
    return data

def duplicated_values(data):
    check_duplicates(data)

    data = remove_duplicates(data)
    return data

# Modify the download_cleaned_data function to check for None
def download_cleaned_data(data):
    data= missing_values(data)
    data = duplicated_values(data)
    st.write("")
    if data is not None:
        # Convert DataFrame to CSV, then encode to UTF-8 bytes
        csv = data.to_csv(index=False).encode('utf-8')
        return csv
    else:
        # Handle the case where data is None
        print("No data available to download.")
        return None

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
        display_columns(data)

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

def train_classification_model(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_pred

def train_regression_model(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_pred

def train_nlp_model(pipe, X_train, y_train, X_test, y_test):
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
        data = data.dropna()
        st.header('Model Creation')

        problem_type = st.radio('Select the models type', ['None','Classification', 'Regression', 'Sentiment Analysis'], horizontal=True)

        st.success(f'The selected model type is: {problem_type}')

        if problem_type != 'None':
            if problem_type == 'Classification':
                models = st.radio('Select the models', ['None','Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'K-Nearest Neighbors'], horizontal=False)
                if models != 'None':
                    st.success(f'The selected model type is: {models}')

                    target_column = st.selectbox('Select the target column', [None] + list(data.columns))
                    features = st.multiselect('Select the features', list(data.columns))

                    if target_column != None and features != None:
                        X = data[features]
                        y = data[target_column]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        class_train, class_test= st.columns(2)
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

                        svm_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('classifier', SVC())
                        ])

                        dt_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('classifier', DecisionTreeClassifier())
                        ])

                        knn_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('classifier', KNeighborsClassifier())
                        ])

                        label_encoder = LabelEncoder()
                        y_train = label_encoder.fit_transform(y_train)
                        y_test = label_encoder.transform(y_test)

                        if models == 'Random Forest':
                            y_pred = train_classification_model(rf_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Logistic Regression':
                            y_pred = train_classification_model(lr_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Support Vector Machine':
                            y_pred = train_classification_model(svm_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Decision Tree':
                            y_pred = train_classification_model(dt_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'K-Nearest Neighbors':
                            y_pred = train_classification_model(knn_pipeline, X_train, y_train, X_test, y_test)

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
                models = st.radio('Select the models', ['None','Random Forest', 'Linear Regression', 'Support Vector Machine', 'Decision Tree', 'K-Nearest Neighbors'])
                if models != 'None':
                    target_column = st.selectbox('Select the target column', [None] + list(data.columns))
                    features = st.multiselect('Select the features', list(data.columns))
                    if target_column != None and features != None:
                        X = data[features]
                        y = data[target_column]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        reg_train, reg_test = st.columns(2)
                        with reg_train:
                            st.write('Train Data')
                            st.write(X_train)

                        with reg_test:
                            st.write('Test Data')
                            st.write(X_test)

                        # create a pipeline for the regression model
                        rf_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('regressor', RandomForestRegressor())
                        ])

                        lr_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('regressor', LinearRegression())
                        ])

                        svm_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('regressor', SVR())
                        ])

                        dt_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('regressor', DecisionTreeRegressor())
                        ])

                        knn_pipeline = Pipeline([
                            ('encoder', OneHotEncoder(), features),
                            ('regressor', KNeighborsRegressor())
                        ])

                        if models == 'Random Forest':
                            y_pred = train_regression_model(rf_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Linear Regression':
                            y_pred = train_regression_model(lr_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Support Vector Machine':
                            y_pred = train_regression_model(svm_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Decision Tree':
                            y_pred = train_regression_model(dt_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'K-Nearest Neighbors':
                            y_pred = train_regression_model(knn_pipeline, X_train, y_train, X_test, y_test)

                        st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
                        st.write('R2 Score:', r2_score(y_test, y_pred))

                    else:
                        st.warning('Please select the target column.')
                else:
                    st.warning('Please select a model.')

            elif problem_type == 'Sentiment Analysis':
                models = st.radio('Select the models', ['None', 'Logistic Regression', 'Support Vector Machine','Random Forest', 'Decision Tree', 'K-Nearest Neighbors', 'Multinomial Naive Bayes'])
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

                        svm_pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', SVC())
                        ])

                        rf_pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', RandomForestClassifier())
                        ])

                        dt_pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', DecisionTreeClassifier())
                        ])

                        knn_pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', KNeighborsClassifier())
                        ])
                        naive_bayes_pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer()),
                            ('classifier', MultinomialNB())
                        ])

                        if models == 'Logistic Regression':
                            y_pred = train_nlp_model(lr_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Support Vector Machine':
                            y_pred = train_nlp_model(svm_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Random Forest':
                            y_pred = train_nlp_model(rf_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'Decision Tree':
                            y_pred = train_nlp_model(dt_pipeline, X_train, y_train, X_test, y_test)

                        elif models == 'K-Nearest Neighbors':
                            y_pred = train_nlp_model(knn_pipeline, X_train, y_train, X_test, y_test)
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
    st.write('')
    show_data_overview(data)
def display_cleaning_methods(data):
    # Call the function to download data, ensuring data is not None
    if data is not None:
        st.header('Data Cleaning')
        csv_data = download_cleaned_data(data)
        if csv_data is not None:        
            st.download_button(label='Download Cleaned CSV', data=csv_data, file_name='cleaned_data.csv', mime='text/csv')
        else:
            print("Data is not available for download.")
    else:
        st.warning("Please select or upload a file.")

with tab2:
    display_cleaning_methods(data)

with tab3:
    display_visualizations(data)
with tab4:
    model_building(data)
    
