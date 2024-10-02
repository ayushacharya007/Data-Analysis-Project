import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

@st.cache_data
def plot_numeric_column(data, col: str) -> None:
    '''
    This function plots a bar chart for a numeric column.
    
    Args:
        - data: DataFrame containing the data.
        - col: Column name to plot.
    '''
    try:
        if data[col].nunique() <= 10 and data[col].nunique() > 3:
            plot_bar_chart(data[col], col)
        elif data[col].nunique() <= 3:
            plot_simple_bar_chart(data, col)
        else:
            plot_histogram(data, col)

    except Exception as e:
        st.info(f":warning: An error occurred while plotting numeric column '{col}': {str(e)}")


@st.cache_data
def plot_object_column(data, col: str) -> None:
    '''
    This function plots a bar chart for an object column.
    '''
    try:
        if data[col].nunique() < 10 and data[col].nunique() > 3:
            plot_bar_chart(data[col], col)
        elif data[col].nunique() <= 3:
            plot_simple_histogram(data, col)
        else:
            plot_wordcloud(data, col)

    except Exception as e:
        st.info(f":warning: An error occurred while plotting object column '{col}': {str(e)}")


@st.cache_data
def plot_datetime_column(data, col: str) -> None:
    '''
    This function plots a histogram for a datetime column.
    '''
    try:
        plot_histogram(data, col)

    except Exception as e:
        st.info(f":warning: An error occurred while plotting datetime column '{col}': {str(e)}")


@st.cache_data
def plot_bar_chart(series, col):
    '''
    This function plots a bar chart for a series.
    '''
    try:
        value_counts = series.value_counts().sort_values(ascending=False)
        top_values = value_counts[:3]
        others = pd.Series(value_counts[3:].sum(), index=['Others'])
        plot_data = pd.concat([top_values, others]).reset_index()
        plot_data.columns = [col, 'count']
        fig = px.bar(plot_data, y=col, x='count', height=350, width=500)
        fig.update_xaxes(title_text='')
        fig.update_yaxes(title_text='')
        st.plotly_chart(fig)

    except Exception as e:
        st.info(f":warning: An error occurred while plotting bar chart for column '{col}': {str(e)}")


@st.cache_data
def plot_simple_bar_chart(data, col):
    '''
    This function plots a bar chart for a column with 3 or fewer unique values.
    '''
    try:
        fig = px.bar(data, x=col, height=350, width=500)
        fig.update_xaxes(title_text='')
        fig.update_yaxes(title_text='')
        st.plotly_chart(fig)

    except Exception as e:
        st.info(f":warning: An error occurred while plotting simple bar chart for column '{col}': {str(e)}")


@st.cache_data
def plot_histogram(data, col):
    '''
    This function plots a histogram for a column.
    '''
    try:
        fig = px.histogram(data, x=col, height=350, width=500)
        fig.update_xaxes(title_text='')
        fig.update_yaxes(title_text='')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

    except Exception as e:
        st.info(f":warning: An error occurred while plotting histogram for column '{col}': {str(e)}")


@st.cache_data
def plot_simple_histogram(data, col):
    '''
    This function plots a histogram for a column with 3 or fewer unique values.
    '''
    try:
        fig = px.histogram(data, y=col, height=350, width=500)
        fig.update_xaxes(title_text='')
        fig.update_yaxes(title_text='')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

    except Exception as e:
        st.info(f":warning: An error occurred while plotting simple histogram for column '{col}': {str(e)}")


@st.cache_data
def plot_wordcloud(data, col):
    '''
    This function plots a word cloud for a column.
    '''
    try:
        wordcloud = WordCloud(width=500, height=300, background_color='white', contour_color='black', contour_width=2).generate(' '.join(data[col].dropna().astype(str)))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    except ValueError as e:
        st.info(f"Could not generate word cloud for column '{col}': {str(e)}")


@st.cache_data
def data_overview(data):
    try:
        with st.container():
            for col in data.columns:
                _, infos, _, viz, _ = st.columns([0.3, 2.5, 0.5, 2, 0.3], vertical_alignment='center')
                with st.container():
                    with infos:
                        st.markdown(f"<div class='info-box'><div class='info-column-title' style='text-align:left; margin-bottom:10px;'>{col}</div>", unsafe_allow_html=True)
                        
                        col_type = str(data[col].dtype)
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
                        # Plotting
                        if col_type in ['int64', 'float64']:
                            if data[col].nunique() < 10 and data[col].nunique() > 3:
                                plot_bar_chart(data[col], col)
                            
                            elif data[col].nunique() <= 3:
                                plot_simple_bar_chart(data, col)

                            else:
                                plot_histogram(data, col)

                        elif col_type == 'object':
                            if data[col].nunique() < 10 and data[col].nunique() > 3:
                                plot_bar_chart(data[col], col)
                            
                            elif data[col].nunique() <= 3:
                                plot_simple_histogram(data, col)

                            else:
                                plot_wordcloud(data, col)

                        elif col_type == 'datetime64[ns]':
                            plot_histogram(data, col)

                        else:
                            st.info(f"Column '{col}' has an unsupported type '{col_type}'.")

    except Exception as e:
        st.info(f":warning: An error occurred while plotting column '{col}': {str(e)}")



