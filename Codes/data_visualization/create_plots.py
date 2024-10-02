import pandas as pd
import streamlit as st
import plotly.express as px


def create_bar_chart(data, x_axis, y_axis, agg_func):
    '''
    Create a bar chart based on the data and the selected columns.
    
    Args:
        - data: DataFrame containing the data.
        - x_axis: Name of the x-axis column.
        - y_axis: Name of the y-axis column.
        - agg_func: Aggregation function to apply.
        
    Returns:
        - fig: Plotly figure object.
    '''
    try:

        if data[x_axis].dtype in ['float64', 'int64']:
            grouped_data = data.groupby(y_axis)[x_axis].agg(agg_func).reset_index()
            fig = px.bar(grouped_data, x=y_axis, y=x_axis, color=y_axis, barmode='group')
        else:
            grouped_data = data.groupby(x_axis)[y_axis].agg(agg_func).reset_index()
            fig = px.bar(grouped_data, x=x_axis, y=y_axis, color=x_axis, barmode='group')
        fig.update_layout(barmode='group', xaxis={'categoryorder': 'total descending'})
        return fig

    except Exception as e:
        return st.info(f":warning: An error occurred: {e}")

def create_line_chart(data, x_axis, y_axis, agg_func):
    '''
    Create a line chart based on the data and the selected columns.
    '''
    try:
        if data[x_axis].dtype in ['float64', 'int64']:
            grouped_data = data.groupby(y_axis)[x_axis].agg(agg_func).reset_index()
            fig = px.line(grouped_data, x=y_axis, y=x_axis)
        else:
            grouped_data = data.groupby(pd.Grouper(key=x_axis, freq='D'))[y_axis].agg(agg_func).reset_index()
            fig = px.line(grouped_data, x=x_axis, y=y_axis)
        return fig
    
    except Exception as e:
        return st.info(f":warning: An error occurred: {e}")

def create_area_chart(data, x_axis, y_axis, agg_func):
    '''
    Create an area chart based on the data and the selected columns.
    '''
    try:
        if data[x_axis].dtype in ['float64', 'int64']:
            grouped_data = data.groupby(y_axis)[x_axis].agg(agg_func).reset_index()
            fig = px.area(grouped_data, x=y_axis, y=x_axis)
        else:
            grouped_data = data.groupby(pd.Grouper(key=x_axis, freq='D'))[y_axis].agg(agg_func).reset_index()
            fig = px.area(grouped_data, x=x_axis, y=y_axis)
        return fig
    
    except Exception as e:
        return st.info(f":warning: An error occurred: {e}")

def create_pie_chart(data, x_axis, y_axis, agg_func):
    '''
    Create a pie chart based on the data and the selected columns.
    '''
    try:
        if data[x_axis].dtype in ['float64', 'int64']:
            grouped_data = data.groupby(y_axis)[x_axis].agg(agg_func).reset_index()
            fig = px.pie(grouped_data, values=x_axis, names=y_axis)
        else:
            grouped_data = data.groupby(x_axis)[y_axis].count().reset_index()
            fig = px.pie(grouped_data, values=y_axis, names=x_axis)
        return fig
    
    except Exception as e:
        return st.info(f":warning: An error occurred: {e}")

def create_visualization(data, x_axis, y_axis, plot_type):
    '''
    Create a visualization based on the selected plot type.
    
    Args:
        - data: DataFrame containing the data.
        - x_axis: Name of the x-axis column.
        - y_axis: Name of the y-axis column.
        - plot_type: Type of plot to create.
        
    Returns:
        - fig: Plotly figure object.
    '''
    try:
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
            fig = create_bar_chart(data, x_axis, y_axis, 'sum')
        elif plot_type == 'Bar Chart (Average)':
            fig = create_bar_chart(data, x_axis, y_axis, 'mean')
        elif plot_type == 'Bar Chart (Count Distinct)':
            fig = create_bar_chart(data, x_axis, y_axis, 'nunique')
        elif plot_type == 'Stacked Bar Chart':
            fig = px.bar(data.groupby([x_axis, y_axis]).size().reset_index(), x=x_axis, y=0, color=y_axis)
            fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
        elif plot_type == 'Box Plot':
            fig = px.box(data, x=x_axis, y=y_axis, hover_data=[x_axis, y_axis])
        elif plot_type == 'Line Chart (Sum)':
            fig = create_line_chart(data, x_axis, y_axis, 'sum')
        elif plot_type == 'Line Chart (Average)':
            fig = create_line_chart(data, x_axis, y_axis, 'mean')
        elif plot_type == 'Area Chart (Sum)':
            fig = create_area_chart(data, x_axis, y_axis, 'sum')
        elif plot_type == 'Area Chart (Average)':
            fig = create_area_chart(data, x_axis, y_axis, 'mean')
        elif plot_type == 'Line Chart (Count Distinct)':
            fig = create_line_chart(data, x_axis, y_axis, 'nunique')
        elif plot_type == 'Pie Chart':
            fig = create_pie_chart(data, x_axis, y_axis, 'sum')
        elif plot_type == 'Pie Chart (Count)':
            fig = px.pie(data.groupby(x_axis).size().reset_index(), values=0, names=x_axis)
        elif plot_type == 'Count Plot':
            fig = px.histogram(data, x=x_axis)
            fig.update_layout(barmode='group', xaxis={'categoryorder': 'total descending'})
        elif plot_type == 'Bar Chart':
            fig = px.bar(data.groupby(x_axis)[y_axis].count().reset_index(), x=x_axis, y=y_axis, color=x_axis, barmode='group')
            fig.update_layout(barmode='group', xaxis={'categoryorder': 'total descending'})

        return fig
    
    except Exception as e:
        return st.info(f":warning: An error occurred: {e}")

