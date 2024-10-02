import streamlit as st

def get_available_plot_types(x_dtype, y_dtype, x_axis, y_axis):
    """
    Determine available plot types based on the data types of x and y axes.

    Parameters:
    x_dtype (str): Data type of the x-axis column.
    y_dtype (str): Data type of the y-axis column.
    x_axis (str): Name of the x-axis column.
    y_axis (str): Name of the y-axis column.

    Returns:
    list: List of available plot types.
    """
    try:
    
        # Check if both x and y are the same column
        if x_axis == y_axis:
            return ['Count Plot']
        
        # Numeric vs Numeric
        if x_dtype in ['float64', 'int64'] and y_dtype in ['float64', 'int64']:
            return ['Scatter Plot', 'Heat Map', 'Bubble Chart']
        
        # Numeric vs Categorical
        if (x_dtype in ['float64', 'int64'] and y_dtype == 'object') or (x_dtype == 'object' and y_dtype in ['float64', 'int64']):
            return [
                'Bar Chart (Sum)', 'Bar Chart (Average)', 'Bar Chart (Count Distinct)', 
                'Stacked Bar Chart', 'Box Plot', 'Pie Chart'
            ]
        
        # DateTime vs Numeric
        if (x_dtype == 'datetime64[ns]' and y_dtype in ['float64', 'int64']) or (x_dtype in ['float64', 'int64'] and y_dtype == 'datetime64[ns]'):
            return [
                'Line Chart (Sum)', 'Line Chart (Average)', 
                'Area Chart (Sum)', 'Area Chart (Average)'
            ]
        
        # DateTime vs Categorical
        if (x_dtype == 'datetime64[ns]' and y_dtype == 'object') or (x_dtype == 'object' and y_dtype == 'datetime64[ns]'):
            return ['Line Chart (Count Distinct)']
        
        # Categorical vs Categorical
        if x_dtype == 'object' and y_dtype == 'object':
            return ['Pie Chart (Count)', 'Stacked Bar Chart']
        
        # Default case
        return ['None']

    except Exception as e:
        return st.info(f":warning: An error occurred: {e}")