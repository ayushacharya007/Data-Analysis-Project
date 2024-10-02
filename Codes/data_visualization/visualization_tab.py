import streamlit as st
from Codes.data_visualization.plot_types import get_available_plot_types
from Codes.data_visualization.create_plots import create_visualization

def display_visualizations(data) -> None:
    '''
    Display the visualization options based on the selected data.
    
    Args:
        - data: DataFrame containing the data.
    '''
    try:
        if data is None:
            st.info(":warning: Please select or upload a file.")
            return
        
        x_axis = st.selectbox(':blue[**Select the x-axis value**] :', [None] + data.columns.to_list())
        y_axis = st.selectbox(':blue[**Select the y-axis value**] :', [None] + data.columns.to_list())

        if x_axis is None or y_axis is None:
            st.info(":warning: Please select both x-axis and y-axis values for visualization.")
            return

        x_axis_dtype = data[x_axis].dtype
        y_axis_dtype = data[y_axis].dtype
        available_plot_types = get_available_plot_types(x_axis_dtype, y_axis_dtype, x_axis, y_axis)

        plot_type = st.selectbox(':blue[**Select the type of plot**] :', available_plot_types)

        if plot_type == 'None':
            st.info(":warning: Please select a valid plot type.")
            return

        fig = create_visualization(data, x_axis, y_axis, plot_type)
        if fig is not None:
            st.write(fig)
        else:
            st.info(":warning: No visualization available for the selected options.")

    except Exception as e:
        st.info(f":warning: An error occurred: {e}")