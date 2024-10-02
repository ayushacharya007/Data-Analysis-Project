import streamlit as st
import time
from Codes.data_overview.find_correct_data_type import correct_data_types
from Codes.data_overview.alerts_generator import generate_alerts
from Codes.data_overview.data_detailed_info import data_informations
from Codes.data_overview.data_exploration import data_overview

# Show data overview
def show_data_overview(data):
    '''
    Show data overview
    
    Args:
        - data: DataFrame containing the data.
    '''
    try:
        if data is None:
            st.info(":warning: Please select or upload a file.")
            return None

        if 'report_profile' not in st.session_state:
            st.session_state.report_profile = False

        if st.button('Generate Report', type='primary'):
            st.session_state.report_profile = not st.session_state.report_profile

        if st.session_state.report_profile:
            with st.spinner('Report is being generated...'):
                # time.sleep(1)  # Simulate a delay to show the spinner
                with st.container():

                    overview_tab, alerts_tab, fix_data_type_tab = st.tabs(["Overview", "Alerts", "Fix Data Types"])

                    with fix_data_type_tab:
                        with st.container():
                            correct_data_types(data)

                    with overview_tab:
                        st.write(':blue[**Data Overview**] :') 
                        with st.container():
                            data_informations(data)
                        
                        with st.container():
                            st.write('')
                            st.write(':blue[**Columns Overview**] :')
                            data_overview(data)

                    with alerts_tab:
                        st.write(':blue[**Data Alerts**] :')
                        with st.container():
                            _, alert, _ = st.columns([0.2, 3, 0.3])
                            with alert:
                                alerts = generate_alerts(data)
                                st.markdown(alerts, unsafe_allow_html=True)

                        with st.container():
                            st.write('')
                            st.write(':blue[**Columns Overview**] :')
                            data_overview(data)
    
    except Exception as e:
        st.info(f":warning: An error occurred while generating the report: {str(e)}")
                
