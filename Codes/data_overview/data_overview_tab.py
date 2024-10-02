import streamlit as st
import time
from Codes.data_overview.find_correct_data_type import correct_data_types
from Codes.data_overview.alerts_generator import generate_alerts
from Codes.data_overview.data_detailed_info import data_informations
from Codes.data_overview.data_exploration import data_overview

# Show data overview
def show_data_overview(data):
    if data is None:
        st.info(":warning: Please select or upload a file.")
        return None

    if 'report_profile' not in st.session_state:
        st.session_state.report_profile = False

    if st.button('Generate Report', type='primary'):
        st.session_state.report_profile = not st.session_state.report_profile

    if st.session_state.report_profile:
        with st.spinner('Report is being generated...'):
            time.sleep(1)  # Simulate a delay to show the spinner

            with st.container():

                overview_tab, alerts_tab, fix_data_type_tab = st.tabs(["Overview", "Alerts", "Fix Data Types"])

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
                    _, alert, _ = st.columns([0.2, 3, 0.3])
                    with alert:
                        alerts = generate_alerts(data)
                        st.markdown(alerts, unsafe_allow_html=True)
                    st.write('')
                    st.write(':blue[**Columns Overview**] :')
                    
                    data_overview(data)
                
                with fix_data_type_tab:
                    correct_data_types(data)
