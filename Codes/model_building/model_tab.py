import streamlit as st
from Codes.model_building.regression import regression
from Codes.model_building.classification import classification
from Codes.model_building.sentiment_analysis import sentiment

def select_and_run_model(data):
    '''
    Select the model type and run the model.
    
    Args:
        - data: DataFrame to run the model on.
    '''
    try:
        if data is not None:
            model = st.radio(':blue[**Select the model type :**]', ['None', 'Regression', 'Classification', 'Sentiment Analysis'], horizontal=True)
            if model == 'Regression':
                regression(data)
            elif model == 'Classification':
                classification(data)
            elif model == 'Sentiment Analysis':
                sentiment(data)
            else:
                st.info(':warning: Please select a model type to proceed.')
        
        else:
            st.info(':warning: Please select or upload a file to run the model.')

    except Exception as e:
        st.info(f":warning: An error occurred: {e}")