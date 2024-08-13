import streamlit as st
from transformers import pipeline
import pandas as pd
import streamlit as st
from transformers import pipeline
import pandas as pd
import tensorflow as tf

st.set_page_config(page_title="Sentiment Analysis App - Pre-Trained Model 🕵️‍♂️", page_icon="🕵️‍♂️", layout="wide")

st.markdown("<h1 style='text-align: center;'>Sentiment Analysis App - Pre-Trained Model 🕵️‍♂️</h1>", unsafe_allow_html=True)
st.markdown('---')

tf.compat.v1.reset_default_graph()


def analyze_sentiment(uploaded_file=None, user_input=None):
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    # Create a sentiment analysis pipeline using a pre-trained model
    nlp = pipeline("sentiment-analysis", model=model_name)

    if user_input is not None:
        # Analyze the sentiment of user input
        result = nlp(user_input)
        for i in range(2):
            st.write("")
        st.write(f"**Text:** {user_input}")
        for item in result:
            st.write(f"**The sentiment of the text is:** {item['label']}.")

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            options = ['utf-8', 'latin1', 'ISO-8859-1']
            for option in options:
                try:
                    df = pd.read_csv(uploaded_file, encoding=option)
                    break
                except ValueError:
                    continue
            else:
                st.error("Failed to read CSV file with provided encodings.")
                return
        elif file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_extension == 'xls':
            df = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            st.error("Unsupported file format.")
            return

        st.expander("View Dataframe", expanded=False).write(df)

        if st.button("Analyze Sentiment", key='dataframe'):
            if not any(col in df.columns for col in ['text', 'Text', 'review', 'Review']):
                st.error("The uploaded file must have a 'text', 'Text', 'review', or 'Review' column.")
                return
            column_name = next((col for col in ['text', 'Text', 'review', 'Review'] if col in df.columns), None)
            if column_name is None:
                st.error("The uploaded file does not have a 'text', 'Text', 'review', or 'Review' column.")
                return
            result = nlp(df[column_name].tolist(), truncation=True)
            df['sentiment'] = [item['label'] for item in result]
            st.dataframe(df)

if __name__ == "__main__":
    user_input = st.text_input("Enter text to analyze:", "")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'])
    analyze_sentiment(uploaded_file, user_input)
