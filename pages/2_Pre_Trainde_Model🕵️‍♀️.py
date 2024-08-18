import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="Sentiment Analysis App - Pre-Trained Model 🕵️‍♂️", page_icon="🕵️‍♂️", layout="wide")

st.markdown("<h3 style='text-align: center;'>Sentiment Analysis Model 🕵️‍♀️</h3>", unsafe_allow_html=True)

for i in range(2):
    st.write("")

# @st.cache_data
def get_model():
    model_name = "siebert/sentiment-roberta-large-english"
    return pipeline("sentiment-analysis", model=model_name)

def select_input_method():
    return st.sidebar.radio(":blue[**Select input method**] :", ("None", "Text", "File"))

def analyze_text(nlp):
    user_input = st.text_input(":blue[**Enter a text to analyze its sentiment**] :")
    if user_input:
        result = nlp(user_input)
        st.write(f":green[**Text**] : {user_input}")
        st.write(f"**The sentiment of the text is:** :blue[{result[0]['label']}]")
    else:
        st.info("Please enter a text to analyze its sentiment.")

def upload_file():
    return st.sidebar.file_uploader(":blue[**Upload a file**] :", type=['csv'])


def load_dataframe(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        options = ['utf-8', 'latin1', 'ISO-8859-1']
        for option in options:
            try:
                return pd.read_csv(uploaded_file, encoding=option)
            except ValueError:
                continue
        st.error("Failed to read CSV file with provided encodings.")
    elif file_extension in ['xlsx', 'xls']:
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
    return None

@st.cache_data
def display_dataframe(df):
    with st.expander("View Dataframe", expanded=False):
        st.write(df)

def analyze_file_sentiment(nlp, df):
    if 'analyze_sentiment' not in st.session_state:
        st.session_state.analyze_sentiment = False

    if st.button("Analyze Sentiment", key='dataframe'):
        st.session_state.analyze_sentiment = not st.session_state.analyze_sentiment

    if st.session_state.analyze_sentiment:
        column_name = next((col for col in ['text', 'Text', 'review', 'Review'] if col in df.columns), None)
        if column_name is None:
            st.error("The uploaded file must have a 'text', 'Text', 'review', or 'Review' column.")
            return

        text = df[column_name].astype(str).tolist()
        result = nlp(text, truncation=True)
        df['sentiment'] = [item['label'] for item in result]

        st.write(":blue[**Sentiment Analysis Results**] :")
        df

def main():
    
    select_input = select_input_method()
    if select_input != "None":
        nlp = get_model()
    else:
        pass

    if select_input == "Text":
        analyze_text(nlp)
    elif select_input == "File":
        uploaded_file = upload_file()
        if uploaded_file is not None:
            df = load_dataframe(uploaded_file)
            if df is not None:
                display_dataframe(df)
                analyze_file_sentiment(nlp, df)
        else:
            st.info("Please upload a file to analyze its sentiment.")
    else:
        st.warning("Please select an input method to analyze sentiment.")

if __name__ == "__main__":
    main()
