import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="Sentiment Analysis App - Pre-Trained Model 🕵️‍♂️", page_icon="🕵️‍♂️", layout="wide")

st.markdown("<h1 style='text-align: center;'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
for i in range(2):
    st.write("")

def analyze_sentiment(uploaded_file=None, user_input=None):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    nlp = pipeline("sentiment-analysis", model=model_name)

    if user_input:
        result = nlp(user_input)
        st.write(f"**Text:** {user_input}")
        st.write(f"**The sentiment of the text is:** {result[0]['label']}.")

    if uploaded_file:
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
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return

        st.expander("View Dataframe", expanded=False).write(df)

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
            st.dataframe(df)

if __name__ == "__main__":
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    user_input = st.text_input(":blue[**Enter a text to analyze its sentiment**] :")
    analyze_sentiment(uploaded_file, user_input)
