from textblob import TextBlob
import streamlit as st
import pandas as pd


st.set_page_config(page_title="Sentiment Analysis App - Pre-Trained Model 🕵️‍♂️", page_icon="🕵️‍♂️", layout="wide")

st.markdown("<h3 style='text-align: center;'>Sentiment Analysis Model 🕵️‍♀️</h3>", unsafe_allow_html=True)

for i in range(2):
    st.write("")

@st.cache_data
def get_model():
    return TextBlob


def select_input_method():
    return st.sidebar.radio(":blue[**Select input method**] :", ("None", "Text", "File"))

def analyze_text(nlp):
    user_input = st.text_input(":blue[**Enter a text to analyze its sentiment**] :")
    if user_input:
        blob = nlp(user_input)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

        st.write(f":green[**Text**] : {user_input}")

        if sentiment_label == "Positive":
            st.write(f"**The sentiment of the text is:** :blue[{sentiment_label}]")
        elif sentiment_label == "Negative":
            st.write(f"**The sentiment of the text is:** :red[{sentiment_label}]")
        else:
            st.write(f"**The sentiment of the text is:** :green[{sentiment_label}]")

        if polarity < 0:
            st.write(f"**Polarity:** :red[{polarity}]")
        elif polarity > 0:
            st.write(f"**Polarity:** :blue[{polarity}]")
        else:
            st.write(f"**Polarity:** :green[{polarity}]")

        st.write(f"**Subjectivity:** :blue[{subjectivity}]")
    else:
        st.info("Please enter a text to analyze its sentiment.")

def upload_file():
    return st.sidebar.file_uploader(":blue[**Upload a file**] :", type=['csv'])

@st.cache_data
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
    column_name = next((col for col in ['text', 'Text', 'review', 'Review'] if col in df.columns), None)

    if column_name is None:
            st.error("The uploaded file must have a 'text', 'Text', 'review', or 'Review' column.")
            return
    
    else:
        if 'analyze_sentiment' not in st.session_state:
            st.session_state.analyze_sentiment = False

        if st.button("Analyze Sentiment", key='dataframe'):
            st.session_state.analyze_sentiment = not st.session_state.analyze_sentiment

        if st.session_state.analyze_sentiment:

            df = df[[column_name]]

            text = df[column_name].astype(str).tolist()
            sentiments = []
            polarities = []
            subjectivities = []
            for t in text:
                blob = nlp(t)
                sentiments.append("Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral")
                polarities.append(blob.sentiment.polarity)
                subjectivities.append(blob.sentiment.subjectivity)
            
            df['sentiment'] = sentiments
            df['polarity'] = polarities
            df['subjectivity'] = subjectivities

            st.write(":blue[**Sentiment Analysis Results**] :")
            display_dataframe(df)


# Main function to run the app
def main():
    nlp = get_model()
    input_method = select_input_method()

    if input_method == "Text":
        analyze_text(nlp)
    elif input_method == "File":
        uploaded_file = upload_file()
        if uploaded_file:
            df = load_dataframe(uploaded_file)
            if df is not None:
                display_dataframe(df)
                analyze_file_sentiment(nlp, df)
            else:
                st.error("Failed to load the file.")
        else:
            st.info("Please upload a file to analyze its sentiment.")
    else:
        st.info("Please select an input method to analyze the sentiment of the text or file.")

if __name__ == "__main__":
    main()
