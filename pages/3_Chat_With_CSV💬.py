"""Working Model"""
import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Set the page title and favicon.
st.set_page_config(page_title="Data Analysis With GPT", page_icon=":bar_chart:", layout="wide")
st.markdown("<h3 style='text-align: center;'>Data Analysis With GPT 💬</h3>", unsafe_allow_html=True)


def get_api_key():
    return st.sidebar.text_input(":blue[**Enter your OpenAI API Key**] :", type="password")

def select_model():
    models = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    return st.sidebar.selectbox(":blue[**Select the model**] :", models)

def upload_file():
    return st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv"])

@st.cache_data
def load_dataframe(uploaded_file):
    df = pd.DataFrame()
    csv_encoding = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    for encoding in csv_encoding:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            break
        except Exception:
            pass
    return df

@st.cache_data
def display_dataframe(df):
    with st.expander("Show Dataframe"):
        st.write(df)


def create_agent(model, api_key, df):
    return create_pandas_dataframe_agent(
        ChatOpenAI(model=model, temperature=0, api_key=api_key, streaming=True, max_tokens=2000),
        df, verbose=True, allow_dangerous_code=True, max_iterations=3, agent_type="tool-calling", engine="pandas"
    )

# @st.cache_data
def display_chat_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

@st.cache_data
def clear_chat_history():
    st.session_state.messages = []

def handle_prompt(agent):
    prompt = st.chat_input("Enter your analysis prompt")
    if prompt is not None and prompt != "":
        with st.container():
            st.spinner("Generating Response...")
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate a new response if last message is not from assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.spinner("Thinking..."):
                    try:
                        responses = agent(prompt)
                    except ValueError:
                        responses = {"output": "An error occurred. LLM could not generate a response. Please try again."}
                
                    placeholder = st.empty()
                with st.chat_message("assistant"):
                    st.write(responses['output'])
                message = {"role": "assistant", "content": responses['output']}
                st.session_state.messages.append(message)

def main():
    api_key = get_api_key()
    if not api_key:
        st.warning("Please enter your OpenAI API Key to chat with the bot.")
        return

    model = select_model()
    uploaded_file = upload_file()
    if not uploaded_file:
        st.warning("Please upload a file to start the analysis.")
        return

    try:
        df = load_dataframe(uploaded_file)
        display_dataframe(df)
        agent = create_agent(model, api_key, df)
        display_chat_messages()
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
        handle_prompt(agent)
    except Exception as e:
        st.error(f"Error loading file: {e}")


if __name__ == "__main__":
    main()