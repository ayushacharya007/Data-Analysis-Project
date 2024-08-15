"""Working Model"""
import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Set the page title and favicon.
st.set_page_config(page_title="Data Analysis With GPT", page_icon=":bar_chart:", layout="wide")
st.markdown("<h4 style='text-align: center; color: black;'>Data Analysis With GPT</h4>", unsafe_allow_html=True)

# create a box to enter the API key
Api_Key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if Api_Key is None or Api_Key == "":
    st.error("Please enter your OpenAI API Key to chat with the bot.")
else:
    # Step 1: Data Input
    models = ["gpt-4o", "gpt-4o-mini-2024-07-18", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
    model = st.sidebar.selectbox("Select the model", models)

    uploaded_files = None
    with st.sidebar.title("Data Input"):
        choice = st.sidebar.radio("Choose data input method", ["Upload Data"])

    df = pd.DataFrame()
    # Step 1: Data Input
    if choice == "Upload Data":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv', 'xlsx'], accept_multiple_files=False)
        # lets check if the file is uploaded and check its extension to confirm if it is a CSV file or a xlsx file then read the file
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Please upload a CSV or Excel file.")
            # file_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]  # Get the base name without extension
            # st.markdown(f"## Chatting with {file_name}")  # Write the file name as a header
    if df.empty:
        st.info("Please upload a CSV file to chat with your data.")
    else:
        with st.expander("Dataframe"):
            st.write(df)

    # Create an agent for the dataframe
    agent = create_pandas_dataframe_agent(ChatOpenAI(model=model, temperature=0, api_key=Api_Key, streaming=True), df, verbose= True, allow_dangerous_code=True)

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
       # clear the chat history
        st.session_state.messages = []

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Create a text input for the user to enter their analysis prompt
    prompt = st.chat_input("Enter your analysis prompt")
    # Check if the prompt contains "plot" or user wants to save the chart as an image
    if prompt and ("plot" in prompt.lower() or "save" in prompt.lower()):
        st.info("This model is not capable of generating plots. Sorry for the inconvenience.")

    if prompt is not None and prompt != "":
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
