"""Working Model"""
import streamlit as st
import pandas as pd
import os
import re
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

# Set the page title and favicon.
st.set_page_config(page_title="Data Analysis With GPT", page_icon=":bar_chart:", layout="wide")
st.markdown("<h4 style='text-align: center;'>Data Analysis With GPT</h4>", unsafe_allow_html=True)

# create a box to enter the API key
Api_Key = st.sidebar.text_input(":blue[**Enter your OpenAI API Key**] :", type="password")

if Api_Key is None or Api_Key == "":
    st.warning("Please enter your OpenAI API Key to chat with the bot.")
else:
    # Step 1: Data Input
    models = ["gpt-4-turbo","gpt-4","gpt-3.5-turbo"]
    model = st.sidebar.selectbox(":blue[**Select the model**] :", models)

    df = pd.DataFrame()
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv"])
    if uploaded_file is not None:
        try:
            csv_encoding = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            for encoding in csv_encoding:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except Exception as e:
                    pass

            with st.expander("Show Dataframe"):
                st.write(df)

            # Create an agent for the dataframe
            agent = create_pandas_dataframe_agent(ChatOpenAI(model=model, temperature=0, api_key=Api_Key, streaming=True, max_tokens=2000), df, verbose= True, allow_dangerous_code=True, max_iterations=3, agent_type="tool-calling", engine="pandas")

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

        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.warning("Please upload a file to start the analysis.")