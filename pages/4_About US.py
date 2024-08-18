import streamlit as st
st.header("About Our App")
st.markdown('---')

st.markdown('''Welcome to our advanced data analysis platform, expertly crafted to streamline your data processes and enhance your insights. Whether you're looking to analyze complex datasets, clean and prepare your data, or visualize trends, our app has you covered.''')

st.markdown('''
Key features include:

- <span style="color:DodgerBlue">Data Analysis & Cleaning:</span> Efficiently handle large datasets and prepare them for deeper insights.
         
- <span style="color:DodgerBlue">Data Visualization:</span> Create compelling visual representations of your data to uncover trends and patterns.
         
- <span style="color:DodgerBlue">Machine Learning Model Creation:</span> Develop and deploy custom machine learning models tailored to your specific needs.
         
- <span style="color:DodgerBlue">Pre-Trained & Custom NLP Models:</span> Access a a robust pre-trained NLP model or train your own for specialized projects.
         
- <span style="color:DodgerBlue">Sentiment Analysis:</span> Input your CSV files or text for accurate sentiment predictions and responses using a robust pre-trained model.

- <span style="color:DodgerBlue">Chat with GPT:</span> Chat with GPT-3.5, GPT-4, or GPT-4 Turbo to generate responses, answer questions, and more.<br>:green[**Note**] : <span style="color:red"> This feature requires an OpenAI API key.</span>

         
Our app is designed to make data analysis and machine learning more accessible and effective, providing you with the tools you need to turn data into actionable insights.
''', unsafe_allow_html=True)

for i in range(4):
    st.write('')

st.markdown("**Created by Ayush Acharya**", unsafe_allow_html=True)