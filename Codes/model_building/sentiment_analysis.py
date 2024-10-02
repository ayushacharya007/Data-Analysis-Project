import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


nltk.download(['stopwords', 'wordnet'])

def preprocess_text_data(data, feature_column, target_column):
    '''
    Preprocess text data by removing special characters, stopwords, and lemmatizing the words.
    
    Args:
        - data: DataFrame containing the data.
        - feature_column: Name of the feature column containing text data.
    
    Returns:
        - vectorizer: TfidfVectorizer object.
        - feature: Transformed feature data.
        - y: Series containing the target variable.
    '''
    try:
        if data[feature_column].dtype == 'object':
            data = data.dropna(subset=[feature_column, target_column])
            feature = data[feature_column].astype(str).fillna('')
            feature = feature.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
            stop_words = set(stopwords.words('english'))
            feature = feature.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
            lemmatizer = WordNetLemmatizer()
            feature = feature.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
            X = feature

            # pipeline = Pipeline([
            #     ('tfidf', TfidfVectorizer()),
            #     ('model', model)
            # ])

            if isinstance(target_column, str):
                y = data[target_column]
                if y.dtype == 'object':
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
            else:
                y = target_column
        else:
            st.info(':warning: The feature column is not of type object.')
            
        if 'X' not in locals() or 'y' not in locals() or X is None or y is None:
            st.warning("Feature and target variables could not be processed correctly.")
            return pd.Series(), pd.Series()  # Returning empty Series to prevent unpacking issues
        return X, y
    
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')
        return pd.Series(), pd.Series()  # Returning empty Series instead of None
    
# create a function to split the data into training and testing sets only feature data not target data
def split_data(feature, target):
    '''
    Split the data into training and testing sets.
    
    Args:
        - feature: DataFrame containing the features.
        - target: Series containing the target variable.
        
    Returns:
        - X_train: DataFrame containing the training features.
        - X_test: DataFrame containing the testing features.
        - y_train: Series containing the training target variable.
        - y_test: Series containing the testing target variable.
    '''
    return train_test_split(feature, target, test_size=0.3, random_state=42)

def display_data(X_train, X_test):
    '''
    Display the training and testing data.
    '''
    try:
        _, class_train, class_test, _ = st.columns([0.1, 2, 2, 0.1])
        with st.container():
            with class_train:
                st.write(':blue[**Train Data**] :')
                st.write(X_train)
            with class_test:
                st.write(':blue[**Test Data**] :')
                st.write(X_test)
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')

def train_sentiment_analysis_model(X_train, y_train, X_test, model):
    '''
    Train a sentiment analysis model using the training data.
    
    Args:
        - X_train: DataFrame containing the training features.
        - y_train: Series containing the training target variable.
        - X_test: DataFrame containing the testing features.
        - model: Model object to be trained.
        
    Returns:
        - model: Trained sentiment analysis model.
    '''
    
    try: 
        model.fit(X_train, y_train)
        return model.predict(X_test)
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')


def save_and_download_model(model):
    '''
    Save and download the sentiment(NLP) model.
    
    Args:
        - model: Pipeline object containing the model.
    '''
    try:
        joblib.dump(model, 'model.pkl')
        with open('model.pkl', 'rb') as file:
            model_data = file.read()
        if model_data:
            st.download_button(label='Download Model', data=model_data, file_name='model.pkl', mime='application/octet-stream', key='classification_model')
        else:
            st.info('No model available to download.')
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')

def sentiment(data):
    '''
    Build a sentiment analysis model.
    '''
    try:
        if data is not None:
            models = st.radio(':blue[**Select the models**] :', ['None', 'Logistic Regression','Random Forest', 'Multinomial Naive Bayes'],
            horizontal=False)
            if models != 'None':
                target_column = st.selectbox(':blue[**Select the target column**] : ', [None] + list(data.columns))
                if target_column:
                    features = st.selectbox(':blue[**Select the feature column**] : ', [None] + [col for col in data.columns if col != target_column])
                else:
                    features = []

                if features != [] and data[features].dtype == 'object':
                    X, y = preprocess_text_data(data, features, target_column)

                    if X.size > 0 and y.size > 0:
                        X_train, X_test, y_train, y_test = split_data(X, y)
                        display_data(X_train, X_test)

                        if models == 'Logistic Regression':
                            lr_model = LogisticRegression()
                            pipeline = Pipeline(
                                steps=[
                                    ('tfidf', TfidfVectorizer()),
                                    ('model', lr_model)
                                ]
                            )

                        elif models == 'Random Forest':
                            rf_model = RandomForestClassifier()
                            pipeline = Pipeline(
                                steps=[
                                    ('tfidf', TfidfVectorizer()),
                                    ('model', rf_model)
                                ]
                            )


                        elif models == 'Multinomial Naive Bayes':
                            nb_model = MultinomialNB()
                            pipeline = Pipeline(
                                steps=[
                                    ('tfidf', TfidfVectorizer()),
                                    ('model', nb_model)
                                ]
                            )

                        y_pred = train_sentiment_analysis_model(X_train, y_train, X_test, pipeline)
                        
                        st.write(':blue[**Model Score**] :', accuracy_score(y_test, y_pred))

                        with st.expander('View Prediction'):
                            st.dataframe(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
                        
                        save_and_download_model(pipeline)
                    
                    else:
                        st.info(':warning: The feature and target columns are empty.')
                
                else:
                    st.info(':warning: Select the feature and target columns.')
                
            else:
                st.info(':warning: Select the target column.')

    except Exception as e:  
        st.info(f':warning: An error occurred: {e}')
        return pd.Series(), pd.Series()  # Returning empty Series instead of None
                            
                            
                        


