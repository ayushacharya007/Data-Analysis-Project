import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


def preprocess_data(data, target_column, features):
    '''
    Preprocess the data by handling NaN values in the target column and features.
    
    Args:
        - data: DataFrame containing the data.
        - target_column: Name of the target column.
        - features: List of feature columns.
        
    Returns:
        - X: DataFrame containing the features.
        - y: Series containing the target variable.
    '''
    try:
        X = data[features]
        y = data[target_column]

        if y.isnull().any():
            st.info(':warning: The target column contains NaN values. Rows with NaN values are dropped.')
            data = data.dropna(subset=[target_column])
            X = data[features]
            y = data[target_column]

        return X, y
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')

def split_data(X, y):
    '''
    Split the data into training and testing sets.
    
    Args:
        - X: DataFrame containing the features.
        - y: Series containing the target variable.
        
    Returns:
        - X_train: DataFrame containing the training features.
        - X_test: DataFrame containing the testing features.
        - y_train: Series containing the training target variable.
        - y_test: Series containing the testing target variable.
    '''
    return train_test_split(X, y, test_size=0.2, random_state=42)

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

def build_pipeline(numeric_features, categorical_features, model):
    '''
    Build a pipeline for the regression model.
    
    Args:
        - numeric_features: List of numeric feature columns.
        - categorical_features: List of categorical feature columns.
        
    Returns:
        - Pipeline object containing the model.
    '''
    try:     
        if numeric_features is not None and categorical_features is not None:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OrdinalEncoder())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            return pipeline
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')

def train_regression_model(pipeline, X_train, y_train, X_test, y_test):
    '''
    Train the regression model.

    Args:
        - pipeline: Pipeline object containing the model.
        - X_train: DataFrame containing the training features.
        - y_train: Series containing the training target variable.
        - X_test: DataFrame containing the testing features.
        - y_test: Series containing the testing target variable.

    Returns:
        - y_pred: Array containing the predicted values.
    '''
    try:
        if pipeline is not None:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            return y_pred
        else:
            st.info(':warning: The pipeline is None.')
            return None
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')
        return None

def save_and_download_model(model):
    '''
    Save and download the regression model.
    
    Args:
        - model: Pipeline object containing the model.
    '''
    try:
        joblib.dump(model, 'model.pkl')
        with open('model.pkl', 'rb') as file:
            model_data = file.read()
        if model_data:
            st.download_button(label='Download Model', data=model_data, file_name='model.pkl', mime='application/octet-stream', key='regression_model')
        else:
            st.info(':warning: No model available to download.')
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')

def regression(data):
    '''
    Build a regression model.
    '''
    try:
        models = st.radio(':blue[**Select the models**] :', ['None','Random Forest', 'Linear Regression', 'SVR'])
        if models != 'None':
            target_column = st.selectbox(':blue[**Select the target column**] : ', [None] + list(data.columns))
            if target_column:
                features = st.multiselect(':blue[**Select the features**] : ', [col for col in data.columns if col != target_column])
            else:
                features = []

            if target_column is not None and features:
                X, y = preprocess_data(data, target_column, features)
                if not X.empty and not y.empty:
                    numeric_features = X.select_dtypes(include='number').columns
                    categorical_features = X.select_dtypes(include='object').columns
                    X_train, X_test, y_train, y_test = split_data(X, y)
                    display_data(X_train, X_test)    

                    if models == 'Random Forest':
                        model = RandomForestRegressor()
                    
                    elif models == 'Linear Regression':
                        model = LinearRegression()

                    elif models == 'SVR':
                        model = SVR()

                    pipeline = build_pipeline(numeric_features, categorical_features, model)
                    y_pred = train_regression_model(pipeline, X_train, y_train, X_test, y_test)

                    st.write(':blue[**R2 Score**] :', r2_score(y_test, y_pred))
                    st.write(':blue[**Mean Squared Error**] :', mean_squared_error(y_test, y_pred))
                    st.write(':blue[**Mean Absolute Error**] :', mean_absolute_error(y_test, y_pred))

                    with st.expander('View Prediction'):
                        # show the actual and predicted values in a dataframe
                        st.write('**Actual vs Predicted**')
                        st.dataframe(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

                    save_and_download_model(pipeline)
                
                else:
                    st.info(':warning: The feature and target columns are empty.')
            
            else:
                st.info(':warning: Please select the target column and features to build the regression model.')
        
        else:
            st.info(':warning: Please select a model to build the regression model.')

    except Exception as e:
        st.info(f':warning: An error occurred: {e}')