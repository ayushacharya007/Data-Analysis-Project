import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

def build_pipeline(numeric_features, categorical_features):
    '''
    Build a pipeline for the classification model.
    
    Args:
        - numeric_features: List of numeric feature columns.
        - categorical_features: List of categorical feature columns.
        
    Returns:
        - Pipeline object containing the model.
    '''
    try:     
        if numeric_features is not None and categorical_features is not None:
            numeric_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

        return preprocessor
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')

def train_classification_model(pipeline, X_train, y_train, X_test):
    '''
    Train the classification model.

    Args:
        - pipeline: Pipeline object containing the model.
        - X_train: DataFrame containing the training features.
        - y_train: Series containing the training target variable.
        - X_test: DataFrame containing the testing features.

    Returns:
        - y_pred: Array containing the predicted values.
    '''
    try:
        pipeline.fit(X_train, y_train)
        return pipeline.predict(X_test)
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')

def save_and_download_model(model):
    '''
    Save and download the classification model.
    
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

def classification(data):
    '''
    Build a classification model.
    '''
    try:
        models = st.radio(':blue[**Select the models**] :', ['None', 'Random Forest', 'Logistic Regression', 'K-Nearest Neighbors'], horizontal=False)
        if models != 'None':
            target_column = st.selectbox(':blue[**Select the target column**] : ', [None] + list(data.columns))
            if target_column:
                features = st.multiselect(':blue[**Select the features**] : ', [col for col in data.columns if col != target_column])
            else:
                features = []

            if target_column is not None and features:
                X, y = preprocess_data(data, target_column, features)
                if not X.empty and not y.empty:
                    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                    categorical_features = X.select_dtypes(include=['object']).columns
                    X_train, X_test, y_train, y_test = split_data(X, y)
                    display_data(X_train, X_test)

                    label_encoder = LabelEncoder()
                    y_train = label_encoder.fit_transform(y_train)
                    y_test = label_encoder.transform(y_test)

                    preprocessor = build_pipeline(numeric_features, categorical_features)

                    if models == 'Random Forest':
                        rf_model = RandomForestClassifier()
                        pipeline = Pipeline(
                            steps=[
                                ('preprocessor', preprocessor),
                                ('classifier', rf_model)
                            ]
                        )


                    elif models == 'Logistic Regression':
                        lr_model = LogisticRegression()
                        pipeline = Pipeline(
                            steps=[
                                ('preprocessor', preprocessor),
                                ('classifier', lr_model)
                            ]
                        )


                    elif models == 'K-Nearest Neighbors':
                        knn_model = KNeighborsClassifier()
                        pipeline = Pipeline(
                            steps=[
                                ('preprocessor', preprocessor),
                                ('classifier', knn_model)
                            ]
                        )

                    y_pred = train_classification_model(pipeline, X_train, y_train, X_test)

                    st.write(':blue[**Accuracy**] :', accuracy_score(y_test, y_pred))
                    st.write(':blue[**Confusion Matrix**] :', confusion_matrix(y_test, y_pred))
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.write(':blue[**Classification Report**] :')
                    st.dataframe(pd.DataFrame(report).transpose())

                    # view predictions
                    with st.expander('View Predictions'):
                        # view actual and predicted values in a dataframe
                        st.dataframe(pd.DataFrame({'Actual': label_encoder.inverse_transform(y_test), 'Predicted': label_encoder.inverse_transform(y_pred)}))
                        
                    save_and_download_model(pipeline)
                
                else:
                    st.info(':warning: The feature and target columns are empty.')

            else:
                st.info(':warning: Please select the target column and features to build the classification model.')
        else:
            st.info(':warning: Please select a model.')
    
    except Exception as e:
        st.info(f':warning: An error occurred: {e}')
