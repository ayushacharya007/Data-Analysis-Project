
from typing import List, Tuple
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def perform_linear_regression(df, target:str, features:List[str], test_size:float=0.2):
    """
    Perform linear regression on the given dataframe with the given target and features.
    
    Parameters:
    df (pd.DataFrame): The dataframe to perform linear regression on.
    target (str): The target column to predict.
    features (List[str]): The list of feature columns to use for prediction.
    test_size (float): The proportion of the dataset to include in the test split.
    
    Returns:
    Tuple[LinearRegression, float, float]: The trained model, the training score, and the testing score.
    """
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, train_score, test_score