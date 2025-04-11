from sklearn.linear_model import LinearRegression
import logging

def train_model(X_train, y_train):
    """
    Trains a Linear Regression model on the training data.
    Returns the trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

