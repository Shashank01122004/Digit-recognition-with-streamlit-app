import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']

# Create a function to train the logistic regression model
def train_model(x_train, y_train):
    clf = LogisticRegression(tol=0.1)
    clf.fit(x_train, y_train)
    return clf

# Create a function to make predictions
def make_prediction(clf, x_test):
    y_pred = clf.predict(x_test)
    return y_pred

# Create a Streamlit app
st.title("MNIST Digit Classifier")
st.write("This app classifies handwritten digits using logistic regression.")

# Add a sidebar to select the digit to classify
st.sidebar.header("Select a digit to classify")
digit_to_classify = st.sidebar.selectbox("Digit", range(10))

# Load the dataset and split it into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the logistic regression model
clf = train_model(x_train, y_train)

# Make a prediction on the selected digit
y_pred = make_prediction(clf, x_test[y_test == digit_to_classify])

# Display the prediction result
st.write(f"Prediction: {y_pred[0]}")