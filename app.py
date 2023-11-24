# Necessary imports
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title for the app
st.title("BYOM - Build Your Own Model")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file into a dataframe
    data = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write(data.head())

    # Select target variable
    target = st.selectbox("Select the target variable", data.columns)

    # Allow the user to select the test-train split ratio
    split_ratio = st.slider("Select test-train split ratio", 0.1, 0.9, 0.2)

    # Splitting data into train and test sets
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    st.write(f"Data split: {int((1-split_ratio)*100)}% train, {int(split_ratio*100)}% test")

    # Model selection
    model_option = st.selectbox("Select a model", ["Linear Regression", "Decision Tree"])

    # Model training
    if st.button("Train Model"):
        st.write(f"Training {model_option}...")
        if model_option == "Linear Regression":
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor()

        model.fit(X_train, y_train)
        st.write(f"{model_option} trained successfully!")

        # Model evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("Model Evaluation:")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R^2 Score: {r2}")
