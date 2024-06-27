import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample Data: Replace with your actual data loading method
def load_data():
    data = {
        'size': [1500, 1600, 1700, 1800, 1900],
        'bedrooms': [3, 3, 3, 4, 4],
        'price': [300000, 320000, 340000, 360000, 380000]
    }
    return pd.DataFrame(data)

# Load Data
df = load_data()

# Train Model
X = df[['size', 'bedrooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title('House Price Prediction')

st.write("This is a simple house price prediction app. Enter the details below to get the price estimate.")

# Input Features
size = st.number_input('Size of the house (in square feet)', min_value=500, max_value=5000, value=1500)
bedrooms = st.number_input('Number of bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of bathrooms', min_value=1, max_value=10, value=3)

# Prediction
if st.button('Predict'):
    input_data = np.array([[size, bedrooms]])
    prediction = model.predict(input_data)
    st.write(f'The predicted price of the house is ${prediction[0]:,.2f}')

