import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Assuming you know the column names used during training
column_names = ['feature_1', 'feature_2', ..., 'feature_14']  # Replace with actual column names from your dataset

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get the input JSON
    data = request.json['data']
    print(f"Received data: {data}")
    
    # Convert input JSON to a DataFrame with correct column names
    input_df = pd.DataFrame([data], columns=column_names)
    
    # Handle missing columns (if any)
    missing_columns = set(column_names) - set(input_df.columns)
    for col in missing_columns:
        input_df[col] = 0  # Fill missing columns with 0 or appropriate default value
    
    # Scale the input data
    scaled_data = scaler.transform(input_df)
    
    # Predict using the regression model
    output = regmodel.predict(scaled_data)
    print(f"Prediction: {output[0]}")
    
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(x) for x in request.form.values()]
    
    # Convert to a DataFrame for scaling
    input_df = pd.DataFrame([data], columns=column_names)
    
    # Scale the input data
    scaled_data = scaler.transform(input_df)
    
    # Predict using the regression model
    output = regmodel.predict(scaled_data)[0]
    return render_template("home.html", prediction_text=f"The prediction is: {output}")

if __name__ == "__main__":
    app.run(debug=True)
