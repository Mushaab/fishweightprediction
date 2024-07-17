from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [[
        data['Length1'],
        data['Length2'],
        data['Length3'],
        data['Height'],
        data['Width']
    ]]
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
