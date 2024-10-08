import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import requests

# Load the dataset
data = pd.read_csv('symbipredict_2022.csv')  # Ensure this file is in the same directory as app.py

# Preprocess the data
X = data.iloc[:, :-3]  # Use symptom columns
y = data['prognosis']   # Target disease column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Flask app setup
app = Flask(__name__)

# Home route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the list of symptoms from the form
    symptoms = request.form.getlist('symptoms')  # Get symptoms as a list
    
    # Convert symptoms to model input
    input_symptoms = [1 if symptom in symptoms else 0 for symptom in X.columns]
    
    # Predict disease
    predicted_disease = model.predict([input_symptoms])[0]
    
    # Fetch medicine name from OpenFDA API
    medicine = get_medicine_name(predicted_disease)
    
    # Get diet and precaution (replace with your logic)
    diet, precaution = get_diet_precaution(predicted_disease)
    
    # Display the results
    return f"""
    <h2>Predicted Disease: {predicted_disease}</h2>
    <p>Medicine: {medicine}</p>
    <p>Precautions: {precaution}</p>
    <p>Diet Plan: {diet}</p>
    """

# Function to get the medicine name from OpenFDA API
def get_medicine_name(disease):
    url = f'https://api.fda.gov/drug/label.json?search={disease}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data:
            medicine_name = data['results'][0].get('openfda', {}).get('brand_name', ['No medicine found'])[0]
            return medicine_name
    return "No medicine found"

# Sample function to get diet and precaution (customize based on your dataset)
def get_diet_precaution(disease):
    disease_info = data[data['prognosis'] == disease].iloc[0]
    return disease_info['diet plan'], disease_info['precaustion']

if __name__ == '__main__':
    app.run(debug=True)
