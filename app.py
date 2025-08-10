from flask import Flask, render_template, request
import joblib
import pandas as pd
import requests // ApI Feture

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Function to get medicine info using OpenFDA API
def get_medicine_info(disease):
    url = f'https://api.fda.gov/drug/label.json?search={disease}&limit=3'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        medicines = []
        if 'results' in data and len(data['results']) > 0:
            for result in data['results']:
                if 'openfda' in result and 'generic_name' in result['openfda']:
                    medicines.append(result['openfda']['generic_name'])
                if len(medicines) == 3:
                    break
            return medicines if medicines else ["No medicine found for this disease"]
        else:
            return ["No medicine found for this disease"]
    
    except Exception as e:
        print(f"Error with OpenFDA API: {e}")
        return ["Error fetching medicine information"]

@app.route('/')
def home():
    return render_template('Medicine_Finder.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = [
            'itching', 'skin_rash', 'continuous_sneezing', 'chills', 'muscle_wasting',
            'fatigue', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
            'irregular_sugar_level', 'cough', 'high_fever', 'breathlessness', 'sweating',
            'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
            'yellowing_of_eyes', 'acute_liver_failure', 'swelled_lymph_nodes', 'malaise',
            'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
            'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'dizziness',
            'obesity', 'excessive_hunger', 'extra_marital_contacts', 'muscle_weakness',
            'stiff_neck', 'swelling_joints', 'movement_stiffness', 'loss_of_smell',
            'toxic_look_(typhos)', 'depression', 'muscle_pain', 'red_spots_over_body',
            'increased_appetite', 'polyuria', 'rusty_sputum', 'lack_of_concentration',
            'visual_disturbances', 'coma', 'stomach_bleeding', 'blood_in_sputum',
            'blackheads', 'scurring'
        ]

        input_data = [1 if symptom in request.form.getlist('symptoms') else 0 for symptom in symptoms]
        input_df = pd.DataFrame([input_data], columns=symptoms)

        prediction = model.predict(input_df)[0]

        medicine_info = get_medicine_info(prediction)

        return render_template('result.html', diagnosis=prediction, medicines=medicine_info)

if __name__ == '__main__':
    app.run(debug=True)

