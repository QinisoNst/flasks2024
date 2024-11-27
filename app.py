from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
with open('svc_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from POST request
    data = request.json
    
    # Expected columns
    column_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 
                    'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
                    'FATIGUE', 'ALLERGY', 'WHEEZING', 
                    'ALCOHOL CONSUMING', 'COUGHING', 
                    'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
                    'CHEST PAIN']
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data], columns=column_names)
    
    #
    #  Scale the input features
    scaled_data = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    result = "Positive" if prediction[0] == 1 else "Negative"
    
    # Return the result as JSON
    return jsonify({'prediction': result})

# Define a test route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
