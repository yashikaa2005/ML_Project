from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# ✅ Home route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Extract features in the same order as training
        features = [
            data['funding_rounds'],
            data['milestones'],
            data['relationships'],
            data['is_top500'],
            data['funding_total_usd'],
            data['has_roundB'],
            data['avg_participants']
        ]

        # Convert to 2D array for prediction
        prediction = model.predict([features])[0]
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)