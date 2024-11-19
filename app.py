from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('winemodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.json
        features = np.array(data['features']).reshape(1, -1)  # Ensure the input shape matches the model
        prediction = model.predict(features)  # Get prediction
        
        # Interpret the prediction as 'Good Quality Wine' or 'Bad Quality Wine'
        result = "Good Quality Wine" if prediction[0] == 1 else "Bad Quality Wine"
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
