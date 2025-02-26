from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load pre-trained model
with open('deployed_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
