from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model (1).pkl")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([data['features']]) 
        prediction = model.predict(features) 
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
