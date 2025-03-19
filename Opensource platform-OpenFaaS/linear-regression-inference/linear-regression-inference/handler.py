import pickle
import json
import os
import numpy as np

def handle(req):
    # Load the model
    try:
        with open('/home/app/functions/linear-regression-inference/linear_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return {"error": f"Model file not found. Please ensure 'linear_regression_model.pkl' is in the correct directory.pwd: {os.getcwd()}"}, 500

    # Parse input data from the request
    try:
        # input_data = json.loads(req)
        # X = [[input_data['feature']]]  # Expects a single feature value
        input_data = json.loads(req)
        X = np.array(input_data).reshape(-1, 1)
    except Exception as e:
        return {"error": f"Error parsing input: {str(e)}"}, 400

    # Perform inference
    try:
        predictions = []
        for i in range(len(X)):
            prediction = model.predict(X[i].reshape(1,-1))
            predictions.append(prediction.item())
        return {"prediction": predictions}
    except Exception as e:
        return {"error": f"Error during inference: {str(e)}"}, 500
