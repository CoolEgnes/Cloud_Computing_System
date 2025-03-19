import azure.functions as func
import logging
import pickle
import json
import os
import numpy as np

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="lr_inference")
def lr_inference(req: func.HttpRequest) -> func.HttpResponse:
    # Load the model
    try:
        logging.info("Starting loading the model")
        with open('linear_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return func.HttpResponse(
             f"error: Model file not found. Please ensure 'linear_regression_model.pkl' is in the correct directory.pwd: {os.getcwd()}",
             status_code=500
        )
    
    # Parse input data from the request
    try:
        logging.info("Start reading the data")
        req_body = req.get_json()
        if "data" not in req_body:
            return func.HttpResponse(
             f"error: Error parsing input: {str(e)}",
             status_code=400
        )
        input_data = req_body.get('data')
        logging.info(f"input_data: {input_data}")
        X = np.array(input_data)
    except Exception as e:
        return func.HttpResponse(
             f"error: Error parsing input: {str(e)}",
             status_code=400
        )
    
    # Perform inference
    try:
        logging.info("Start inference")
        predictions = []
        for i in range(len(X)):
            prediction = model.predict(X[i].reshape(-1,1))
            predictions.append(prediction.item())
        return func.HttpResponse(
             f"prediction: {predictions}",
             status_code=200
        )
    except Exception as e:
        return func.HttpResponse(
             f"error: Error during inference: {str(e)}",
             status_code=500
        )
    # logging.info('Python HTTP trigger function processed a request.')
    # name = req.params.get('name')
    # if not name:
    #     try:
    #         req_body = req.get_json()
    #     except ValueError:
    #         pass
    #     else:
    #         name = req_body.get('name')

    # if name:
    #     return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    # else:
    #     return func.HttpResponse(
    #          "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
    #          status_code=200
    #     )