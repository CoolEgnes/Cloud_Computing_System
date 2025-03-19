import azure.functions as func
import logging
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json
import time

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="linear_regression")
def linear_regression(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # Generate synthetic data for demonstration
    np.random.seed(42)
    X = 2 * np.random.rand(1000, 1)  # 1000 samples, 1 feature
    y = 4 + 3 * X + np.random.randn(1000, 1)  # y = 4 + 3x + noise

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Start the timer
    start_time = time.time()
    formatted_start_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

    # Train the model
    model.fit(X_train, y_train)

    # Stop the timer
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)

    # Prepare the response
    result = {
        "start_time": formatted_start_time,
        "model_coefficients": model.coef_.tolist(),
        "model_intercept": model.intercept_.tolist(),
        "mean_squared_error": mse,
        "training_time": training_time
    }

    # Return the response as JSON
    return func.HttpResponse(json.dumps(result), mimetype="application/json")