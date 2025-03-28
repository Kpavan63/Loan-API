from flask import Flask, request, jsonify, redirect
import joblib
import threading
import time
import requests
import numpy as np
from flask_cors import CORS
from flask import render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load the trained models (scaler and random forest model)
scaler = joblib.load("scaler.pkl")
model = joblib.load("random_forest_model.pkl")

# Initialize Flask app
app = Flask(__name__)


CORS(app)

@app.route('/ping')
def ping():
    return "OK", 200

def keep_awake():
    url = "https://loan-api-c0xd.onrender.com/ping"  # Update with your actual API URL
    while True:
        try:
            response = requests.get(url, timeout=5)
            print(f"Ping response: {response.status_code}")
        except Exception as e:
            print(f"Ping failed: {e}")
        time.sleep(300)  # Ping every 5 minutes

# Start self-pinging in a separate thread
threading.Thread(target=keep_awake, daemon=True).start()

@app.before_request
def enforce_https():
    if not request.is_secure:
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

@app.after_request
def apply_csp(response):
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://fonts.googleapis.com 'unsafe-inline'; "
        "font-src 'self' https://fonts.gstatic.com; "
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Initialize Flask-Limiter for rate limiting
limiter = Limiter(
    get_remote_address,  # Use the client's IP address for rate limiting
    app=app,
    default_limits=["10 per minute"],  # Default limit: 10 requests per minute
    storage_uri="memory://",  # Store rate-limit data in memory (suitable for single-instance apps)
)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/status', methods=['GET'])
def status():
    return render_template('status.html')

# Define label encoding mappings (must match those used during training)
label_mappings = {
    "Gender": {"Male": 1, "Female": 0, 1: 1, 0: 0}, 
    "Married": {"Yes": 1, "No": 0, 1: 1, 0: 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3, 0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3},  
    "Education": {"Graduate": 1, "Not Graduate": 0, 1: 1, 0: 0},
    "Self_Employed": {"Yes": 1, "No": 0, 1: 1, 0: 0}  
}

# Input validation function for numerical fields
def validate_numeric(field_value, field_name):
    try:
        value = float(field_value)
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative.")
        return value
    except ValueError as e:
        print(f"Error: {e}")
        return None

@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute")  # Limit to 5 requests per minute
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Print the received data to help debug
        print("Received data:", data)

        # Check for missing fields
        required_fields = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", 
                           "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"})

        # Extract the data
        gender = data.get("Gender")
        married = data.get("Married")
        dependents = data.get("Dependents")
        education = data.get("Education")
        self_employed = data.get("Self_Employed")
        applicant_income = validate_numeric(data.get("ApplicantIncome"), "ApplicantIncome")
        coapplicant_income = validate_numeric(data.get("CoapplicantIncome"), "CoapplicantIncome")
        loan_amount = validate_numeric(data.get("LoanAmount"), "LoanAmount")
        loan_amount_term = validate_numeric(data.get("Loan_Amount_Term"), "Loan_Amount_Term")
        credit_history = validate_numeric(data.get("Credit_History"), "Credit_History")

        # Return error if any numerical validation fails
        if None in [applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history]:
            return jsonify({"error": "Invalid numerical input values"})

        # Handle the mapping for categorical values
        gender = label_mappings["Gender"].get(gender, -1)
        married = label_mappings["Married"].get(married, -1)
        dependents = label_mappings["Dependents"].get(dependents, -1)
        education = label_mappings["Education"].get(education, -1)
        self_employed = label_mappings["Self_Employed"].get(self_employed, -1)

        # Check for invalid categorical values
        if -1 in [gender, married, dependents, education, self_employed]:
            return jsonify({"error": "Invalid categorical input values"})

        # Create input array for prediction
        user_data = np.array([[gender, married, dependents, education, self_employed,
                               applicant_income, coapplicant_income, loan_amount,
                               loan_amount_term, credit_history]])

        # Scale the input data
        scaled_data = scaler.transform(user_data)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Interpret result
        result = 'Approved' if prediction[0] == 1 else 'Rejected'

        # Return result as JSON
        return jsonify({
            "input": {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history
            },
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
