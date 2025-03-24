from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained models (scaler and random forest model)
scaler = joblib.load("scaler.pkl")
model = joblib.load("random_forest_model.pkl")

# Initialize Flask app
app = Flask(__name__)

from flask import render_template

@app.route('/')
def home():
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Loan API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                }
                .content {
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }
                a {
                    color: #007BFF;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to the Loan API!</h1>
            <div class="content">
                <h2>API Documentation</h2>
                <p>This API allows you to process loan-related data. Below are the available endpoints:</p>
                <ul>
                    <li><strong>/loan</strong> - Endpoint to process loan information.</li>
                    <li><strong>/predict</strong> - Predict loan outcomes based on input features.</li>
                    <li><strong>/status</strong> - Check the status of the API.</li>
                </ul>
                <p>To use the API, make sure to send your requests in the correct format and specify the necessary parameters.</p>
                <p>For more details, refer to the <a href="/docs">API Documentation</a>.</p>
            </div>
        </body>
        </html>
    '''


@app.route('/status', methods=['GET'])
def status():
    return {
        "status": "API is live and running.",
        "code": 200
    }

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
