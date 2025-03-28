# 🏦 Loan Prediction API 🚀

A Flask-based Loan Prediction API that uses a trained **Random Forest model** to determine loan approval based on user input. The API includes **rate limiting, security headers, and CSP policies** for enhanced protection.

## 📜 Features
✅ Machine Learning Loan Prediction (Random Forest)  
✅ Secure API with **CSP, XSS, and HTTPS enforcement**  
✅ **Rate Limiting** to prevent abuse  
✅ **Flask & Tailwind UI** for a simple frontend  
✅ **JSON-based API** for easy integration  

---

## 🛠 Tech Stack
- **Backend:** Flask, Flask-Limiter, Flask-CORS
- **Database:** PostgreSQL (Neon.tech) *(for tracking API logs & user limits)*
- **Machine Learning:** Scikit-learn, Joblib
- **Security:** CSP, HTTPS enforcement, Security Headers

---

Here's a simple Responce POST to Api:

```json
{
    "Gender": 1,
    "Married": 0,
    "Dependents": 3,
    "Education": 1,
    "Self_Employed": "Yes",
    "ApplicantIncome": 100000,
    "CoapplicantIncome": 45000,
    "LoanAmount": 500000,
    "Loan_Amount_Term": 30,
    "Credit_History": 1
}
```

Out Responce From Api:

```json

{
    "input": {
        "ApplicantIncome": 100000.0,
        "CoapplicantIncome": 45000.0,
        "Credit_History": 1.0,
        "Dependents": 3,
        "Education": 1,
        "Gender": 1,
        "LoanAmount": 500000.0,
        "Loan_Amount_Term": 30.0,
        "Married": 0,
        "Self_Employed": 1
    },
    "prediction": "Approved"
}
```
