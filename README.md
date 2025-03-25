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

##Example Request
POST /predict
{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 100,
    "Loan_Amount_Term": 360,
    "Credit_History": 1
}

---

##Example Response
{
    "input": { ... },
    "prediction": "Approved"
}
