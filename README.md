# 🏦 Loan Prediction API 🚀

A Flask-based Loan Prediction API that uses a trained **Random Forest model** to determine loan approval based on user input. The API includes **rate limiting, security headers, and CSP policies** for enhanced protection.

## 📜 Features
✅ Machine Learning Loan Prediction (Random Forest)  
✅ Secure API with **CSP, XSS, and HTTPS enforcement**  
✅ **Rate Limiting** to prevent abuse  
✅ **Flask & Tailwind UI** for a simple frontend  
✅ **JSON-based API** for easy integration  

---

## 📂 Project Structure
📦 loan-prediction-api
 ┣ 📂 templates
 ┃ ┣ 📜 index.html          # Frontend UI (Home Page)
 ┃ ┣ 📜 status.html         # API Status Page
 ┣ 📂 static
 ┃ ┣ 📂 css
 ┃ ┃ ┗ 📜 styles.css        # Styles (Tailwind or Custom)
 ┃ ┣ 📂 js
 ┃ ┃ ┗ 📜 script.js         # JavaScript (If Needed)
 ┣ 📜 app.py                # Flask API & Web Server
 ┣ 📜 scaler.pkl            # Pre-trained Scaler Model
 ┣ 📜 random_forest_model.pkl  # Trained ML Model
 ┣ 📜 requirements.txt      # Python Dependencies
 ┣ 📜 README.md             # Project Documentation
 ┣ 📜 .gitignore            # Ignore Unnecessary Files
 ┣ 📜 config.py             # Configurations (if needed)
 ┗ 📜 LICENSE               # License File

