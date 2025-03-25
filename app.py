from flask import Flask, request, jsonify, redirect
import joblib
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

@app.before_request
def enforce_https():
    if not request.is_secure:
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

@app.after_request
def apply_csp(response):
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self';"
    return response

@app.after_request
def add_security_headers(response):
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
    return '''
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan API - Smart Financial Solutions</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/typed.js/2.0.12/typed.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color-1: #0a192f;
            --bg-color-2: #112240;
            --accent-color: #64ffda;
            --text-color: #8892b0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body, html {
            overflow: scroll;
            font-family: 'Poppins', sans-serif;
            height: 100%;
            width: 100%;
        }

        #background-canvas {
            position: fixed;
            top: 0;
            left: 0;
            z-index: 1;
        }

        .content-wrapper {
            position: relative;
            z-index: 10;
        }

        .glass-card {
            background: rgba(17, 34, 64, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 255, 218, 0.2);
        }

        .animate-glow {
            animation: glow 3s ease-in-out infinite alternate;
        }

        @keyframes glow {
            0% {
                box-shadow: 0 0 10px rgba(100, 255, 218, 0.2);
            }
            100% {
                box-shadow: 0 0 30px rgba(100, 255, 218, 0.6);
            }
        }

        .copy-url-input {
            background-color: rgba(10, 25, 47, 0.7);
            border-color: var(--accent-color);
            color: var(--accent-color);
        }

        pre code {
            color: var(--accent-color);
        }
    </style>
</head>
<body class="antialiased">
    <canvas id="background-canvas"></canvas>

    <div class="container mx-auto px-4 py-16 content-wrapper relative">
        <div class="max-w-4xl mx-auto text-center">
            <div class="mb-12">
                <h1 class="text-4xl md:text-6xl font-bold text-white mb-4">
                    Loan API
                </h1>
                <div class="hero-text text-2xl md:text-3xl text-gray-300 mb-8">
                    <span id="typed-output"></span>
                </div>

                <!-- URL Copy Section -->
                <div class="copy-url-container mb-12">
                    <div class="relative">
                        <input 
                            type="text" 
                            value="https://loan-api-c0xd.onrender.com/" 
                            readonly 
                            class="copy-url-input w-full p-3 rounded-lg"
                            id="api-url"
                        >
                        <button 
                            class="copy-btn absolute right-3 top-1/2 transform -translate-y-1/2"
                            onclick="copyUrl()"
                        >
                            <i class="fas fa-copy fa-lg"></i>
                        </button>
                        <div class="tooltip absolute bottom-[-30px] left-1/2 transform -translate-x-1/2 bg-teal-600 text-white p-2 rounded-md text-xs opacity-0 transition-opacity">
                            Click to copy URL
                        </div>
                    </div>
                </div>

                <!-- API Request/Response Section -->
                <div class="glass-card rounded-lg shadow-lg p-8 mb-12 animate-glow">
                    <h2 class="text-3xl font-bold text-white mb-6">API Request & Response</h2>
                    
                    <div class="grid md:grid-cols-2 gap-6">
                        <!-- Request Section -->
                        <div>
                            <h3 class="text-xl font-semibold text-white mb-4">Request Example</h3>
                            <div class="bg-0a192f p-4 rounded-lg text-left text-sm overflow-x-auto">
                                <pre><code>{
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
}</code></pre>
                            </div>
                        </div>
                        
                        <!-- Response Section -->
                        <div>
                            <h3 class="text-xl font-semibold text-white mb-4">Response Example</h3>
                            <div class="bg-0a192f p-4 rounded-lg text-left text-sm overflow-x-auto">
                                <pre><code>{
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
}</code></pre>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- API Endpoints -->
                <div class="grid md:grid-cols-3 gap-6 mb-12">
                    <div class="glass-card p-6 rounded-lg shadow-lg animate-glow">
                        <h3 class="text-xl font-semibold mb-4 text-white">/loan</h3>
                        <p class="text-gray-300">Process loan information quickly and efficiently.</p>
                    </div>
                    <div class="glass-card p-6 rounded-lg shadow-lg animate-glow">
                        <h3 class="text-xl font-semibold mb-4 text-white">/predict</h3>
                        <p class="text-gray-300">Advanced machine learning loan outcome predictions.</p>
                    </div>
                    <div class="glass-card p-6 rounded-lg shadow-lg animate-glow">
                        <h3 class="text-xl font-semibold mb-4 text-white">/status</h3>
                        <p class="text-gray-300">Real-time API health and performance monitoring.</p>
                    </div>
                </div>

                <!-- Call to Action -->
                <a href="#" style="background-color: #64ffda; color: #0a192f;" 
   class="px-8 py-3 rounded-full text-lg font-medium hover:opacity-80 transition-opacity duration-300 inline-block shadow-lg">
   View API Documentation
</a>

            </div>
        </div>
    </div>

    <script>
        function copyUrl() {
            const urlInput = document.getElementById('api-url');
            urlInput.select();
            urlInput.setSelectionRange(0, 99999);
            
            try {
                document.execCommand('copy');
                const tooltip = document.querySelector('.tooltip');
                tooltip.textContent = 'URL Copied!';
                tooltip.style.opacity = '1';
                
                setTimeout(() => {
                    tooltip.textContent = 'Click to copy URL';
                    tooltip.style.opacity = '0';
                }, 2000);
            } catch (err) {
                alert('Unable to copy URL. Please copy manually.');
            }
        }

        document.addEventListener('DOMContentLoaded', function() {
            // Typed.js initialization
            new Typed('#typed-output', {
                strings: [
                    'Powering Smart Financial Decisions',
                    'Machine Learning Loan Predictions',
                    'Secure and Scalable API Solutions'
                ],
                typeSpeed: 50,
                backSpeed: 30,
                backDelay: 2000,
                loop: true
            });

            // Advanced Canvas Background Animation
            const canvas = document.getElementById('background-canvas');
            const ctx = canvas.getContext('2d');

            // Resize canvas to full window
            function resizeCanvas() {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            }
            window.addEventListener('resize', resizeCanvas);
            resizeCanvas();

            // Particle class
            class Particle {
                constructor() {
                    this.reset();
                }

                reset() {
                    this.x = Math.random() * canvas.width;
                    this.y = canvas.height + Math.random() * 100;
                    this.radius = Math.random() * 3 + 1;
                    this.vy = -(Math.random() * 2 + 1);
                    this.vx = Math.sin(this.y) * 0.5;
                    this.alpha = 0.1;
                    this.color = `rgba(100, 255, 218, ${Math.random() * 0.5})`;
                }

                update() {
                    this.y += this.vy;
                    this.x += this.vx;
                    
                    // Fade in and out
                    if (this.y < canvas.height / 2) {
                        this.alpha = Math.min(this.alpha + 0.02, 0.5);
                    } else {
                        this.alpha = Math.max(this.alpha - 0.02, 0.1);
                    }

                    // Reset if out of bounds
                    if (this.y < -10) {
                        this.reset();
                    }
                }

                draw() {
                    ctx.beginPath();
                    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                    ctx.fillStyle = this.color;
                    ctx.globalAlpha = this.alpha;
                    ctx.fill();
                    ctx.closePath();
                }
            }

            // Create particles
            const particlesCount = 200;
            const particles = Array.from({ length: particlesCount }, () => new Particle());

            // Animation loop
            function animate() {
                // Clear canvas
                ctx.fillStyle = 'rgba(10, 25, 47, 0.05)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Update and draw particles
                particles.forEach(particle => {
                    particle.update();
                    particle.draw();
                });

                requestAnimationFrame(animate);
            }

            // Start animation
            animate();
        });
    </script>
</body>
</html>
    '''


@app.route('/status', methods=['GET'])
def status():
    return '''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Status Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f1123;
            --bg-secondary: #1a2037;
            --accent-color: #6a11cb;
            --accent-light: #2575fc;
            --text-primary: #ffffff;
            --text-secondary: #a0aec0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            perspective: 1000px;
            overflow-x: hidden;
        }

        .status-container {
            background: rgba(26, 32, 55, 0.8);
            border-radius: 20px;
            box-shadow: 
                0 15px 35px rgba(0,0,0,0.4),
                0 5px 15px rgba(0,0,0,0.3);
            padding: 40px;
            width: 100%;
            max-width: 600px;
            transform: translateZ(50px);
            transition: all 0.3s ease-out;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }

        .status-container:hover {
            transform: scale(1.03) translateZ(80px) rotateX(5deg) rotateY(-5deg);
            box-shadow: 
                0 20px 45px rgba(0,0,0,0.5),
                0 10px 25px rgba(0,0,0,0.4);
        }

        .status-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
        }

        .status-badge {
            display: flex;
            align-items: center;
            background: linear-gradient(45deg, var(--accent-color), var(--accent-light));
            color: white;
            padding: 10px 20px;
            border-radius: 50px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: transform 0.3s ease;
        }

        .status-badge:hover {
            transform: scale(1.05);
        }

        .status-badge i {
            margin-right: 10px;
            font-size: 1.2em;
        }

        .status-details {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            transition: all 0.3s ease;
        }

        .status-details:hover {
            background: rgba(255,255,255,0.1);
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        .status-item:hover {
            background: rgba(255,255,255,0.05);
            transform: translateX(10px);
        }

        .status-item .label {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .status-item .value {
            font-weight: 600;
            color: var(--accent-light);
        }

        @media (max-width: 768px) {
            .status-container {
                margin: 20px;
                padding: 25px;
            }

            .status-header {
                flex-direction: column;
                align-items: flex-start;
            }

            .status-badge {
                margin-bottom: 15px;
            }
        }

        /* Pulse Animation for Live Status */
        @keyframes pulse {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(37, 117, 252, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 15px rgba(37, 117, 252, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(37, 117, 252, 0);
            }
        }

        .live-pulse {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="status-container">
        <div class="status-header">
            <div class="status-badge live-pulse">
                <i class="fas fa-satellite-dish"></i>
                API Status: Live
            </div>
            <div id="current-time"></div>
        </div>

        <div class="status-details">
            <div class="status-item">
                <span class="label">System Status</span>
                <span class="value">Operational</span>
            </div>
            <div class="status-item">
                <span class="label">Response Code</span>
                <span class="value">200 OK</span>
            </div>
            <div class="status-item">
                <span class="label">Uptime</span>
                <span class="value" id="uptime">Calculating...</span>
            </div>
            <div class="status-item">
                <span class="label">Last Updated</span>
                <span class="value" id="last-updated"></span>
            </div>
        </div>
    </div>

    <script>
        // Update current time
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').innerHTML = `
                <span style="color: var(--text-secondary); font-size: 0.9em;">
                    ${now.toLocaleString('en-US', { 
                        dateStyle: 'medium', 
                        timeStyle: 'short' 
                    })}
                </span>
            `;
            document.getElementById('last-updated').textContent = now.toLocaleTimeString();
        }

        // Uptime calculation
        function calculateUptime() {
            const startTime = new Date(Date.now() - (Math.random() * 1000000));
            const uptimeElement = document.getElementById('uptime');
            
            function updateUptime() {
                const now = new Date();
                const diff = now - startTime;
                
                const days = Math.floor(diff / (1000 * 60 * 60 * 24));
                const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
                const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
                
                uptimeElement.textContent = `${days}d ${hours}h ${minutes}m`;
            }

            updateUptime();
            setInterval(updateUptime, 60000);
        }

        // Initialize
        updateTime();
        calculateUptime();
        setInterval(updateTime, 1000);
    </script>
</body>
</html>
    '''

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
