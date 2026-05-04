from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Safely load the model
try:
    with open("RE.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: RE.pkl not found. Predictions will not work.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_text = ""

    if request.method == "POST":
        if model is None:
            return render_template("predict.html", prediction_text="Error: Model file missing.")

        try:
            # Extract inputs from the form
            transaction_date = float(request.form["transaction_date"])
            house_age = float(request.form["house_age"])
            distance_mrt = float(request.form["distance_mrt"])
            stores = float(request.form["stores"])
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])

            # Structure for the model
            input_data = np.array([[transaction_date, house_age, distance_mrt, stores, latitude, longitude]])

            # Run prediction
            prediction = model.predict(input_data)[0]
            
            # Format the output with commas for readability (e.g., 450,000.00)
            prediction_text = f"{round(float(prediction), 2):,}"

        except ValueError:
            prediction_text = "Invalid input. Please enter numbers only."
        except Exception as e:
            prediction_text = "An error occurred during prediction."

    return render_template("predict.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")