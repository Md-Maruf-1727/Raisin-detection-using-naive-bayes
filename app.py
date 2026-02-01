# Raisin Classification Web App using Flask
# This app takes input features of a raisin and predicts its class (Besni or Kecimen)
# using a pre-trained KNN model with a scaler for feature normalization.

from flask import Flask, render_template, request  # Flask modules for web app
import pickle  # For loading saved ML model and scaler
import numpy as np  # For handling numerical input data

# Initialize the Flask app and set template folder
app = Flask(__name__, template_folder="template")

# Load the pre-trained scaler and KNN model from disk
scaler = pickle.load(open("scaler.pkl", "rb"))  # StandardScaler object
model = pickle.load(open("knn_raisin.pkl", "rb"))  # KNN classifier

@app.route("/", methods=["GET"])
@app.route("/predict", methods=["POST"])
def home():
    """
    Home route of the web app.
    - GET request: renders the input form.
    - POST request: processes input data, scales it, predicts the class, and returns the result.
    """
    prediction = None  # Default value for prediction

    if request.method == "POST":
        # Retrieve form data from HTML inputs and convert to float
        area = float(request.form["Area"])
        major = float(request.form["MajorAxisLength"])
        minor = float(request.form["MinorAxisLength"])
        eccen = float(request.form["Eccentricity"])
        convex = float(request.form["ConvexArea"])
        extent = float(request.form["Extent"])
        perimeter = float(request.form["Perimeter"])

        # Create a numpy array for model input
        input_data = np.array([[area, major, minor, eccen, convex, extent, perimeter]])

        # Scale the input data using the pre-fitted scaler
        input_scaled = scaler.transform(input_data)

        # Predict class using the trained KNN model
        pred = model.predict(input_scaled)

        # Map numeric prediction to human-readable class
        classes = {0: 'Besni', 1: 'Kecimen'}
        prediction = classes[pred[0]]

    # Render the HTML template with prediction result (None if GET request)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    # Run the app in debug mode (auto reloads on code change)
    app.run(debug=True)
