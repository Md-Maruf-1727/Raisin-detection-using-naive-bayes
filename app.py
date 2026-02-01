from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder="template")

scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("knn_raisin.pkl", "rb"))

@app.route("/", )
@app.route("/predict", methods= ["POST"])
def home():
    prediction = None

    if request.method == "POST":
        area = float(request.form["Area"])
        major = float(request.form["MajorAxisLength"])
        minor = float(request.form["MinorAxisLength"])
        eccen = float(request.form["Eccentricity"])
        convex = float(request.form["ConvexArea"])
        extent = float(request.form["Extent"])
        perimeter = float(request.form["Perimeter"])

        input_data = np.array([[area, major, minor, eccen, convex, extent, perimeter]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)
        classes = {0:'Besni', 1:'Kecimen'}
        prediction = classes[pred[0]]

    return render_template("index.html", prediction=prediction)



if __name__ == "__main__":
    app.run(debug=True)