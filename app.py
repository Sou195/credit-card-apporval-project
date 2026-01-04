from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/details")
def details():
    return render_template("details.html")

@app.route("/form")
def form_page():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        income = float(request.form["income"])
        credit = float(request.form["credit_score"])
        loan = float(request.form["loan_amount"])

        values = [age, income, credit, loan]
        scaled = scaler.transform([values])

        result = model.predict(scaled)[0]
        text = "APPROVED" if result == 1 else "REJECTED"

        # GRAPH
        labels = ['Age', 'Income', 'CreditScore', 'LoanAmount']
        plt.figure(figsize=(5,4))
        plt.plot(labels, values, marker='o')
        plt.title("User Data Graph")
        plt.xlabel("Feature")
        plt.ylabel("Value")
        plt.savefig("static/graph.png")
        plt.close()

        return render_template("result.html",
                               age=age,
                               income=income,
                               credit=credit,
                               loan=loan,
                               result=text)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
