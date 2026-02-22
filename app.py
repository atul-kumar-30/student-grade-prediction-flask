from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.form.to_dict()

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert values to numeric
    input_df = input_df.apply(pd.to_numeric)

    # Match training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)

    return render_template("index.html",
                           prediction_text=f"Predicted Final Grade (G3): {prediction[0]:.2f}")

if __name__ == "__main__":
    app.run(debug=True)