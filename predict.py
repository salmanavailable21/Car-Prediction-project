"""
Load the trained model and predict the selling price for a new (used) car.

Example:
    python src/predict.py
"""

import os

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "model")


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "car_price_model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, encoders, feature_names


def predict_price(car: dict) -> float:
    """
    car: dict with keys
        Present_Price (float, lakhs), Kms_Driven (int), Fuel_Type (str),
        Seller_Type (str), Transmission (str), Owner (int), Car_Age (int)
    """
    model, encoders, feature_names = load_artifacts()

    row = {}
    for col in feature_names:
        if col in encoders:  # categorical -> encode using saved LabelEncoder
            row[col] = encoders[col].transform([car[col]])[0]
        else:
            row[col] = car[col]

    X_new = pd.DataFrame([row])[feature_names]
    price = model.predict(X_new)[0]
    return round(float(price), 2)


if __name__ == "__main__":
    sample_car = {
        "Present_Price": 9.5,   # current showroom price, in lakhs INR
        "Kms_Driven": 35000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
        "Car_Age": 5,
    }

    predicted = predict_price(sample_car)
    print(f"Sample car input: {sample_car}")
    print(f"Predicted selling price: {predicted} lakhs INR")
