'''from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize API
app = FastAPI()

# Load models
classifier = joblib.load("model/tuned_xgboost_classifier.pkl")
regressor = joblib.load("model/tuned_xgboost_regressor.pkl")

# Load preprocessing utilities
scaler = joblib.load("loan_data_scaler.pkl")
encoder = joblib.load("loan_data_onehot_encoder.pkl")

# Define categorical & numerical columns
categorical_columns = ["loan_purpose"]
numerical_columns = [
    "interest_rate",
    "monthly_installment",
    "log_annual_income",
    "debt_to_income",
    "credit_line_days",
    "revolving_balance",
    "revolving_utilization",
    "inquiries_last_6mths",
    "past_due_2yrs",
    "public_records"
]

def preprocess_input(data: dict):
    """Convert raw input to model-ready format."""
    df = pd.DataFrame([data])

    # One-hot encoding for categorical column
    encoded_df = pd.DataFrame(
        encoder.transform(df[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )

    # Scale numerical columns
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Drop original categorical column and combine encoded data
    df = df.drop(columns=categorical_columns).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

    return df.values  # Convert to NumPy array for prediction

@app.post("/predict")
def predict_credit_eligibility(data: dict):
    """API Endpoint for Credit Eligibility Prediction"""
    
    # Preprocess the input
    processed_data = preprocess_input(data)

    # Get predictions
    credit_eligibility = classifier.predict(processed_data)[0]  # 0 (Not Eligible) or 1 (Eligible)
    credit_score = regressor.predict(processed_data)[0]  # Predicted Credit Score

    return {
        "credit_score": round(credit_score, 2),
        "credit_eligibility": int(credit_eligibility)
    }
'''




from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load the models
classifier = joblib.load("model\tuned_xgboost_classifier.pkl")
regressor = joblib.load("model/tuned_xgboost_regressor.pkl")

# Load preprocessing tools
scaler = joblib.load("loan_data_scaler.pkl")
encoder = joblib.load("loan_data_onehot_encoder.pkl")

# Define categorical & numerical columns
categorical_columns = ['loan_purpose']
numerical_columns = ['interest_rate', 'monthly_installment', 'log_annual_income', 'debt_to_income', 
                     'credit_line_days', 'revolving_balance', 'revolving_utilization', 
                     'inquiries_last_6mths', 'past_due_2yrs', 'public_records']

def preprocess_input(input_data):
    """
    Converts user input into a model-ready NumPy array.
    """
    import pandas as pd

    df = pd.DataFrame([input_data])

    # Encode categorical variables
    encoded_df = pd.DataFrame(encoder.transform(df[categorical_columns]), 
                              columns=encoder.get_feature_names_out(categorical_columns))

    # Scale numerical features
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Drop original categorical column & concatenate encoded data
    df = df.drop(columns=categorical_columns).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

    return df.values  # Return as NumPy array for model input

@app.post("/predict")
def predict_credit_eligibility(data: dict):
    """
    API Endpoint for Credit Eligibility & Credit Score Prediction.
    """

    # Preprocess input
    processed_data = preprocess_input(data)

    # Get predictions
    credit_eligibility = classifier.predict(processed_data)[0]  # 0 (Not Eligible) or 1 (Eligible)
    credit_score = regressor.predict(processed_data)[0]  # Predicted Credit Score

    # Ensure JSON serializable types
    return {
        "credit_score": round(float(credit_score), 2),  # Convert NumPy float to Python float
        "credit_eligibility": int(credit_eligibility)   # Convert NumPy int to Python int
    }
