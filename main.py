from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

# Load the TensorFlow model
model = tf.keras.models.load_model("loan-eligibility-api\model\new_loan_approval_nn_model.keras")

# Define the input schema
class UserInput(BaseModel):
    person_age: int
    person_income: float
    person_emp_length: int
    loan_amnt: float
    loan_int_rate: float

# Encode input for the neural network
def encode_input(input_dict):
    return np.array([[input_dict['person_age'],
                      input_dict['person_income'],
                      input_dict['person_emp_length'],
                      input_dict['loan_amnt'],
                      input_dict['loan_int_rate']]])

# Custom logic to calculate credit score
def calculate_credit_score(input_data):
    # Example formula for demonstration purposes
    base_score = 300
    income_factor = input_data['person_income'] / 1000  # Scaled income
    emp_factor = input_data['person_emp_length'] * 10
    loan_factor = max(0, 50 - (input_data['loan_amnt'] / 10000))
    
    # Total credit score with cap between 300-850
    score = base_score + income_factor + emp_factor + loan_factor
    return min(850, max(300, score))

# Define API endpoint for prediction
@app.post("/predict/")
async def predict(user_input: UserInput):
    input_dict = user_input.dict()
    input_array = encode_input(input_dict)
    
    # Neural network prediction
    prediction = model.predict(input_array)[0][0]
    default_status = "Not Default" if prediction > 0.5 else "Default"
    
    # Calculate credit score
    credit_score = calculate_credit_score(input_dict)
    
    return {
        "default_status": default_status,
        "credit_score": credit_score
    }

# Run the server for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
