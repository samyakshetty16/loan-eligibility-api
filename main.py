'''from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

# Load the TensorFlow model
model = tf.keras.models.load_model("model/new_loan_approval_nn_model.keras")

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
'''

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("model/Neural_Network_Model.keras")

# Define the input schema with 12 features
class UserInput(BaseModel):
    person_age: int
    person_income: float
    person_emp_length: int
    loan_amnt: float
    loan_int_rate: float
    person_home_ownership: str
    loan_intent: str
    loan_grade: str
    loan_status: int
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# Function to calculate credit score (Example Logic)
def calculate_credit_score(input_data):
    # Assign weights to important factors
    score = 300  # Base score
    
    # Increment based on income and employment length
    score += (input_data['person_income'] / 1000) * 1.5  # Higher income, better score
    score += input_data['person_emp_length'] * 10  # Longer employment, better score

    # Decrement for high interest rate
    score -= input_data['loan_int_rate'] * 5  # Higher interest rate, lower score

    # Decrement if loan uses a high percentage of income
    if input_data['loan_percent_income'] > 0.4:
        score -= 50  # Heavy burden on income

    # Decrement for previous defaults
    if input_data['cb_person_default_on_file'] == "Y":
        score -= 100  # Major penalty for past default
    
    # Cap score between 300 and 850
    return max(300, min(850, score))

@app.get("/")
async def root():
    return {"message": "Loan Eligibility API is running!"}


# Prediction route
@app.post("/predict/")
async def predict(user_input: UserInput):
    try:
        # Convert input to dictionary
        input_dict = user_input.dict()

        # Create input array for the model
        input_array = np.array([[ 
            input_dict['person_age'], 
            input_dict['person_income'],
            input_dict['person_emp_length'], 
            input_dict['loan_amnt'],
            input_dict['loan_int_rate'],
            input_dict['person_home_ownership'] == "OWN",  # Encode as 1/0
            input_dict['loan_intent'] == "PERSONAL",       # Encode as 1/0
            input_dict['loan_grade'] == "A",               # Encode Grade A as 1
            input_dict['loan_status'], 
            input_dict['loan_percent_income'],
            input_dict['cb_person_default_on_file'] == "Y",  # Encode default flag
            input_dict['cb_person_cred_hist_length']
        ]])

        # Get model prediction
        prediction = model.predict(input_array)[0][0]
        loan_eligibility = "Eligible" if prediction > 0.5 else "Not Eligible"
        
        # Calculate credit score
        credit_score = calculate_credit_score(input_dict)

        return {
            "loan_eligibility": loan_eligibility,
            "credit_score": credit_score
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}
