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
'''


'''
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("model/Neural_Network_Model.keras")

# Define the input schema
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

# Function to calculate credit score
def calculate_credit_score(input_data):
    score = 300
    score += (input_data['person_income'] / 1000) * 1.5
    score += input_data['person_emp_length'] * 10
    score -= input_data['loan_int_rate'] * 5
    if input_data['loan_percent_income'] > 0.4:
        score -= 50
    if input_data['cb_person_default_on_file'] == "Y":
        score -= 100
    return max(300, min(850, score))

@app.get("/")
async def root():
    return {"message": "Loan Eligibility API is running!"}

@app.post("/predict/")
async def predict(user_input: UserInput):
    try:
        input_dict = user_input.dict()

        # One-hot encode categorical features
        home_ownership_encoded = [
            input_dict['person_home_ownership'] == "MORTGAGE",
            input_dict['person_home_ownership'] == "OTHER",
            input_dict['person_home_ownership'] == "RENT",
            input_dict['person_home_ownership'] == "OWN",
        ]

        loan_intent_encoded = [
            input_dict['loan_intent'] == "DEBTCONSOLIDATION",
            input_dict['loan_intent'] == "EDUCATION",
            input_dict['loan_intent'] == "HOMEIMPROVEMENT",
            input_dict['loan_intent'] == "MEDICAL",
            input_dict['loan_intent'] == "PERSONAL",
            input_dict['loan_intent'] == "VENTURE",
        ]

        loan_grade_encoded = [
            input_dict['loan_grade'] == "A",
            input_dict['loan_grade'] == "B",
            input_dict['loan_grade'] == "C",
            input_dict['loan_grade'] == "D",
            input_dict['loan_grade'] == "E",
            input_dict['loan_grade'] == "F",
            input_dict['loan_grade'] == "G",
        ]

        # Final input array
        input_array = np.array([[ 
            input_dict['person_age'], 
            input_dict['person_income'],
            input_dict['person_emp_length'], 
            input_dict['loan_amnt'],
            input_dict['loan_int_rate'],
            *home_ownership_encoded,  # Unpack list
            *loan_intent_encoded,     # Unpack list
            *loan_grade_encoded,      # Unpack list
            input_dict['loan_status'], 
            input_dict['loan_percent_income'],
            input_dict['cb_person_default_on_file'] == "Y",
            input_dict['cb_person_cred_hist_length']
        ]])

        # Predict loan eligibility
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
'''



'''
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app = FastAPI()

# Load trained model
model = tf.keras.models.load_model("model/Credit_Forecasting_Model.keras")

# Load preprocessing tools
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")

# Define categorical and numerical columns
numerical_columns = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
]
categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade"]

# Define the input schema
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

# Function to calculate credit score
def calculate_credit_score(input_data):
    score = 300
    score += (input_data['person_income'] / 1000) * 1.5
    score += input_data['person_emp_length'] * 10
    score -= input_data['loan_int_rate'] * 5
    if input_data['loan_percent_income'] > 0.4:
        score -= 50
    if input_data['cb_person_default_on_file'] == "Y":
        score -= 100
    return max(300, min(850, score))

@app.get("/")
async def root():
    return {"message": "Loan Eligibility API is running!"}

@app.post("/predict/")
async def predict(user_input: UserInput):
    try:
        input_dict = user_input.dict()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Convert binary feature
        input_df["cb_person_default_on_file"] = input_df["cb_person_default_on_file"].map({"N": 0, "Y": 1})

        # One-hot encode categorical features
        encoded_cats = encoder.transform(input_df[categorical_columns])

        # Scale numerical features
        scaled_nums = scaler.transform(input_df[numerical_columns])

        # Concatenate processed features
        final_input = np.hstack((scaled_nums, encoded_cats))

        # Predict loan eligibility
        prediction = model.predict(final_input)[0][0]
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

'''



from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Define focal loss function
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

@register_keras_serializable()
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss_value = alpha * K.pow((1 - p_t), gamma) * bce
        return K.mean(focal_loss_value)
    return loss

# Load trained model with focal loss
model = tf.keras.models.load_model("model/Credit_Forecasting_Model.keras", 
                                   custom_objects={"loss": focal_loss()})

# Load preprocessing tools
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("onehot_encoder.pkl")

# Define categorical and numerical columns
numerical_columns = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
]
categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade"]

# Define the input schema
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

# Function to calculate credit score
def calculate_credit_score(input_data):
    score = 300
    score += (input_data['person_income'] / 1000) * 1.5
    score += input_data['person_emp_length'] * 10
    score -= input_data['loan_int_rate'] * 5
    if input_data['loan_percent_income'] > 0.4:
        score -= 50
    if input_data['cb_person_default_on_file'] == "Y":
        score -= 100
    return max(300, min(850, score))

@app.get("/")
async def root():
    return {"message": "Loan Eligibility API is running!"}

@app.post("/predict/")
async def predict(user_input: UserInput):
    try:
        input_dict = user_input.model_dump()  # Updated for FastAPI 0.95+

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Convert binary feature safely
        input_df["cb_person_default_on_file"] = input_df["cb_person_default_on_file"].map({"N": 0, "Y": 1}).astype(int)

        # One-hot encode categorical features
        encoded_cats = encoder.transform(input_df[categorical_columns])
        #encoded_cats = encoder.transform(input_df[categorical_columns]).toarray()  # Ensure it's a NumPy array
        logging.info(f"Encoded Categorical Shape: {encoded_cats.shape}")

        # Scale numerical features
        scaled_nums = scaler.transform(input_df[numerical_columns])

        # Concatenate processed features
        #final_input = np.hstack((scaled_nums, encoded_cats))
        final_input = np.hstack((scaled_nums, input_df[["cb_person_default_on_file"]].values, encoded_cats))

        # **Logging input shapes for debugging**
        logging.info(f"✅ Input Shape: {final_input.shape}")
        logging.info(f"✅ Expected Model Input Shape: {model.input_shape}")


        # Predict loan eligibility
        prediction = model.predict(final_input)[0][0]
        loan_eligibility = "Eligible" if prediction > 0.5 else "Not Eligible"

        # Calculate credit score
        credit_score = calculate_credit_score(input_dict)

        return {
            "loan_eligibility": loan_eligibility,
            "credit_score": credit_score
        }

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return {"error": str(e)}
