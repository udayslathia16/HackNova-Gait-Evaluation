from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Define the input data model
class GaitConditionInput(BaseModel):
    Age: float
    Gender: str
    StrideLength: float
    StrideDuration: float

# Load the trained XGBoost model from the pickle file
xg_model_final = pickle.load(open("tddmodel.pkl", "rb"))

# Load the label encoder used during training
encoder = pickle.load(open("encoder.pickle", "rb"))

# Define CORS settings
origins = [
    "http://127.0.0.1:5500"  # Update the origin URL with the correct port and path
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Define the prediction route
@app.post("/predict_gait_condition")
async def predict_gait_condition_api(data: GaitConditionInput):
    # Convert sex to numerical value (0 for 'Female', 1 for 'Male')
    sex_numeric = 0 if data.Gender.lower() == 'female' else 1

    # Create a numpy array with the user input
    input_data = np.array([[data.Age, sex_numeric, data.StrideLength, data.StrideDuration]])

    # Make predictions using the loaded model
    predicted_class = xg_model_final.predict(input_data)

    # Decode the predicted class using the label encoder
    decoded_class = encoder.inverse_transform(predicted_class)

    # Return the predicted gait condition
    return {"Predicted_GaitCondition": decoded_class[0]}
