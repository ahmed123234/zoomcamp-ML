import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load the Pipeline Model
MODEL_FILE = 'pipeline_v2.bin'
try:
    with open(MODEL_FILE, 'rb') as f_in:
        pipeline = pickle.load(f_in)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
    exit()

# 2. Define the Input Data Schema
# This ensures that the incoming JSON matches the expected data types
class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# 3. Initialize the FastAPI app
app = FastAPI()

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_conversion(data: LeadData):
    # Convert Pydantic model to a dictionary that the DictVectorizer expects
    lead_dict = data.dict()
    
    # The pipeline expects a list of dictionaries, even for one record
    features = [lead_dict]
    
    # Predict probability for the positive class (1)
    # [0, 1] means: take the first (and only) prediction, and the second element (prob of 1)
    probability = pipeline.predict_proba(features)[0, 1]
    
    # Return the result
    return {
        "conversion_probability": float(probability)
    }

if __name__ == '__main__':
    import uvicorn
    # You would typically run this via the command line, but this is helpful for testing.
    uvicorn.run(app, host="0.0.0.0", port=8000)
