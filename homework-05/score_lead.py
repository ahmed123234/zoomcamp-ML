import pickle
import numpy as np

# 1. Define the model file and the new record
MODEL_FILE = 'pipeline_v1.bin'
NEW_RECORD = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# 2. Load the pipeline using pickle
try:
    with open(MODEL_FILE, 'rb') as f_in:
        pipeline = pickle.load(f_in)
    
    # Check if the pipeline was loaded successfully
    print("Pipeline loaded successfully.")

except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found. Did you run the wget command?")
    exit()
except Exception as e:
    print(f"Error loading the pipeline: {e}")
    # If the error is related to 'DictVectorizer' or 'LogisticRegression', 
    # ensure scikit-learn is installed in your virtual environment: 
    # uv pip install scikit-learn==1.6.1
    exit()

# 3. Score the record
# The pipeline expects a list of dictionaries, even for a single record
features = [NEW_RECORD]
probability = pipeline.predict_proba(features)[0, 1] # Probability of the positive class (1)

# 4. Print the result
print(f"Features: {NEW_RECORD}")
print(f"Probability of conversion (Positive Class): {probability:.3f}")
