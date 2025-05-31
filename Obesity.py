# obesity_predictor.py

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Load the trained model and preprocessing objects
model = joblib.load('obesity_model.pkl')               # Change filename if needed
scaler = joblib.load('scaler.pkl')                     # StandardScaler used during training
encoder = joblib.load('label_encoder.pkl')             # Encoder for categorical columns (optional)

# Step 2: Define your input data (you can modify or load from CSV)
# Example input - as a dictionary
new_data = {
    'Gender': ['Male'],
    'Age': [25],
    'Height': [1.75],
    'Weight': [85],
    'family_history_with_overweight': ['yes'],
    'FAVC': ['yes'],
    'FCVC': [3],
    'NCP': [3],
    'CAEC': ['Sometimes'],
    'SMOKE': ['no'],
    'CH2O': [2],
    'SCC': ['no'],
    'FAF': [0],
    'TUE': [1],
    'CALC': ['Sometimes'],
    'MTRANS': ['Public_Transportation']
}

# Step 3: Create DataFrame
input_df = pd.DataFrame(new_data)

# Step 4: Encode categorical variables (match training pipeline)
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
                       'SMOKE', 'SCC', 'CALC', 'MTRANS']

for col in categorical_columns:
    input_df[col] = encoder[col].transform(input_df[col])

# Step 5: Scale numerical features
numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# Step 6: Predict
prediction = model.predict(input_df)
print("Predicted Obesity Category:", prediction[0])