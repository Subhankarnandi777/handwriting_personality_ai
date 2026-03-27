import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import MODELS_ML

def train_dummy_model():
    """
    Creates a synthetic dataset to train the RandomForestRegressor,
    saving it to models/ml_models/personality_model.pkl
    so the system uses ML instead of the rule engine automatically.
    
    In a real-world scenario, you would replace `X` and `y` with data
    parsed from a labeled handwriting dataset (like IAM).
    """
    print("Generating simulated dataset...")
    n_samples = 1000
    n_features = 2830 # Approx size of the fused_vec
    
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, 5) # 5 Big Five traits
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
    ])
    
    print("Training Random Forest Regressor...")
    pipeline.fit(X, y)
    
    os.makedirs(MODELS_ML, exist_ok=True)
    out_path = os.path.join(MODELS_ML, "personality_model.pkl")
    joblib.dump(pipeline, out_path)
    
    print(f"Model successfully saved to {out_path}")
    print("The pipeline will now automatically load and use this ML model.")

if __name__ == "__main__":
    train_dummy_model()
