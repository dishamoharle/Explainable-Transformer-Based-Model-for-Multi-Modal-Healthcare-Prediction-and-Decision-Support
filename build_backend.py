import os
import json
import base64

def make_dirs():
    dirs = [
        "backend",
        "backend/data/raw",
        "backend/data/cleaned",
        "backend/data/processed",
        "backend/notebooks",
        "backend/models/trained_models",
        "backend/models/preprocessing",
        "backend/src",
        "backend/results",
        "backend/results/confusion_matrices",
        "backend/results/shap_plots"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories created successfully.")

make_dirs()
