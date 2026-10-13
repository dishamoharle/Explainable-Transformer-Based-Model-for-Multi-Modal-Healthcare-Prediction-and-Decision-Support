# Multi-Modal Explainable Healthcare Transformer - Machine Learning Backend

This directory contains the complete end-to-end Machine Learning healthcare backend pipeline for the **Explainable Transformer-Based Model for Multi-Modal Healthcare Prediction and Decision Support** project.

---

## 📁 Folder Structure

```
backend/
├── app.py                      # Flask / FastAPI REST API server
├── requirements.txt            # Python dependencies
│
├── data/                       # Healthcare Datasets
│   ├── raw/
│   │   └── healthcare_dataset.csv
│   ├── cleaned/
│   │   └── cleaned_healthcare_dataset.csv
│   └── processed/
│       └── processed_features.csv
│
├── notebooks/                  # Step-by-Step Jupyter Notebooks
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Data_Cleaning.ipynb  # Cleaning, Imputation & Type Casting
│   ├── 03_Feature_Engineering.ipynb # Clinical Risk Indicators & Categories
│   ├── 04_Model_Training.ipynb # Logistic Regression, Decision Tree, Random Forest, XGBoost
│   ├── 05_Model_Comparison.ipynb # Comparative Performance Benchmark Metrics
│   └── 06_Explainable_AI_SHAP.ipynb # SHAP Force & Feature Attribution
│
├── models/                     # Model Binaries & Artifacts
│   ├── trained_models/         # Logistic Regression, Decision Tree, Random Forest, XGBoost
│   └── preprocessing/          # Scalers & Encoders
│
├── src/                        # Modular Python Pipeline Scripts
│   ├── data_preprocessing.py   # Preprocessing & Normalization
│   ├── feature_engineering.py  # Clinical Risk Encoding & Feature Calculation
│   ├── train_models.py         # Stratified K-Fold Training & Model Serializer
│   ├── model_comparison.py     # Evaluation Matrix Generator
│   ├── prediction.py           # Multi-Modal Transformer Fusion Model
│   └── explainability.py       # Dynamic SHAP Explainer Engine
│
└── results/                    # Performance Metrics & Visual Plots
    ├── model-comparison.csv    # Metric Data (Accuracy, Precision, Recall, F1, ROC-AUC)
    ├── model_comparison.png    # Comparison Graph
    ├── confusion_matrices/     # Confusion Matrix PNGs
    └── shap_plots/             # SHAP Summary, Bar, and Waterfall Plots
```

---

## 🧠 Multi-Modal Transformer Architecture

The pipeline processes 3 distinct clinical modalities and fuses them using a **Multi-Head Self-Attention Fusion Layer**:

1. **Tabular Data Encoder**: Processes age, BMI, blood pressure, glucose, cholesterol, and hemoglobin metrics.
2. **Clinical Text Encoder**: Embeds unstructured physician notes and symptoms (ClinicalBERT token representations).
3. **Medical Image Encoder**: Feature extractor for radiological scans (Chest X-Ray, CT Scan, MRI).
4. **Cross-Modal Fusion Layer**: Fuses vector representations from all 3 encoders into a combined decision matrix to predict risk scores and target diseases.

---

## 📊 Supported Target Diseases

1. **Healthy**
2. **Pneumonia**
3. **Diabetes / Hyperglycemia**
4. **Cardiovascular Disease**
5. **Acute Appendicitis**
6. **Stroke Risk**
7. **Anemia**

---

## 🔌 REST API Endpoints (`app.py`)

- `GET /health`: Health check endpoint.
- `POST /predict`: Receives patient profile, returns predicted disease, confidence, risk level, and SHAP features.
- `POST /upload`: Handles diagnostic image uploads.
- `GET /model-comparison`: Loads benchmark performance metrics from `backend/results/model-comparison.csv`.
- `POST /explain`: Generates dynamic SHAP impact scores for a specific patient input.

---

## 🚀 How to Run the Backend

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Run the API server
python backend/app.py
```
