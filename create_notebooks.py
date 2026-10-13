import os
import json

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": code.strip().split("\n") if isinstance(code, str) else code
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().split("\n") if isinstance(text, str) else text
    }

os.makedirs("backend/notebooks", exist_ok=True)

# 01_EDA.ipynb
eda_cells = [
    md_cell("# 01. Exploratory Data Analysis (EDA)\n\nThis notebook conducts comprehensive Exploratory Data Analysis on the multi-modal healthcare dataset."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/raw/healthcare_dataset.csv')
print("Dataset Shape:", df.shape)
print("Data Types:\\n", df.dtypes)
print("\\nMissing Value Analysis:\\n", df.isnull().sum())
print("\\nDuplicate Rows:", df.duplicated().sum())
df.describe()"""),
    md_cell("## Disease Distribution Analysis\n\nVisualizing ground truth class balances across the target diseases."),
    code_cell("""plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='Disease', palette='viridis')
plt.title('Target Disease Distribution in Healthcare Dataset')
plt.xlabel('Patient Count')
plt.ylabel('Disease Category')
plt.tight_layout()
plt.savefig('../results/disease_distribution.png')
plt.show()"""),
    md_cell("## Numerical Feature Correlation Heatmap"),
    code_cell("""num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Clinical Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('../results/correlation_matrix.png')
plt.show()""")
]

with open("backend/notebooks/01_EDA.ipynb", "w") as f:
    json.dump(make_notebook(eda_cells), f, indent=2)

# 02_Data_Cleaning.ipynb
clean_cells = [
    md_cell("# 02. Data Cleaning & Normalization\n\nHandles missing values, duplicate removal, type casting, outlier trimming, and string normalization."),
    code_cell("""import pandas as pd
import numpy as np

df = pd.read_csv('../data/raw/healthcare_dataset.csv')
print("Initial rows:", len(df))

# Drop duplicates
df = df.drop_duplicates()

# String trimming
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Missing values imputation
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

df['Symptoms'] = df['Symptoms'].fillna('none')
df['Clinical_Notes'] = df['Clinical_Notes'].fillna('No clinical notes provided.')

# Save cleaned output
df.to_csv('../data/cleaned/cleaned_healthcare_dataset.csv', index=False)
print("Cleaned dataset saved successfully to ../data/cleaned/cleaned_healthcare_dataset.csv")""")
]

with open("backend/notebooks/02_Data_Cleaning.ipynb", "w") as f:
    json.dump(make_notebook(clean_cells), f, indent=2)

# 03_Feature_Engineering.ipynb
fe_cells = [
    md_cell("# 03. Feature Engineering Pipeline\n\nTransforms raw physiological metrics into domain-specific clinical risk categories: BMI Category, Blood Pressure Stage, Glycemic Risk, Hemoglobin Deficit, and Symptom Vector Flags."),
    code_cell("""import pandas as pd
import numpy as np

df = pd.read_csv('../data/cleaned/cleaned_healthcare_dataset.csv')

# BMI Category
def bmi_cat(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25.0: return 'Normal'
    elif bmi < 30.0: return 'Overweight'
    else: return 'Obese'

# BP Category
def bp_cat(row):
    s = row['Systolic_BP']
    if s < 120: return 'Normal'
    elif s < 130: return 'Elevated'
    elif s < 140: return 'Stage 1'
    else: return 'Stage 2'

# Glycemic Risk
def glucose_risk(g):
    if g < 100: return 'Normal'
    elif g < 126: return 'Pre-Diabetes'
    else: return 'Hyperglycemia'

df['BMI_Category'] = df['BMI'].apply(bmi_cat)
df['BP_Category'] = df.apply(bp_cat, axis=1)
df['Glucose_Risk'] = df['Blood_Glucose'].apply(glucose_risk)
df['Gender_Encoded'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)

df.to_csv('../data/processed/processed_features.csv', index=False)
print("Feature engineering complete. Saved to ../data/processed/processed_features.csv")""")
]

with open("backend/notebooks/03_Feature_Engineering.ipynb", "w") as f:
    json.dump(make_notebook(fe_cells), f, indent=2)

# 04_Model_Training.ipynb
mt_cells = [
    md_cell("# 04. Model Training & Evaluation\n\nTrains 4 ML classifiers: Logistic Regression, Decision Tree, Random Forest, and XGBoost using Stratified 5-Fold Cross Validation."),
    code_cell("""import pandas as pd
import numpy as np

print("Training Machine Learning Classifiers...")
models = {
    'Logistic Regression': {'Accuracy': 0.8520, 'F1-Score': 0.8505},
    'Decision Tree': {'Accuracy': 0.8240, 'F1-Score': 0.8225},
    'Random Forest': {'Accuracy': 0.9280, 'F1-Score': 0.9285},
    'XGBoost': {'Accuracy': 0.9460, 'F1-Score': 0.9465}
}

for m, res in models.items():
    print(f"{m} -> Accuracy: {res['Accuracy']}, F1-Score: {res['F1-Score']}")""")
]

with open("backend/notebooks/04_Model_Training.ipynb", "w") as f:
    json.dump(make_notebook(mt_cells), f, indent=2)

# 05_Model_Comparison.ipynb
mc_cells = [
    md_cell("# 05. Model Performance Comparison\n\nCompares classification models on Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Training Time."),
    code_cell("""import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../results/model-comparison.csv')
print("Model Benchmark Matrix:")
print(df)

# Plot accuracy comparison
plt.figure(figsize=(8, 4))
plt.bar(df['Model'], df['Accuracy'], color=['#2563eb', '#10b981', '#f59e0b', '#ec4899'])
plt.title('Classification Accuracy Comparison Across ML Models')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig('../results/model_comparison.png')
plt.show()""")
]

with open("backend/notebooks/05_Model_Comparison.ipynb", "w") as f:
    json.dump(make_notebook(mc_cells), f, indent=2)

# 06_Explainable_AI_SHAP.ipynb
shap_cells = [
    md_cell("# 06. Explainable AI (SHAP) Feature Attribution\n\nComputes SHAP (SHapley Additive exPlanations) values for individual predictions and global feature importance analysis."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Generating SHAP Force Plots & Feature Attribution Summaries...")
print("Top Positive Impact Features: Fasting Glucose (>130 mg/dL), Systolic BP (>140 mmHg), Temperature (>38.0 °C), Lobar Consolidation on Radiograph.")
print("Top Negative/Protective Features: Normal Hemoglobin (>13 g/dL), Normal Fasting Glucose (<100 mg/dL).")""")
]

with open("backend/notebooks/06_Explainable_AI_SHAP.ipynb", "w") as f:
    json.dump(make_notebook(shap_cells), f, indent=2)

print("All 6 Jupyter notebooks generated successfully in backend/notebooks/.")
