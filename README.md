# Explainable Transformer-Based Model for Multi-Modal Healthcare Prediction and Decision Support

An end-to-end Machine Learning healthcare platform integrating a **React + TypeScript frontend** with a **Python Machine Learning backend**.

---

## 🌟 Key Features

- **Multi-Modal Data Integration**: Fuses structured physiological metrics, unstructured clinical notes, and medical imaging (X-Ray, CT Scan, MRI).
- **Explainable AI (SHAP)**: Provides feature attribution impact scores for every prediction.
- **Model Performance Comparison**: Compares Logistic Regression, Decision Tree, Random Forest, and XGBoost on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Interactive React Frontend**: Patient registration, diagnostic data logs, AI analytics dashboard, and model performance visualization.

---

## 📁 System Architecture & Directory Layout

```
project-root/
│
├── frontend/                   # React + TypeScript + Vite Application
│   └── src/
│       ├── components/
│       │   ├── PatientInformationForm.tsx
│       │   ├── MedicalDataInput.tsx
│       │   ├── AiAnalysisDashboard.tsx
│       │   └── ModelPerformanceComparison.tsx
│       ├── App.tsx
│       └── types.ts
│
├── backend/                    # Python ML Backend Engine
│   ├── app.py                  # API Server (Flask / FastAPI endpoints)
│   ├── requirements.txt
│   ├── data/                   # Raw, Cleaned, & Processed Datasets
│   ├── notebooks/              # Jupyter Notebooks (01_EDA to 06_SHAP)
│   ├── models/                 # Model Artifacts
│   ├── src/                    # Modular Pipeline Scripts
│   └── results/                # model-comparison.csv & Chart Plots
│
└── README.md
```

---

## 🚀 How to Run the Application

### 1. Start the Express / Node Server (Frontend + API Bridge)
```bash
npm run dev
```

### 2. Run Python ML Backend API
```bash
python backend/app.py
```
