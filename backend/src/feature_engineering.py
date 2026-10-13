import pandas as pd
import numpy as np

class HealthcareFeatureEngineer:
    """
    Feature Engineering module for multi-modal healthcare analysis.
    Generates domain-specific clinical indicators: BMI categories, Blood Pressure stages,
    Glycemic risk levels, Hemoglobin risk levels, and symptom vector flags.
    """

    @staticmethod
    def get_bmi_category(bmi: float) -> str:
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25.0:
            return "Normal Weight"
        elif bmi < 30.0:
            return "Overweight"
        else:
            return "Obese"

    @staticmethod
    def get_bp_category(sys_bp: int, dia_bp: int) -> str:
        if sys_bp < 120 and dia_bp < 80:
            return "Normal BP"
        elif sys_bp < 130 and dia_bp < 80:
            return "Elevated BP"
        elif sys_bp < 140 or dia_bp < 90:
            return "Stage 1 Hypertension"
        else:
            return "Stage 2 Hypertension"

    @staticmethod
    def get_glucose_risk(glucose: int) -> str:
        if glucose < 100:
            return "Normal Fasting"
        elif glucose < 126:
            return "Pre-Diabetes Sign"
        else:
            return "High Fasting Hyperglycemia"

    @staticmethod
    def get_hemoglobin_risk(hb: float) -> str:
        if hb < 10.0:
            return "Severe Anemic Deficit"
        elif hb < 12.0:
            return "Mild-Moderate Anemia"
        else:
            return "Standard Hemoglobin Level"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering transformations across all dataset rows."""
        fe_df = df.copy()

        if 'BMI' in fe_df.columns:
            fe_df['BMI_Category'] = fe_df['BMI'].apply(self.get_bmi_category)
            fe_df['BMI_Category_Code'] = fe_df['BMI_Category'].map({
                'Underweight': 0, 'Normal Weight': 1, 'Overweight': 2, 'Obese': 3
            }).fillna(1)

        if 'Systolic_BP' in fe_df.columns and 'Diastolic_BP' in fe_df.columns:
            fe_df['BP_Category'] = fe_df.apply(lambda r: self.get_bp_category(int(r['Systolic_BP']), int(r['Diastolic_BP'])), axis=1)
            fe_df['BP_Category_Code'] = fe_df['BP_Category'].map({
                'Normal BP': 0, 'Elevated BP': 1, 'Stage 1 Hypertension': 2, 'Stage 2 Hypertension': 3
            }).fillna(0)

        if 'Blood_Glucose' in fe_df.columns:
            fe_df['Glucose_Risk'] = fe_df['Blood_Glucose'].apply(self.get_glucose_risk)
            fe_df['Glucose_Risk_Code'] = fe_df['Glucose_Risk'].map({
                'Normal Fasting': 0, 'Pre-Diabetes Sign': 1, 'High Fasting Hyperglycemia': 2
            }).fillna(0)

        if 'Hemoglobin' in fe_df.columns:
            fe_df['Hemoglobin_Risk'] = fe_df['Hemoglobin'].apply(self.get_hemoglobin_risk)
            fe_df['Hemoglobin_Risk_Code'] = fe_df['Hemoglobin_Risk'].map({
                'Standard Hemoglobin Level': 0, 'Mild-Moderate Anemia': 1, 'Severe Anemic Deficit': 2
            }).fillna(0)

        return fe_df


if __name__ == "__main__":
    print("Testing HealthcareFeatureEngineer module...")
    df = pd.read_csv("backend/data/cleaned/cleaned_healthcare_dataset.csv")
    engineer = HealthcareFeatureEngineer()
    processed = engineer.transform(df)
    processed.to_csv("backend/data/processed/processed_features.csv", index=False)
    print("Feature engineering completed successfully. Processed shape:", processed.shape)
