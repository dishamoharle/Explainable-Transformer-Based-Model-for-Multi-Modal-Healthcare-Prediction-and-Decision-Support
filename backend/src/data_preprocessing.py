import pandas as pd
import numpy as np

class HealthcareDataPreprocessor:
    """
    Data Preprocessing module for Healthcare Multi-Modal Dataset.
    Handles missing values, duplicate removal, categorical encoding, and feature scaling.
    """
    def __init__(self):
        self.feature_means = {}
        self.categorical_mappings = {}

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates, trims strings, and handles missing values."""
        cleaned_df = df.copy()
        
        # 1. Drop exact duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # 2. String trimming for string columns
        str_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in str_cols:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            
        # 3. Handle missing values
        num_cols = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
                    'Heart_Rate', 'Temperature_C', 'Blood_Glucose', 'Cholesterol', 'Hemoglobin']
        for col in num_cols:
            if col in cleaned_df.columns:
                mean_val = cleaned_df[col].mean()
                self.feature_means[col] = mean_val
                cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                
        if 'Symptoms' in cleaned_df.columns:
            cleaned_df['Symptoms'] = cleaned_df['Symptoms'].fillna('none')
            
        if 'Clinical_Notes' in cleaned_df.columns:
            cleaned_df['Clinical_Notes'] = cleaned_df['Clinical_Notes'].fillna('No additional notes provided.')
            
        return cleaned_df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical features like Gender into numerical representations."""
        encoded_df = df.copy()
        if 'Gender' in encoded_df.columns:
            encoded_df['Gender_Encoded'] = encoded_df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)
        return encoded_df

    def normalize_numerical(self, df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
        """Normalizes numerical features using Standard Z-Score normalization."""
        norm_df = df.copy()
        for col in num_cols:
            if col in norm_df.columns:
                mean = norm_df[col].mean()
                std = norm_df[col].std()
                if std > 0:
                    norm_df[f'{col}_Scaled'] = (norm_df[col] - mean) / std
        return norm_df


if __name__ == "__main__":
    print("Testing HealthcareDataPreprocessor module...")
    df = pd.read_csv("backend/data/raw/healthcare_dataset.csv")
    preprocessor = HealthcareDataPreprocessor()
    cleaned = preprocessor.clean_raw_data(df)
    encoded = preprocessor.encode_categorical(cleaned)
    encoded.to_csv("backend/data/cleaned/cleaned_healthcare_dataset.csv", index=False)
    print("Preprocessing completed successfully. Cleaned shape:", encoded.shape)
