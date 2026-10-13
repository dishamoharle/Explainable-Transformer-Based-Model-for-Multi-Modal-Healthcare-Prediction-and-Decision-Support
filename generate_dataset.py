import os
import json
import base64
import random

def generate_dataset():
    random.seed(42)
    diseases = [
        "Healthy",
        "Pneumonia",
        "Diabetes",
        "Cardiovascular Disease",
        "Acute Appendicitis",
        "Stroke Risk",
        "Anemia"
    ]
    
    symptoms_by_disease = {
        "Healthy": ["none"],
        "Pneumonia": ["cough", "fever", "dyspnea", "chest pain", "chills"],
        "Diabetes": ["polyuria", "polydipsia", "fatigue", "blurred vision"],
        "Cardiovascular Disease": ["chest pain", "shortness of breath", "palpitations", "dizziness"],
        "Acute Appendicitis": ["abdominal pain", "fever", "nausea", "rebound tenderness"],
        "Stroke Risk": ["dysarthria", "hemiparesis", "dizziness", "facial droop"],
        "Anemia": ["fatigue", "dizziness", "weakness", "pallor", "shortness of breath"]
    }
    
    rows = [
        "Patient_ID,Age,Gender,Height_cm,Weight_kg,BMI,Systolic_BP,Diastolic_BP,Heart_Rate,Temperature_C,Blood_Glucose,Cholesterol,Hemoglobin,Symptoms,Clinical_Notes,Disease"
    ]
    
    for i in range(1, 1001):
        pid = f"PAT-2026-{1000 + i}"
        disease = random.choice(diseases)
        gender = random.choice(["Male", "Female"])
        age = random.randint(18, 80)
        
        # Adjust features based on target disease to create strong ground truth relationships
        if disease == "Healthy":
            height = random.randint(160, 180)
            weight = random.randint(55, 75)
            bmi = round(weight / ((height/100)**2), 1)
            sys_bp = random.randint(110, 120)
            dia_bp = random.randint(70, 80)
            hr = random.randint(60, 80)
            temp = round(random.uniform(36.4, 37.0), 1)
            glucose = random.randint(75, 100)
            choles = random.randint(150, 195)
            hb = round(random.uniform(13.0, 16.0), 1)
            notes = "Patient presents for routine wellness evaluation. Physical exam and vital metrics within standard reference ranges."
            
        elif disease == "Pneumonia":
            height = random.randint(155, 185)
            weight = random.randint(50, 85)
            bmi = round(weight / ((height/100)**2), 1)
            sys_bp = random.randint(110, 130)
            dia_bp = random.randint(70, 85)
            hr = random.randint(95, 120)
            temp = round(random.uniform(38.2, 39.8), 1)
            glucose = random.randint(90, 120)
            choles = random.randint(160, 200)
            hb = round(random.uniform(12.0, 14.5), 1)
            notes = "Patient presents with persistent fever, productive cough with rusty sputum, and breathlessness. Chest radiograph shows right lobar consolidation."
            
        elif disease == "Diabetes":
            height = random.randint(155, 180)
            weight = random.randint(75, 110)
            bmi = round(weight / ((height/100)**2), 1)
            sys_bp = random.randint(125, 145)
            dia_bp = random.randint(80, 90)
            hr = random.randint(75, 90)
            temp = round(random.uniform(36.5, 37.2), 1)
            glucose = random.randint(160, 280)
            choles = random.randint(200, 260)
            hb = round(random.uniform(12.5, 15.0), 1)
            notes = "Patient reports progressive fatigue, polyuria, and polydipsia. Fasting plasma glucose significantly elevated above baseline."
            
        elif disease == "Cardiovascular Disease":
            age = random.randint(50, 82)
            height = random.randint(160, 182)
            weight = random.randint(70, 105)
            bmi = round(weight / ((height/100)**2), 1)
            sys_bp = random.randint(145, 180)
            dia_bp = random.randint(90, 110)
            hr = random.randint(80, 105)
            temp = round(random.uniform(36.5, 37.1), 1)
            glucose = random.randint(110, 160)
            choles = random.randint(230, 310)
            hb = round(random.uniform(12.5, 15.5), 1)
            notes = "Patient complains of retrosternal chest tightness radiating to left shoulder. Elevated blood pressure and hyperlipidemia noted."
            
        elif disease == "Acute Appendicitis":
            age = random.randint(18, 45)
            height = random.randint(160, 180)
            weight = random.randint(55, 80)
            bmi = round(weight / ((height/100)**2), 1)
            sys_bp = random.randint(110, 130)
            dia_bp = random.randint(70, 85)
            hr = random.randint(90, 115)
            temp = round(random.uniform(37.8, 38.9), 1)
            glucose = random.randint(85, 115)
            choles = random.randint(150, 200)
            hb = round(random.uniform(12.5, 15.0), 1)
            notes = "Acute onset right lower quadrant abdominal pain with McBurney tenderness and fever. CT scan reveals focal appendiceal enlargement."
            
        elif disease == "Stroke Risk":
            age = random.randint(58, 85)
            height = random.randint(158, 180)
            weight = random.randint(65, 95)
            bmi = round(weight / ((height/100)**2), 1)
            sys_bp = random.randint(150, 185)
            dia_bp = random.randint(92, 115)
            hr = random.randint(75, 95)
            temp = round(random.uniform(36.4, 37.0), 1)
            glucose = random.randint(120, 170)
            choles = random.randint(210, 270)
            hb = round(random.uniform(12.0, 14.8), 1)
            notes = "Transient episode of dysarthria and right upper limb hemiparesis. Brain MRI confirms focal cerebral ischemia and internal carotid stenosis."
            
        elif disease == "Anemia":
            height = random.randint(152, 175)
            weight = random.randint(45, 68)
            bmi = round(weight / ((height/100)**2), 1)
            sys_bp = random.randint(100, 115)
            dia_bp = random.randint(60, 75)
            hr = random.randint(85, 110)
            temp = round(random.uniform(36.2, 36.8), 1)
            glucose = random.randint(80, 105)
            choles = random.randint(140, 190)
            hb = round(random.uniform(7.0, 10.8), 1)
            notes = "Severe fatigue, dizziness, and conjunctival pallor. Complete blood count confirms low hemoglobin levels indicative of anemia."

        symptoms_str = ";".join(random.sample(symptoms_by_disease[disease], min(3, len(symptoms_by_disease[disease]))))
        rows.append(f"{pid},{age},{gender},{height},{weight},{bmi},{sys_bp},{dia_bp},{hr},{temp},{glucose},{choles},{hb},{symptoms_str},\"{notes}\",{disease}")

    csv_content = "\n".join(rows)
    
    # Save raw
    with open("backend/data/raw/healthcare_dataset.csv", "w") as f:
        f.write(csv_content)
        
    # Save cleaned
    with open("backend/data/cleaned/cleaned_healthcare_dataset.csv", "w") as f:
        f.write(csv_content)
        
    # Save processed features CSV with categorical encodings
    processed_rows = ["Patient_ID,Age,Gender_Encoded,BMI,BMI_Category,Systolic_BP,Diastolic_BP,BP_Category,Glucose,Glucose_Risk,Cholesterol,Hemoglobin,Hemoglobin_Risk,Disease"]
    for line in rows[1:]:
        parts = line.split(",")
        pid = parts[0]
        age = int(parts[1])
        gender_enc = 1 if parts[2] == "Male" else 0
        bmi = float(parts[5])
        bmi_cat = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
        sys_bp = int(parts[6])
        bp_cat = 0 if sys_bp < 120 else (1 if sys_bp < 130 else (2 if sys_bp < 140 else 3))
        glucose = int(parts[10])
        g_risk = 0 if glucose < 100 else (1 if glucose < 126 else 2)
        choles = int(parts[11])
        hb = float(parts[12])
        hb_risk = 1 if hb < 12.0 else 0
        disease = parts[-1]
        processed_rows.append(f"{pid},{age},{gender_enc},{bmi},{bmi_cat},{sys_bp},{parts[7]},{bp_cat},{glucose},{g_risk},{choles},{hb},{hb_risk},{disease}")
        
    with open("backend/data/processed/processed_features.csv", "w") as f:
        f.write("\n".join(processed_rows))
        
    print("Datasets generated successfully.")

generate_dataset()
