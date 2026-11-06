"""
=============================================================================
LoanIQ - Machine Learning Risk Assessment Engine (model.py)
AI for Finance CIA-2 Project
=============================================================================
This module generates synthetic financial applicant data, trains a
Random Forest classifier for credit risk categorization (Low, Medium, High),
and provides inference functions used by the Streamlit application.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Paths for persisting artifacts
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "synthetic_loans.csv")
MODEL_PATH = os.path.join(DATA_DIR, "loan_model.pkl")

# Mapping of employment types to numeric values for ML processing
EMPLOYMENT_MAP = {
    "Salaried": 0,
    "Self-Employed": 1,
    "Business Owner": 2,
    "Freelancer": 3
}

# The feature columns used by the Random Forest model
FEATURE_COLUMNS = [
    "age",
    "employment_encoded",
    "monthly_income",
    "existing_debt",
    "credit_score",
    "loan_amount",
    "tenure_months",
    "interest_rate",
    "emi",
    "dti_ratio",
    "loan_to_income_ratio"
]


def calculate_emi_raw(principal: float, annual_rate: float, tenure_months: int) -> float:
    """
    Standard Financial EMI Formula:
    EMI = [P x r x (1+r)^n] / [(1+r)^n - 1]
    where:
    - P is principal loan amount
    - r is monthly interest rate
    - n is tenure in months
    """
    if tenure_months <= 0 or principal <= 0:
        return 0.0
    
    monthly_rate = (annual_rate / 100.0) / 12.0
    if monthly_rate == 0:
        return principal / tenure_months
        
    emi = (principal * monthly_rate * ((1.0 + monthly_rate) ** tenure_months)) / (
        ((1.0 + monthly_rate) ** tenure_months) - 1.0
    )
    return round(emi, 2)


def generate_synthetic_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for loan credit risk classification.
    Uses realistic financial heuristics with minor probabilistic variation to reflect
    genuine underwriting criteria.
    """
    np.random.seed(random_state)
    
    # 1. Applicant demographics & base inputs
    ages = np.random.randint(21, 65, size=n_samples)
    emp_types = np.random.choice(
        ["Salaried", "Self-Employed", "Business Owner", "Freelancer"],
        size=n_samples,
        p=[0.55, 0.25, 0.12, 0.08]
    )
    
    # Incomes: Log-normal distribution scaled realistically (₹20,000 to ₹350,000)
    incomes = np.random.lognormal(mean=10.8, sigma=0.55, size=n_samples)
    incomes = np.clip(incomes, 20000, 350000).round(-2)
    
    # Credit scores: Beta-like distribution shifted to range [300, 850]
    credit_scores = (np.random.beta(a=5, b=2.2, size=n_samples) * 550 + 300).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)
    
    # Existing debts: Typically 0% to 35% of monthly income
    debt_factor = np.random.uniform(0.0, 0.38, size=n_samples)
    # Some applicants carry no debt (zero debt probability)
    no_debt_mask = np.random.rand(n_samples) < 0.22
    debt_factor[no_debt_mask] = 0.0
    existing_debts = (incomes * debt_factor).round(-2)
    
    # Requested loan amounts: Typically 1x to 15x monthly income
    loan_multiplier = np.random.uniform(2.0, 14.0, size=n_samples)
    loan_amounts = (incomes * loan_multiplier).round(-3)
    loan_amounts = np.clip(loan_amounts, 50000, 5000000)
    
    # Tenure in months: Typical tenures between 12 and 180 months
    tenures = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180], size=n_samples)
    
    # Interest rates: typically between 8.5% and 18.0%
    interest_rates = np.random.uniform(8.5, 17.5, size=n_samples).round(2)
    
    # 2. Financial Metrics Calculations
    emis = []
    dtis = []
    ltis = []
    
    for i in range(n_samples):
        emi_val = calculate_emi_raw(loan_amounts[i], interest_rates[i], tenures[i])
        total_debt = existing_debts[i] + emi_val
        dti_val = (total_debt / incomes[i]) * 100.0
        lti_val = loan_amounts[i] / (incomes[i] * 12.0)
        
        emis.append(emi_val)
        dtis.append(round(dti_val, 2))
        ltis.append(round(lti_val, 2))
        
    emis = np.array(emis)
    dtis = np.array(dtis)
    ltis = np.array(ltis)
    
    # 3. Ground Truth Label Generation (Low, Medium, High Risk)
    risk_labels = []
    for i in range(n_samples):
        score = credit_scores[i]
        dti = dtis[i]
        lti = ltis[i]
        
        # Scoring logic based on banking underwriting guidelines:
        # High Risk: Bad credit score, high DTI, excessive leverage
        if score < 580 or dti > 52.0 or (dti > 45.0 and score < 640) or (lti > 4.5 and score < 620):
            risk = "High"
        # Low Risk: Prime credit score, conservative DTI, sensible leverage
        elif score >= 730 and dti <= 38.0 and lti <= 3.5:
            risk = "Low"
        # Medium Risk: Acceptable credit score and manageable leverage
        else:
            risk = "Medium"
            
        # Add realistic 3% market boundary stochasticity
        noise = np.random.rand()
        if noise < 0.03:
            if risk == "Low":
                risk = "Medium"
            elif risk == "High":
                risk = "Medium"
            else:
                risk = "Low" if score > 680 else "High"
                
        risk_labels.append(risk)
        
    df = pd.DataFrame({
        "age": ages,
        "employment_type": emp_types,
        "employment_encoded": [EMPLOYMENT_MAP[e] for e in emp_types],
        "monthly_income": incomes,
        "existing_debt": existing_debts,
        "credit_score": credit_scores,
        "loan_amount": loan_amounts,
        "tenure_months": tenures,
        "interest_rate": interest_rates,
        "emi": emis,
        "dti_ratio": dtis,
        "loan_to_income_ratio": ltis,
        "risk_tier": risk_labels
    })
    
    return df


def train_and_save_model() -> dict:
    """
    Generates synthetic training data, trains a Random Forest model,
    evaluates it, and saves the trained artifacts to the data directory.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("[LoanIQ] Generating synthetic financial dataset...")
    df = generate_synthetic_data(n_samples=2500, random_state=42)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[LoanIQ] Dataset saved successfully to: {DATASET_PATH}")
    
    # Feature matrix X and target y
    X = df[FEATURE_COLUMNS]
    y = df["risk_tier"]
    
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("[LoanIQ] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"[LoanIQ] Model Accuracy: {acc * 100:.2f}%")
    
    # Feature importances
    feature_importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "employment_map": EMPLOYMENT_MAP,
        "classes": list(model.classes_),
        "accuracy": acc,
        "report": report,
        "feature_importances": feature_importances
    }
    
    joblib.dump(artifact, MODEL_PATH)
    print(f"[LoanIQ] Model artifact saved to: {MODEL_PATH}")
    return artifact


def load_or_train_model() -> dict:
    """
    Loads pre-trained model artifact if available, otherwise trains a new one.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            # If load fails due to version or file corruption, re-train
            return train_and_save_model()
    else:
        return train_and_save_model()


def predict_risk(applicant_data: dict, model_artifact: dict = None) -> dict:
    """
    Performs credit risk classification on a single applicant's profile.
    
    applicant_data expects:
    - age (int)
    - employment_type (str)
    - monthly_income (float)
    - existing_debt (float)
    - credit_score (int)
    - loan_amount (float)
    - tenure_months (int)
    - interest_rate (float)
    - emi (float)
    - dti_ratio (float)
    """
    if model_artifact is None:
        model_artifact = load_or_train_model()
        
    model = model_artifact["model"]
    classes = model_artifact["classes"]
    
    # Encode categorical employment
    emp_encoded = EMPLOYMENT_MAP.get(applicant_data.get("employment_type", "Salaried"), 0)
    
    # Calculate derived ratio
    monthly_inc = max(applicant_data.get("monthly_income", 1.0), 1.0)
    loan_amt = applicant_data.get("loan_amount", 0.0)
    lti = round(loan_amt / (monthly_inc * 12.0), 2)
    
    # Build single-row DataFrame aligned with training feature columns
    row = pd.DataFrame([{
        "age": applicant_data.get("age", 30),
        "employment_encoded": emp_encoded,
        "monthly_income": monthly_inc,
        "existing_debt": applicant_data.get("existing_debt", 0.0),
        "credit_score": applicant_data.get("credit_score", 700),
        "loan_amount": loan_amt,
        "tenure_months": applicant_data.get("tenure_months", 36),
        "interest_rate": applicant_data.get("interest_rate", 10.5),
        "emi": applicant_data.get("emi", 0.0),
        "dti_ratio": applicant_data.get("dti_ratio", 0.0),
        "loan_to_income_ratio": lti
    }])[FEATURE_COLUMNS]
    
    # Predict risk tier & class probabilities
    pred_tier = model.predict(row)[0]
    pred_probs = model.predict_proba(row)[0]
    
    prob_dict = {cls_name: float(round(prob * 100, 1)) for cls_name, prob in zip(classes, pred_probs)}
    
    # Ensure ordered keys for consistent display
    ordered_probs = {
        "Low": float(prob_dict.get("Low", 0.0)),
        "Medium": float(prob_dict.get("Medium", 0.0)),
        "High": float(prob_dict.get("High", 0.0))
    }
    
    confidence = float(ordered_probs[pred_tier])
    
    return {
        "predicted_risk": pred_tier,
        "confidence": confidence,
        "probabilities": ordered_probs,
        "feature_importances": model_artifact["feature_importances"],
        "loan_to_income_ratio": lti
    }


if __name__ == "__main__":
    # Test script standalone execution
    artifact = train_and_save_model()
    print("Test prediction on sample applicant:")
    sample = {
        "age": 32,
        "employment_type": "Salaried",
        "monthly_income": 85000,
        "existing_debt": 12000,
        "credit_score": 760,
        "loan_amount": 500000,
        "tenure_months": 36,
        "interest_rate": 10.5,
        "emi": 16254,
        "dti_ratio": 33.24
    }
    result = predict_risk(sample, artifact)
    print("Prediction Result:", result)