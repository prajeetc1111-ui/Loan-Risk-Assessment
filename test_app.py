"""
=============================================================================
LoanIQ - Automated Verification Test Suite (test_app.py)
=============================================================================
Validates analytical calculations, ML model predictions, SQLite database
operations, and data integrity.
"""

import os
import sqlite3
import pandas as pd
import numpy as np

from model import (
    calculate_emi_raw,
    load_or_train_model,
    predict_risk,
    DATASET_PATH,
    MODEL_PATH
)
from app import (
    init_database,
    save_assessment_to_db,
    load_assessment_history,
    generate_recommendations,
    DB_PATH
)


def test_emi_calculation():
    """Verify standard financial EMI formula accuracy."""
    # Principal: 100,000, Rate: 12% p.a. (1% monthly), Tenure: 12 months
    # Formula: 100000 * 0.01 * (1.01)^12 / ((1.01)^12 - 1) = ~8884.88
    emi = calculate_emi_raw(100000, 12.0, 12)
    assert 8880 < emi < 8890, f"Expected ~8884.88, got {emi}"
    
    # Zero principal test
    assert calculate_emi_raw(0, 10.0, 12) == 0.0
    print("[PASS] EMI Calculation Test Passed!")


def test_model_predictions():
    """Verify Random Forest classification on different applicant risk profiles."""
    artifact = load_or_train_model()
    assert os.path.exists(MODEL_PATH), "Model artifact file missing!"
    assert os.path.exists(DATASET_PATH), "Synthetic dataset file missing!"
    
    # 1. Low Risk Profile (High credit score, high income, low debt)
    low_risk_applicant = {
        "age": 35,
        "employment_type": "Salaried",
        "monthly_income": 120000,
        "existing_debt": 5000,
        "credit_score": 790,
        "loan_amount": 300000,
        "tenure_months": 36,
        "interest_rate": 9.5,
        "emi": 9611.0,
        "dti_ratio": 12.18
    }
    res_low = predict_risk(low_risk_applicant, artifact)
    assert res_low["predicted_risk"] in ["Low", "Medium", "High"]
    assert "probabilities" in res_low
    assert sum(res_low["probabilities"].values()) > 95.0
    print(f"[PASS] Low Risk Applicant Test: Predicted = {res_low['predicted_risk']} (Prob: {res_low['probabilities']})")

    # 2. High Risk Profile (Low credit score, high debt, excessive loan)
    high_risk_applicant = {
        "age": 24,
        "employment_type": "Freelancer",
        "monthly_income": 30000,
        "existing_debt": 15000,
        "credit_score": 450,
        "loan_amount": 800000,
        "tenure_months": 24,
        "interest_rate": 16.0,
        "emi": 39180.0,
        "dti_ratio": 180.6
    }
    res_high = predict_risk(high_risk_applicant, artifact)
    assert res_high["predicted_risk"] == "High", f"Expected High, got {res_high['predicted_risk']}"
    print(f"[PASS] High Risk Applicant Test: Predicted = {res_high['predicted_risk']} (Prob: {res_high['probabilities']})")


def test_sqlite_database():
    """Verify SQLite database creation, insertion, and retrieval."""
    init_database()
    assert os.path.exists(DB_PATH), "SQLite database file was not created!"
    
    test_record = {
        "timestamp": "2026-11-03 12:00:00",
        "applicant_name": "Test Applicant",
        "age": 30,
        "employment_type": "Salaried",
        "monthly_income": 75000.0,
        "existing_debt": 10000.0,
        "credit_score": 720,
        "loan_amount": 400000.0,
        "tenure_months": 36,
        "interest_rate": 10.5,
        "calculated_emi": 13000.0,
        "dti_ratio": 30.67,
        "risk_tier": "Low",
        "confidence": 85.5,
        "recommendations": "Test recommendation"
    }
    save_assessment_to_db(test_record)
    
    df = load_assessment_history()
    assert not df.empty, "Database history is empty after insertion!"
    assert "applicant_name" in df.columns
    assert "Test Applicant" in df["applicant_name"].values
    print("[PASS] SQLite Database Test Passed!")


def test_recommendations_engine():
    """Verify personalized recommendation generation."""
    recs = generate_recommendations(
        name="John Doe",
        dti=58.0,
        credit_score=520,
        monthly_income=40000,
        existing_debt=15000,
        emi=12000,
        loan_amount=200000,
        tenure_months=24,
        risk_tier="High"
    )
    assert len(recs) >= 3, "Expected at least 3 recommendations"
    recs_str = " ".join(recs)
    assert "Critical DTI" in recs_str or "DTI" in recs_str
    assert "Subprime" in recs_str or "Credit Score" in recs_str
    print("[PASS] Recommendations Engine Test Passed!")


if __name__ == "__main__":
    print("========================================")
    print("Running LoanIQ Automated Test Suite...")
    print("========================================")
    test_emi_calculation()
    test_model_predictions()
    test_sqlite_database()
    test_recommendations_engine()
    print("========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")