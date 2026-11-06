"""
=============================================================================
LoanIQ - Intelligent Credit Assessment & Risk Advisory System
Streamlit Application (app.py)
AI for Finance CIA-2 College-Level Project
=============================================================================
This application provides a comprehensive banking-grade interface for:
1. Evaluating loan affordability using standard financial mathematics (EMI, DTI).
2. Predicting credit risk (Low, Medium, High) using a trained Random Forest model.
3. Providing personalized financial recommendations.
4. Logging assessment records to an SQLite database.
5. Exporting formal loan evaluation reports.
"""

import os
import sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import ML model inference and training functions from model.py
from model import (
    calculate_emi_raw,
    load_or_train_model,
    predict_risk,
    DATASET_PATH,
    MODEL_PATH
)

# -----------------------------------------------------------------------------
# 1. SQLITE DATABASE SETUP & HELPER FUNCTIONS
# -----------------------------------------------------------------------------
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "loan_history.db")

def init_database():
    """
    Initializes the SQLite database table to store applicant assessments.
    Runs automatically on startup.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            applicant_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            employment_type TEXT NOT NULL,
            monthly_income REAL NOT NULL,
            existing_debt REAL NOT NULL,
            credit_score INTEGER NOT NULL,
            loan_amount REAL NOT NULL,
            tenure_months INTEGER NOT NULL,
            interest_rate REAL NOT NULL,
            calculated_emi REAL NOT NULL,
            dti_ratio REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            confidence REAL NOT NULL,
            recommendations TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_assessment_to_db(data: dict):
    """
    Saves an assessment dictionary into the SQLite database.
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO assessments (
            timestamp, applicant_name, age, employment_type,
            monthly_income, existing_debt, credit_score,
            loan_amount, tenure_months, interest_rate,
            calculated_emi, dti_ratio, risk_tier, confidence, recommendations
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["timestamp"],
        data["applicant_name"],
        data["age"],
        data["employment_type"],
        data["monthly_income"],
        data["existing_debt"],
        data["credit_score"],
        data["loan_amount"],
        data["tenure_months"],
        data["interest_rate"],
        data["calculated_emi"],
        data["dti_ratio"],
        data["risk_tier"],
        data["confidence"],
        data["recommendations"]
    ))
    conn.commit()
    conn.close()

def load_assessment_history() -> pd.DataFrame:
    """
    Loads all previous assessments from SQLite as a pandas DataFrame.
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM assessments ORDER BY id DESC", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


# -----------------------------------------------------------------------------
# 2. FINANCIAL RECOMMENDATION GENERATOR
# -----------------------------------------------------------------------------
def generate_recommendations(
    name: str,
    dti: float,
    credit_score: int,
    monthly_income: float,
    existing_debt: float,
    emi: float,
    loan_amount: float,
    tenure_months: int,
    risk_tier: str
) -> list:
    """
    Generates rule-based personalized financial recommendations.
    Provides actionable insights for both the applicant and the credit underwriter.
    """
    recs = []
    total_debt = existing_debt + emi
    disposable = monthly_income - total_debt
    
    # 1. DTI Ratio Guidance
    if dti <= 35.0:
        recs.append("🟢 **Healthy Debt-to-Income (DTI <= 35%)**: The applicant's debt burden is conservative, leaving substantial cash buffer for unexpected emergencies.")
    elif 35.0 < dti <= 45.0:
        recs.append(f"🟡 **Moderate DTI ({dti:.1f}%)**: Within standard banking tolerance (up to 43-45%), but discretionary spending should be monitored.")
    elif 45.0 < dti <= 55.0:
        longer_tenure = min(tenure_months + 24, 240)
        suggested_emi = calculate_emi_raw(loan_amount, 10.5, longer_tenure)
        recs.append(f"🟠 **Elevated DTI ({dti:.1f}%)**: Near lenders' maximum ceiling. Extending tenure from {tenure_months} to {longer_tenure} months would reduce monthly EMI to ~₹{suggested_emi:,.0f}, lowering DTI.")
    else:
        recs.append(f"🔴 **Critical DTI ({dti:.1f}%)**: More than half of monthly income is committed to debt service. High risk of default unless existing debts (₹{existing_debt:,.0f}) are paid off first.")
        
    # 2. Credit Score Guidance
    if credit_score >= 750:
        recs.append(f"🟢 **Prime Credit Score ({credit_score})**: Qualifies for the bank's lowest interest rate brackets and accelerated approval processing.")
    elif 650 <= credit_score < 750:
        recs.append(f"🟡 **Fair Credit Score ({credit_score})**: Standard interest rates apply. Reducing credit card utilization below 30% can help reach the 750+ prime tier within 3-6 months.")
    else:
        recs.append(f"🔴 **Subprime Credit Score ({credit_score})**: High historical delinquency risk. We strongly recommend adding a creditworthy co-applicant or pledging collateral.")
        
    # 3. Cash Flow / Disposable Income Guidance
    if disposable < 15000:
        recs.append(f"⚠️ **Tight Cash Flow Margin**: After paying existing debt and proposed EMI, only ₹{disposable:,.0f} remains for living expenses.")
    else:
        recs.append(f"💼 **Adequate Net Liquidity**: ₹{disposable:,.0f} net surplus remains each month after fulfilling all debt payments.")
        
    # 4. Loan Underwriting Decision Summary
    if risk_tier == "Low":
        recs.append("✅ **Underwriting Verdict**: Recommended for immediate loan sanction with standard documentation.")
    elif risk_tier == "Medium":
        recs.append("⚠️ **Underwriting Verdict**: Conditional approval recommended. Request bank statements for last 6 months and proof of emergency savings.")
    else:
        recs.append("⛔ **Underwriting Verdict**: High lending risk. Advise applicant to reduce requested loan principal or restructure existing obligations before re-applying.")
        
    return recs


# -----------------------------------------------------------------------------
# 3. STREAMLIT APPLICATION USER INTERFACE
# -----------------------------------------------------------------------------
def main():
    # Page setup
    st.set_page_config(
        page_title="LoanIQ | Credit Assessment & Risk Advisory",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Professional Banking Interface Styling (Navy Blue & Crisp White)
    st.markdown("""
    <style>
        .stApp {
            background-color: #F8FAFC;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .brand-banner {
            background: linear-gradient(135deg, #0A2540 0%, #1A365D 100%);
            padding: 24px 28px;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(10, 37, 64, 0.12);
        }
        .brand-banner h1 {
            color: #FFFFFF;
            font-size: 28px;
            font-weight: 700;
            margin: 0 0 6px 0;
            letter-spacing: -0.5px;
        }
        .brand-banner p {
            color: #CBD5E1;
            font-size: 14px;
            margin: 0;
        }
        .content-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        .risk-badge-low {
            display: inline-block;
            background-color: #DEF7EC;
            color: #03543F;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid #BCF0DA;
        }
        .risk-badge-medium {
            display: inline-block;
            background-color: #FEF08A;
            color: #713F12;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid #FDE047;
        }
        .risk-badge-high {
            display: inline-block;
            background-color: #FDE8E8;
            color: #9B1C1C;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid #F8B4B4;
        }
        .metric-card {
            background-color: #FFFFFF;
            border-top: 4px solid #1A365D;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            text-align: center;
        }
        .metric-card-label {
            font-size: 12px;
            text-transform: uppercase;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .metric-card-value {
            font-size: 22px;
            font-weight: 700;
            color: #0F172A;
        }
        .metric-card-sub {
            font-size: 11px;
            color: #94A3B8;
            margin-top: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize database
    init_database()

    # Application Header Banner
    st.markdown("""
    <div class="brand-banner">
        <h1>🏦 LoanIQ — Credit Assessment & Risk Advisory</h1>
        <p>AI-Powered Financial Underwriting System | AI for Finance CIA-2 Project</p>
    </div>
    """, unsafe_allow_html=True)

    # Main Navigation Tabs
    tab_assessment, tab_history, tab_model_info = st.tabs([
        "📋 New Loan Assessment",
        "📊 Historical Assessments & Analytics",
        "🧠 AI Model Architecture"
    ])

    # =========================================================================
    # TAB 1: NEW LOAN ASSESSMENT
    # =========================================================================
    with tab_assessment:
        st.markdown("### Applicant Evaluation Form")
        st.markdown("Enter applicant details to compute deterministic affordability metrics and run the Random Forest credit risk classifier.")
        
        with st.form("loan_application_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 👤 Applicant Profile")
                applicant_name = st.text_input("Full Name", value="Rahul Sharma", help="Applicant legal name")
                age = st.slider("Age (Years)", min_value=21, max_value=65, value=32, step=1)
                employment_type = st.selectbox(
                    "Employment Type",
                    options=["Salaried", "Self-Employed", "Business Owner", "Freelancer"],
                    index=0
                )
                monthly_income = st.number_input(
                    "Monthly Income (₹)",
                    min_value=10000.0,
                    max_value=2000000.0,
                    value=85000.0,
                    step=5000.0,
                    format="%.2f",
                    help="Net monthly salary or verified business earnings"
                )
                existing_debt = st.number_input(
                    "Existing Monthly Debt Obligations (₹)",
                    min_value=0.0,
                    max_value=500000.0,
                    value=12000.0,
                    step=1000.0,
                    format="%.2f",
                    help="Sum of ongoing EMIs, credit card minimums, and personal loans"
                )
                
            with col2:
                st.markdown("##### 💳 Credit & Loan Request Details")
                credit_score = st.slider(
                    "Credit Score (CIBIL / FICO)",
                    min_value=300,
                    max_value=850,
                    value=740,
                    step=5,
                    help="300-579 (Poor), 580-669 (Fair), 670-739 (Good), 740+ (Excellent)"
                )
                loan_amount = st.number_input(
                    "Requested Loan Amount (₹)",
                    min_value=25000.0,
                    max_value=10000000.0,
                    value=500000.0,
                    step=25000.0,
                    format="%.2f"
                )
                tenure_years = st.slider(
                    "Loan Tenure (Years)",
                    min_value=1,
                    max_value=20,
                    value=3,
                    step=1,
                    help="Repayment duration in years"
                )
                interest_rate = st.number_input(
                    "Annual Interest Rate (% p.a.)",
                    min_value=5.0,
                    max_value=30.0,
                    value=10.5,
                    step=0.25,
                    format="%.2f"
                )
                loan_purpose = st.selectbox(
                    "Loan Purpose",
                    options=["Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Business Expansion"],
                    index=0
                )
                
            submit_btn = st.form_submit_button("🚀 Run Comprehensive Assessment", use_container_width=True)

        if submit_btn or "last_assessment" in st.session_state:
            if submit_btn:
                tenure_months = int(tenure_years * 12)
                
                # 1. Deterministic Financial Math Calculations
                calculated_emi = calculate_emi_raw(loan_amount, interest_rate, tenure_months)
                total_repayment = round(calculated_emi * tenure_months, 2)
                total_interest = round(total_repayment - loan_amount, 2)
                total_monthly_debt = round(existing_debt + calculated_emi, 2)
                dti_ratio = round((total_monthly_debt / monthly_income) * 100.0, 2)
                net_disposable = round(monthly_income - total_monthly_debt, 2)
                loan_to_income = round(loan_amount / (monthly_income * 12.0), 2)
                
                # 2. AI Risk Prediction via Machine Learning Model
                model_artifact = load_or_train_model()
                applicant_payload = {
                    "age": age,
                    "employment_type": employment_type,
                    "monthly_income": monthly_income,
                    "existing_debt": existing_debt,
                    "credit_score": credit_score,
                    "loan_amount": loan_amount,
                    "tenure_months": tenure_months,
                    "interest_rate": interest_rate,
                    "emi": calculated_emi,
                    "dti_ratio": dti_ratio
                }
                pred_res = predict_risk(applicant_payload, model_artifact)
                risk_tier = pred_res["predicted_risk"]
                confidence = pred_res["confidence"]
                probabilities = pred_res["probabilities"]
                
                # 3. Personalized Financial Recommendations
                recommendations_list = generate_recommendations(
                    name=applicant_name,
                    dti=dti_ratio,
                    credit_score=credit_score,
                    monthly_income=monthly_income,
                    existing_debt=existing_debt,
                    emi=calculated_emi,
                    loan_amount=loan_amount,
                    tenure_months=tenure_months,
                    risk_tier=risk_tier
                )
                
                # Store in session state
                st.session_state["last_assessment"] = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "applicant_name": applicant_name,
                    "age": age,
                    "employment_type": employment_type,
                    "monthly_income": monthly_income,
                    "existing_debt": existing_debt,
                    "credit_score": credit_score,
                    "loan_amount": loan_amount,
                    "tenure_months": tenure_months,
                    "tenure_years": tenure_years,
                    "interest_rate": interest_rate,
                    "loan_purpose": loan_purpose,
                    "calculated_emi": calculated_emi,
                    "total_repayment": total_repayment,
                    "total_interest": total_interest,
                    "dti_ratio": dti_ratio,
                    "net_disposable": net_disposable,
                    "loan_to_income": loan_to_income,
                    "risk_tier": risk_tier,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "recommendations_list": recommendations_list,
                    "recommendations": "\n".join(recommendations_list)
                }
            
            res = st.session_state["last_assessment"]
            
            st.markdown("---")
            st.markdown(f"### Assessment Results for: **{res['applicant_name']}**")
            st.caption(f"Evaluated on {res['timestamp']} | Purpose: {res['loan_purpose']}")
            
            # Top KPI Metric Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">Monthly EMI</div>
                    <div class="metric-card-value">₹{res['calculated_emi']:,.0f}</div>
                    <div class="metric-card-sub">For {res['tenure_months']} months @ {res['interest_rate']}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with kpi2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">Debt-To-Income (DTI)</div>
                    <div class="metric-card-value">{res['dti_ratio']:.1f}%</div>
                    <div class="metric-card-sub">Existing + New EMI</div>
                </div>
                """, unsafe_allow_html=True)
                
            with kpi3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">Net Monthly Surplus</div>
                    <div class="metric-card-value">₹{res['net_disposable']:,.0f}</div>
                    <div class="metric-card-sub">Disposable after all debts</div>
                </div>
                """, unsafe_allow_html=True)
                
            with kpi4:
                badge_class = f"risk-badge-{res['risk_tier'].lower()}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-label">AI Risk Classification</div>
                    <div style="margin-top: 6px;">
                        <span class="{badge_class}">{res['risk_tier'].upper()} RISK</span>
                    </div>
                    <div class="metric-card-sub" style="margin-top: 6px;">Confidence: {res['confidence']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # -----------------------------------------------------------------
            # SECTION A: FINANCIAL AFFORDABILITY (Deterministic Math)
            # -----------------------------------------------------------------
            st.markdown("""
            <div class="content-card">
                <h4 style="color: #0A2540; margin-bottom: 4px;">📐 1. Financial Affordability Analysis</h4>
                <p style="color: #64748B; font-size: 13px;">Deterministic mathematical calculations based on applicant cash flow and banking benchmark ratios.</p>
            </div>
            """, unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                labels_cf = ["Net Disposable Surplus", "Existing Monthly Debt", "Proposed Loan EMI"]
                values_cf = [
                    max(res["net_disposable"], 0.0),
                    res["existing_debt"],
                    res["calculated_emi"]
                ]
                fig_cf = go.Figure(data=[go.Pie(
                    labels=labels_cf,
                    values=values_cf,
                    hole=0.55,
                    marker=dict(colors=["#10B981", "#F59E0B", "#1E3A8A"]),
                    textinfo="label+percent"
                )])
                fig_cf.update_layout(
                    title="<b>Monthly Income Allocation</b>",
                    showlegend=False,
                    margin=dict(t=40, b=20, l=20, r=20),
                    height=280
                )
                st.plotly_chart(fig_cf, use_container_width=True)
                
            with chart_col2:
                labels_rep = ["Principal Loan Amount", "Total Interest Payable"]
                values_rep = [res["loan_amount"], res["total_interest"]]
                fig_rep = go.Figure(data=[go.Pie(
                    labels=labels_rep,
                    values=values_rep,
                    hole=0.55,
                    marker=dict(colors=["#0A2540", "#60A5FA"]),
                    textinfo="label+percent"
                )])
                fig_rep.update_layout(
                    title="<b>Total Repayment Breakdown</b>",
                    showlegend=False,
                    margin=dict(t=40, b=20, l=20, r=20),
                    height=280
                )
                st.plotly_chart(fig_rep, use_container_width=True)
                
            afford_df = pd.DataFrame({
                "Financial Metric": [
                    "Gross Monthly Income",
                    "Existing Debt Burden",
                    "Proposed New EMI",
                    "Total Monthly Outflow",
                    "Remaining Monthly Surplus",
                    "Total Principal Requested",
                    "Total Interest Across Tenure",
                    "Total Lifetime Repayment",
                    "Loan-to-Annual-Income (LTI)"
                ],
                "Value": [
                    f"₹{res['monthly_income']:,.2f}",
                    f"₹{res['existing_debt']:,.2f}",
                    f"₹{res['calculated_emi']:,.2f}",
                    f"₹{res['existing_debt'] + res['calculated_emi']:,.2f}",
                    f"₹{res['net_disposable']:,.2f}",
                    f"₹{res['loan_amount']:,.2f}",
                    f"₹{res['total_interest']:,.2f}",
                    f"₹{res['total_repayment']:,.2f}",
                    f"{res['loan_to_income']:.2f}x"
                ],
                "Industry Benchmark": [
                    "—",
                    "Should ideally be < 20% of income",
                    "Comfortable below 25-30% of income",
                    "Max tolerable 45-50%",
                    "Minimum ₹15,000 safety buffer",
                    "—",
                    "Lower tenure reduces total interest",
                    "—",
                    "Healthy if < 3.5x annual salary"
                ]
            })
            st.dataframe(afford_df, use_container_width=True, hide_index=True)
            
            # -----------------------------------------------------------------
            # SECTION B: AI RISK PREDICTION (Machine Learning Engine)
            # -----------------------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="content-card">
                <h4 style="color: #0A2540; margin-bottom: 4px;">🤖 2. AI Credit Risk Classification (Random Forest Model)</h4>
                <p style="color: #64748B; font-size: 13px;">Probabilistic assessment generated by the supervised ensemble learning model trained on historical lending patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            ml_col1, ml_col2 = st.columns([1.2, 1])
            
            with ml_col1:
                prob_df = pd.DataFrame({
                    "Risk Tier": ["Low", "Medium", "High"],
                    "Probability (%)": [
                        res["probabilities"]["Low"],
                        res["probabilities"]["Medium"],
                        res["probabilities"]["High"]
                    ],
                    "Color": ["#10B981", "#F59E0B", "#EF4444"]
                })
                
                fig_prob = go.Figure(go.Bar(
                    x=prob_df["Risk Tier"],
                    y=prob_df["Probability (%)"],
                    marker_color=prob_df["Color"],
                    text=[f"{p:.1f}%" for p in prob_df["Probability (%)"]],
                    textposition="auto"
                ))
                fig_prob.update_layout(
                    title="<b>AI Risk Probability Distribution</b>",
                    yaxis=dict(range=[0, 105], title="Confidence (%)"),
                    margin=dict(t=40, b=20, l=20, r=20),
                    height=260
                )
                st.plotly_chart(fig_prob, use_container_width=True)
                
            with ml_col2:
                st.markdown("##### Model Decision Summary")
                st.markdown(f"- **Predicted Classification**: **`{res['risk_tier']} Risk`**")
                st.markdown(f"- **Prediction Confidence**: **`{res['confidence']:.1f}%`**")
                st.markdown(f"- **Applicant Credit Score**: **`{res['credit_score']}` / 850**")
                st.markdown(f"- **Computed DTI Ratio**: **`{res['dti_ratio']:.1f}%`**")
                
                if res["risk_tier"] == "Low":
                    st.success("🟢 **Low Risk Profile**: Applicant exhibits prime credit score, balanced leverage, and healthy cash-flow coverage. Excellent lending candidate.")
                elif res["risk_tier"] == "Medium":
                    st.warning("🟡 **Medium Risk Profile**: Applicant presents moderate debt exposure or borderline credit score. Additional scrutiny or collateral recommended.")
                else:
                    st.error("🔴 **High Risk Profile**: High probability of delinquency detected due to severe debt leverage, elevated DTI, or subprime credit score.")
                    
            # -----------------------------------------------------------------
            # SECTION C: PERSONALIZED RECOMMENDATIONS & ACTIONS
            # -----------------------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="content-card">
                <h4 style="color: #0A2540; margin-bottom: 4px;">💡 3. Personalized Financial Recommendations</h4>
                <p style="color: #64748B; font-size: 13px;">Tailored financial advisory insights for credit optimization and risk mitigation.</p>
            </div>
            """, unsafe_allow_html=True)
            
            for rec in res["recommendations_list"]:
                st.markdown(f"- {rec}")
                
            # -----------------------------------------------------------------
            # SECTION D: DATABASE SAVE & DOWNLOAD REPORT
            # -----------------------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("💾 Save Assessment to History (SQLite)", use_container_width=True):
                    save_assessment_to_db({
                        "timestamp": res["timestamp"],
                        "applicant_name": res["applicant_name"],
                        "age": res["age"],
                        "employment_type": res["employment_type"],
                        "monthly_income": res["monthly_income"],
                        "existing_debt": res["existing_debt"],
                        "credit_score": res["credit_score"],
                        "loan_amount": res["loan_amount"],
                        "tenure_months": res["tenure_months"],
                        "interest_rate": res["interest_rate"],
                        "calculated_emi": res["calculated_emi"],
                        "dti_ratio": res["dti_ratio"],
                        "risk_tier": res["risk_tier"],
                        "confidence": res["confidence"],
                        "recommendations": res["recommendations"]
                    })
                    st.success("✅ Assessment successfully stored in SQLite database!")
                    
            with btn_col2:
                report_text = f"""=============================================================================
LOANIQ CREDIT ASSESSMENT & RISK ADVISORY REPORT
Generated on: {res['timestamp']}
Confidential Banking Assessment Document
=============================================================================

1. APPLICANT DETAILS
-----------------------------------------------------------------------------
Applicant Name        : {res['applicant_name']}
Age                   : {res['age']} years
Employment Type       : {res['employment_type']}
Monthly Income        : Rs. {res['monthly_income']:,.2f}
Existing Monthly Debt : Rs. {res['existing_debt']:,.2f}
Credit Bureau Score   : {res['credit_score']} / 850

2. REQUESTED LOAN TERMS
-----------------------------------------------------------------------------
Loan Purpose          : {res['loan_purpose']}
Principal Loan Amount : Rs. {res['loan_amount']:,.2f}
Tenure                : {res['tenure_months']} months ({res['tenure_years']} years)
Annual Interest Rate  : {res['interest_rate']}% p.a.

3. FINANCIAL AFFORDABILITY ANALYSIS
-----------------------------------------------------------------------------
Calculated Monthly EMI: Rs. {res['calculated_emi']:,.2f}
Total Interest Payable: Rs. {res['total_interest']:,.2f}
Total Lifetime Outflow: Rs. {res['total_repayment']:,.2f}
Total Monthly Debt    : Rs. {res['existing_debt'] + res['calculated_emi']:,.2f}
Debt-to-Income (DTI)  : {res['dti_ratio']:.2f}%
Remaining Net Surplus : Rs. {res['net_disposable']:,.2f}
Loan-to-Annual-Income : {res['loan_to_income']:.2f}x

4. AI MACHINE LEARNING RISK PREDICTION
-----------------------------------------------------------------------------
Model Architecture    : Random Forest Classifier (100 Estimators)
Classified Risk Tier  : {res['risk_tier'].upper()} RISK
Model Confidence      : {res['confidence']:.1f}%
Probability Breakdown :
  - Low Risk          : {res['probabilities']['Low']:.1f}%
  - Medium Risk       : {res['probabilities']['Medium']:.1f}%
  - High Risk         : {res['probabilities']['High']:.1f}%

5. STRATEGIC FINANCIAL RECOMMENDATIONS
-----------------------------------------------------------------------------
{res['recommendations']}

=============================================================================
End of Report | LoanIQ Autonomous Credit Evaluation System
=============================================================================
"""
                filename = f"LoanIQ_Report_{res['applicant_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M')}.txt"
                st.download_button(
                    label="📄 Download Official Assessment Report (.txt)",
                    data=report_text,
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True
                )

    # =========================================================================
    # TAB 2: HISTORICAL ASSESSMENTS & ANALYTICS
    # =========================================================================
    with tab_history:
        st.markdown("### Historical Assessments Database")
        st.markdown("All applicant assessments saved into the local SQLite database (`data/loan_history.db`) are displayed below.")
        
        history_df = load_assessment_history()
        
        if history_df.empty:
            st.info("ℹ️ No saved assessments found in the database yet. Run an assessment on Tab 1 and click 'Save Assessment to History'.")
        else:
            st.markdown(f"**Total Records Stored**: `{len(history_df)}`")
            
            hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
            with hist_col1:
                st.metric("Total Applicants", len(history_df))
            with hist_col2:
                low_count = int((history_df["risk_tier"] == "Low").sum())
                st.metric("Low Risk Count", f"{low_count} ({low_count/len(history_df)*100:.0f}%)")
            with hist_col3:
                avg_dti = float(history_df["dti_ratio"].mean())
                st.metric("Average DTI", f"{avg_dti:.1f}%")
            with hist_col4:
                total_loans_val = float(history_df["loan_amount"].sum())
                st.metric("Total Requested", f"₹{total_loans_val:,.0f}")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            hist_chart1, hist_chart2 = st.columns(2)
            with hist_chart1:
                risk_counts = history_df["risk_tier"].value_counts().reset_index()
                risk_counts.columns = ["Risk Tier", "Count"]
                color_map = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
                fig_hist_risk = px.pie(
                    risk_counts,
                    names="Risk Tier",
                    values="Count",
                    title="<b>Historical Risk Tier Distribution</b>",
                    color="Risk Tier",
                    color_discrete_map=color_map,
                    hole=0.4
                )
                fig_hist_risk.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=280)
                st.plotly_chart(fig_hist_risk, use_container_width=True)
                
            with hist_chart2:
                fig_hist_scatter = px.scatter(
                    history_df,
                    x="credit_score",
                    y="dti_ratio",
                    color="risk_tier",
                    color_discrete_map=color_map,
                    hover_data=["applicant_name", "loan_amount"],
                    title="<b>Credit Score vs. DTI Ratio Distribution</b>",
                    labels={"credit_score": "Credit Score", "dti_ratio": "DTI Ratio (%)"}
                )
                fig_hist_scatter.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=280)
                st.plotly_chart(fig_hist_scatter, use_container_width=True)
                
            st.markdown("##### Assessment Records")
            display_cols = [
                "id", "timestamp", "applicant_name", "credit_score",
                "monthly_income", "loan_amount", "calculated_emi",
                "dti_ratio", "risk_tier", "confidence"
            ]
            st.dataframe(history_df[display_cols], use_container_width=True, hide_index=True)
            
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Full Assessment History (CSV)",
                data=csv_data,
                file_name=f"LoanIQ_Assessment_History_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # =========================================================================
    # TAB 3: AI MODEL ARCHITECTURE & INSIGHTS
    # =========================================================================
    with tab_model_info:
        st.markdown("### AI Model Architecture & Underwriting Insights")
        st.markdown("This section details the machine learning methodology powering LoanIQ's risk engine.")
        
        model_artifact = load_or_train_model()
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("""
            <div class="content-card">
                <h4>Algorithm: Random Forest Classifier</h4>
                <ul>
                    <li><b>Ensemble Method</b>: 100 Decision Trees with balanced class weights.</li>
                    <li><b>Target Classes</b>: 3-Tier Classification (<code>Low</code>, <code>Medium</code>, <code>High</code> Risk).</li>
                    <li><b>Input Features</b>: 11 financial metrics including Credit Score, DTI, LTI, Income, and Debt.</li>
                    <li><b>Training Strategy</b>: Stratified 80/20 train-test split on 2,500 synthetic applicant records.</li>
                    <li><b>Evaluation Metric</b>: Balanced Accuracy & Multiclass F1-Score.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric(
                label="Model Test Accuracy",
                value=f"{model_artifact.get('accuracy', 0.95) * 100:.2f}%"
            )
            
        with col_m2:
            feat_imp = model_artifact.get("feature_importances", {})
            feat_df = pd.DataFrame({
                "Feature": list(feat_imp.keys()),
                "Importance": [float(v) for v in feat_imp.values()]
            }).sort_values(by="Importance", ascending=True)
            
            fig_imp = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="<b>Random Forest Feature Importance</b>",
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig_imp.update_layout(
                margin=dict(t=40, b=20, l=20, r=20),
                height=320,
                showlegend=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Synthetic Training Dataset Overview")
        if os.path.exists(DATASET_PATH):
            sample_df = pd.read_csv(DATASET_PATH)
            st.markdown(f"Synthetic dataset saved at `data/synthetic_loans.csv` ({len(sample_df)} rows). First 5 rows:")
            st.dataframe(sample_df.head(5), use_container_width=True, hide_index=True)
        else:
            st.info("Dataset file will be generated automatically when model is trained.")


if __name__ == "__main__":
    main()