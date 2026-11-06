# 🏦 LoanIQ — Intelligent Credit Assessment & Risk Advisory System

**Course Project:** AI for Finance (CIA-2)  
**Academic Level:** Undergraduate / College-Level  
**Domain:** Fintech / Autonomous Credit Underwriting & Risk Analytics  

---

## 📌 Executive Summary

**LoanIQ** is a financial technology application designed to evaluate loan applications through two decoupled, complementary analytical pipelines:
1. **Deterministic Financial Affordability Analysis:** Uses exact banking mathematics to calculate the Equated Monthly Installment (EMI), Debt-to-Income (DTI) ratio, total interest payable, and net disposable monthly surplus.
2. **AI-Powered Credit Risk Classification:** Uses a supervised **Random Forest Machine Learning model** trained on synthetic lending data to classify applicants into **Low**, **Medium**, or **High** credit risk tiers along with calibrated confidence scores.

The system also provides **personalized financial recommendations**, records assessment transactions into a persistent **SQLite database**, and enables users to **download formal assessment reports**.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Interactive web dashboard with a clean Navy Blue banking theme |
| **Data Visualizations** | [Plotly](https://plotly.com/python/) | Interactive charts (Donut charts, Bar charts, Scatter plots) |
| **Machine Learning** | [Scikit-learn](https://scikit-learn.org/) | Random Forest Classifier, model evaluation & stratification |
| **Numerical Processing** | [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) | Vectorized financial computations and data frame manipulation |
| **Persistence** | [SQLite3](https://docs.python.org/3/library/sqlite3.html) | Local lightweight relational database for assessment history |
| **Model Serialization** | [Joblib](https://joblib.readthedocs.io/) | Model persistence and caching (`loan_model.pkl`) |

---

## 🚀 Key Features

1. **Applicant Details Form:** Captures legal applicant name, age, employment type, monthly income, and existing monthly debt.
2. **Credit & Loan Terms:** Captures credit bureau score (300–850), loan amount, tenure (years/months), annual interest rate, and loan purpose.
3. **Deterministic EMI Calculation:** Implements the standard banking reducing-balance formula:
   $$\text{EMI} = \frac{P \times r \times (1 + r)^n}{(1 + r)^n - 1}$$
4. **Debt-to-Income (DTI) Analysis:** Computes the applicant's total debt obligations relative to gross monthly income:
   $$\text{DTI} = \left(\frac{\text{Existing Debt} + \text{Proposed EMI}}{\text{Monthly Income}}\right) \times 100\%$$
5. **Separation of Concerns:** AI predictions and financial affordability calculations are distinctly segregated in both code and user interface.
6. **Random Forest ML Risk Model:** Classifies applicants into **Low**, **Medium**, or **High** risk with a multiclass probability breakdown.
7. **Personalized Financial Recommendations:** Rule-based advisory engine offering customized suggestions for DTI reduction, credit score improvement, and loan structuring.
8. **Persistent SQLite Audit Trail:** Automatically logs all evaluated loan applications into `data/loan_history.db`.
9. **Interactive Analytics:** Historical distributions of credit scores, DTI ratios, and risk ratings visualized with Plotly.
10. **Report Export:** Instant download of formal assessment reports (`.txt`) and historical logs (`.csv`).

---

## 📁 Project Directory Structure

```text
LoanFRESH/
├── app.py                  # Main Streamlit web application & UI logic
├── model.py                # Synthetic data generation, ML training & prediction
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation & academic report
└── data/
    ├── synthetic_loans.csv # Synthetic training dataset (2,500 samples)
    ├── loan_model.pkl      # Serialized Random Forest model artifact
    └── loan_history.db     # SQLite database storing loan assessment history
```

---

## 🧮 Mathematical & Financial Foundations

### 1. Equated Monthly Installment (EMI)
$$\text{EMI} = \frac{P \times r \times (1 + r)^n}{(1 + r)^n - 1}$$
Where:
- $P$ = Principal loan amount
- $r$ = Monthly interest rate ($\text{Annual Rate} / 12 / 100$)
- $n$ = Total loan tenure in months

### 2. Debt-to-Income (DTI) Ratio
$$\text{DTI} = \left(\frac{\text{Existing Debt} + \text{New EMI}}{\text{Monthly Income}}\right) \times 100$$
- **DTI $\le$ 35%:** Healthy — Strong buffer against financial shocks.
- **36% to 45%:** Moderate — Standard institutional lending tolerance.
- **46% to 55%:** Elevated — Stretched cash flow; tenure extension advised.
- **DTI > 55%:** Critical — Severe over-leverage; high probability of default.

### 3. Net Monthly Surplus
$$\text{Net Disposable Surplus} = \text{Monthly Income} - (\text{Existing Debt} + \text{New EMI})$$

---

## 🤖 Machine Learning Architecture (`model.py`)

- **Algorithm:** Random Forest Classifier (`n_estimators=100`, `max_depth=12`, `class_weight='balanced'`)
- **Dataset:** 2,500 synthetic applicant records generated using realistic banking distributions (log-normal income, beta credit score, empirical debt-to-income limits).
- **Features (11):**
  1. `age`
  2. `employment_encoded`
  3. `monthly_income`
  4. `existing_debt`
  5. `credit_score`
  6. `loan_amount`
  7. `tenure_months`
  8. `interest_rate`
  9. `emi`
  10. `dti_ratio`
  11. `loan_to_income_ratio`
- **Target Variable:** `risk_tier` (`Low`, `Medium`, `High`)
- **Validation Accuracy:** $\approx$ 95.6% on 20% stratified test split.

---

## 💻 Installation & Setup Instructions

### Step 1: Clone or Open the Workspace
Ensure your command prompt is located in the project directory:
```bash
cd c:\Users\PRAJEET\Downloads\LoanFRESH
```

### Step 2: Install Required Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
Launch the LoanIQ interface:
```bash
streamlit run app.py
```
Streamlit will automatically open your default browser at `http://localhost:8501`.

---

## 📖 Application Walkthrough

### 1. New Loan Assessment Tab
- Fill in the applicant's profile (Income, Debt, Credit Score, Loan Amount, Tenure, Purpose).
- Click **"Run Comprehensive Assessment"**.
- View top-level KPI metrics: Monthly EMI, DTI Ratio, Net Surplus, and AI Risk Tier.
- Review **Section 1: Financial Affordability Analysis** (income allocation & repayment breakdown charts).
- Review **Section 2: AI Credit Risk Classification** (Random Forest probability distribution).
- Review **Section 3: Personalized Financial Recommendations** (actionable advisory points).
- Click **"Save Assessment to History"** to record into SQLite.
- Click **"Download Official Assessment Report"** to export the text summary.

### 2. Historical Assessments & Analytics Tab
- View a table of all past loan evaluations stored in `loan_history.db`.
- Inspect aggregated portfolio KPIs (Total Applicants, Low Risk %, Average DTI, Total Capital Requested).
- Explore Plotly scatter charts relating credit scores and DTI ratios.
- Click **"Export Full Assessment History (CSV)"** to download the database.

### 3. AI Model Architecture Tab
- Inspect model hyperparameters, test accuracy, and feature importance rankings.
- Preview the first 5 records of the synthetic training dataset.

---

## 🎓 Academic Integrity & Disclaimer
This project is developed exclusively for academic evaluation under the **AI for Finance (CIA-2)** coursework. The machine learning model is trained on synthetic data for educational demonstration of underwriting heuristics and should not be used as the sole determinant for real-world commercial lending without institutional calibration.

