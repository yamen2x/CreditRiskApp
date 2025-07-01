# 💳 Credit Risk Estimator with XGBoost
This Streamlit application estimates **credit default risk** using a trained XGBoost classifier. Users can evaluate risk for **individual borrowers** or run predictions on **bulk loan data via CSV upload**. The app also includes insightful **FICO score visualizations** using both K-Means and Log-Likelihood bucketing.
## 🚀 Features
- **Single Borrower Risk Estimation**
  - Input borrower details (FICO score, income, employment, etc.)
  - Predict:
    - 📉 Probability of Default (PD)
    - 💸 Expected Loss (EL)
  - Option to save each prediction to `saved_results.csv`
- **Batch Prediction via CSV**
  - Upload borrower datasets
  - Auto-computes features like DTI and PTI
  - Predicts PD and EL
  - Visualizes distribution of risk
  - Download full results as CSV
- **FICO Score Visual Analysis**
  - 📊 K-Means Buckets
  - 📈 Log-Likelihood Buckets
  - 📉 Full FICO Distribution
## ⚙️ Model Details
- **Model**: `XGBClassifier` (XGBoost)
- **Trained on**: Realistic synthetic loan data
- **Features Used**:
  - `credit_lines_outstanding`
  - `debt_to_income`
  - `payment_to_income`
  - `years_employed`
  - `fico_score`
- **Target**:
  - `default` (binary classification)
## ▶️ Run Locally
```bash
git clone https://github.com/your-username/CreditRiskApp.git
cd CreditRiskApp
pip install -r requirements.txt
streamlit run app.py
## 📄 License
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software with attribution.
