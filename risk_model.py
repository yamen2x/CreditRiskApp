import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier 

# --- 1. Load data ---
data = pd.read_csv('data/Loan_Data.csv')  

# --- 2. Feature engineering ---
data['payment_to_income'] = data['loan_amt_outstanding'] / data['income']
data['debt_to_income'] = data['total_debt_outstanding'] / data['income']

# --- 3. Feature list ---
features = [
    'credit_lines_outstanding',
    'debt_to_income',
    'payment_to_income',
    'years_employed',
    'fico_score'
]

# --- 4. Split data ---
X = data[features]
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Train XGBoost model ---
clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
clf.fit(X_train, y_train)

# --- 6. Evaluate model ---
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
accuracy = metrics.accuracy_score(y_test, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.auc(fpr, tpr)

print(f"Test accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")

# --- 7. Expected loss function ---
def expected_loss(model, feature_list, loan_features, loan_amount, recovery_rate=0.1):
    """
    Calculate expected loss for a loan.
    """
    X_new = pd.DataFrame([loan_features])[feature_list]
    pd_est = model.predict_proba(X_new)[0, 1]  # probability of default
    lgd = 1 - recovery_rate
    exp_loss = pd_est * lgd * loan_amount
    print(f"Probability of default: {pd_est:.3f}, Expected loss: {exp_loss:.2f}")
    return exp_loss

# --- 8. Example loan prediction ---
loan_features = {
    'credit_lines_outstanding': 5,
    'debt_to_income': 0.4,
    'payment_to_income': 0.2,
    'years_employed': 3,
    'fico_score': 700
}
loan_amount = 15000

expected_loss(clf, features, loan_features, loan_amount)

# --- 9. Plot & save XGBoost feature importance ---
import os
import matplotlib.pyplot as plt
from xgboost import plot_importance

# ensure a figures folder exists
os.makedirs("figures", exist_ok=True)

# plot and save
plt.figure(figsize=(8, 6))
plot_importance(clf, xlabel='Gain', height=0.6)   # clf is your XGBClassifier
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig("figures/xgb_feature_importance.png")  # saved here
plt.show()

import joblib
joblib.dump(clf, "xgb_model.pkl")