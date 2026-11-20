import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from config import PROCESSED, RESULTS

X_test = pd.read_csv(PROCESSED / "X_test.csv")
y_test = pd.read_csv(PROCESSED / "y_test.csv").iloc[:, 0]

model = joblib.load(RESULTS / "best_model.pkl")
with open(RESULTS / "best_settings.json") as f:
    settings = json.load(f)
threshold = settings["threshold"]

proba = model.predict_proba(X_test)[:, 1]

def metrics(label, t):
    pred = (proba >= t).astype(int)
    return {
        "Model": label,
        "Threshold": t,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
    }

results = pd.DataFrame([
    metrics("Tuned LR (default cut-off)", 0.50),
    metrics("Tuned LR (chosen cut-off)", threshold),
])
print(results.to_string(index=False))
results.to_csv(RESULTS / "model_results.csv", index=False)
print("\nSaved results to results/model_results.csv")