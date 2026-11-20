import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score, precision_score, accuracy_score
from config import PROCESSED, RESULTS, RANDOM_STATE, TARGET_RECALL

X_train = pd.read_csv(PROCESSED / "X_train.csv")
y_train = pd.read_csv(PROCESSED / "y_train.csv").iloc[:, 0]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 1. Tune C and class_weight, scored by ROC-AUC (training data only)
param_grid = {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid,
                    scoring="roc_auc", cv=cv, n_jobs=1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best settings:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))

# 2. Choose the threshold using out-of-fold predictions (still training data only)
oof_proba = cross_val_predict(best_model, X_train, y_train,
                              cv=cv, method="predict_proba")[:, 1]

rows = []
for t in np.arange(0.05, 0.96, 0.05):
    pred = (oof_proba >= t).astype(int)
    rows.append({
        "threshold": round(t, 2),
        "recall": recall_score(y_train, pred, zero_division=0),
        "precision": precision_score(y_train, pred, zero_division=0),
        "accuracy": accuracy_score(y_train, pred),
    })
thr_table = pd.DataFrame(rows)
thr_table.to_csv(RESULTS / "threshold_table.csv", index=False)

eligible = thr_table[thr_table["recall"] >= TARGET_RECALL]
chosen_threshold = float(eligible["threshold"].max()) if not eligible.empty else 0.5
print(f"Chosen threshold (target recall {TARGET_RECALL}):", chosen_threshold)

# 3. Save the model and its settings for the evaluation file
joblib.dump(best_model, RESULTS / "best_model.pkl")
with open(RESULTS / "best_settings.json", "w") as f:
    json.dump({"params": grid.best_params_,
               "cv_roc_auc": round(grid.best_score_, 4),
               "threshold": chosen_threshold}, f, indent=2)
print("Saved model and settings to results/")