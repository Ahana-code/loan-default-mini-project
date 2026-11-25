import json
import joblib
import pandas as pd
from config import RESULTS, APPLICATION_FEATURES

model = joblib.load(RESULTS / "best_model.pkl")
prep = joblib.load(RESULTS / "preprocessors.pkl")
with open(RESULTS / "best_settings.json") as f:
    THRESHOLD = json.load(f)["threshold"]

NUM_COLS = prep["num_cols"]
CAT_COLS = prep["cat_cols"]

# ---------- INPUT: the applicant form (fill this in) ----------
applicant = {
    "loan_amnt": 10000,
    "term": " 36 months",              # copy exact text from study_data.py
    "emp_length": "5 years",           # copy exact text from study_data.py
    "home_ownership": "RENT",
    "annual_inc": 60000,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 18.5,
    "delinq_2yrs": 0,
    "fico_range_low": 700,
    "fico_range_high": 704,            # usually fico_range_low + 4
    "inq_last_6mths": 1,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 8000,
    "revol_util": 45.0,
    "total_acc": 25,
    "application_type": "Individual",
}


def to_model_input(applicant):
    # Same steps as training: fill blanks, encode text, scale numbers
    row = pd.DataFrame([applicant])[APPLICATION_FEATURES]
    row[NUM_COLS] = prep["num_imputer"].transform(row[NUM_COLS])
    row[CAT_COLS] = prep["cat_imputer"].transform(row[CAT_COLS])
    cat = pd.DataFrame(prep["encoder"].transform(row[CAT_COLS]),
                       columns=prep["encoder"].get_feature_names_out(CAT_COLS))
    num = pd.DataFrame(prep["scaler"].transform(row[NUM_COLS]), columns=NUM_COLS)
    return pd.concat([num, cat], axis=1)[model.feature_names_in_]


def explain(applicant):
    X = to_model_input(applicant)
    proba = model.predict_proba(X)[0, 1]

    # Logistic Regression is additive: each feature adds coef * value to the score
    contrib = pd.Series(X.iloc[0].values * model.coef_[0], index=X.columns)

    # Add up one-hot columns (e.g. home_ownership_RENT) back to the original feature
    grouped = {}
    for col, val in contrib.items():
        orig = next((c for c in CAT_COLS if col.startswith(c + "_")), col)
        grouped[orig] = grouped.get(orig, 0) + val
    return proba, pd.Series(grouped).sort_values()


def assess(applicant):
    proba, contrib = explain(applicant)
    decision = "FLAG FOR REVIEW (higher risk)" if proba >= THRESHOLD else "LOW RISK (eligible)"
    if applicant["fico_range_low"] < 625:
        print("WARNING: FICO below 625 is rare in the training data. Verify this value.")

    print("=" * 55)
    print(f"Default risk probability: {proba:.1%}")
    print(f"Decision threshold:       {THRESHOLD:.0%}")
    print(f"Decision:                 {decision}")
    print("=" * 55)

    print("\nFactors pushing risk UP:")
    for name, val in contrib[contrib > 0].sort_values(ascending=False).head(3).items():
        print(f"  + {name:22s} (impact {val:+.2f})")

    print("\nFactors pushing risk DOWN:")
    for name, val in contrib[contrib < 0].sort_values().head(3).items():
        print(f"  - {name:22s} (impact {val:+.2f})")

    print("\nNote: impacts are in model units. Bigger absolute value = bigger influence.")
    print("Final lending decisions need human review and bank policy checks.")


if __name__ == "__main__":
    assess(applicant)
    DOCS = {
    "home_ownership": "Rent agreement or property papers",
    "verification_status": "Income verification record",
    "term": "Signed loan agreement",
    "purpose": "Purpose declaration",
    "annual_inc": "Salary slips / ITR",
    "fico_range_low": "Credit bureau report",
    "dti": "Existing loan statements",
    "emp_length": "Employment letter",
}