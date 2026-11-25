import pandas as pd
from explain import assess, explain, prep, CAT_COLS, THRESHOLD

# Allowed answers come from the training data, so they always match the model
OPTIONS = {col: list(cats) for col, cats in zip(CAT_COLS, prep["encoder"].categories_)}


def ask_number(label, low, high):
    while True:
        try:
            value = float(input(f"{label} [{low} to {high}]: "))
            if low <= value <= high:
                return value
            print("  Out of the allowed range, try again.")
        except ValueError:
            print("  Please type a number.")


def ask_choice(label, column):
    options = OPTIONS[column]
    print(f"\n{label}:")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    while True:
        choice = input("Enter the number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("  Invalid choice, try again.")


def fill_form():
    print("=== NEW LOAN APPLICATION ===")
    applicant = {
        "loan_amnt":           ask_number("Loan amount (Rs/$)", 500, 400000),
        "term":                ask_choice("Loan term", "term"),
        "emp_length":          ask_choice("Years in current job", "emp_length"),
        "home_ownership":      ask_choice("Home ownership", "home_ownership"),
        "annual_inc":          ask_number("Annual income", 0, 10000000),
        "verification_status": ask_choice("Income verification", "verification_status"),
        "purpose":             ask_choice("Purpose of loan", "purpose"),
        "dti":                 ask_number("Debt-to-income ratio (%)", 0, 100),
        "delinq_2yrs":         ask_number("Late payments in last 2 years", 0, 50),
        "fico_range_low":      ask_number("Credit score (FICO)", 625, 850),
        "inq_last_6mths":      ask_number("Credit enquiries in last 6 months", 0, 50),
        "open_acc":            ask_number("Open credit accounts", 0, 100),
        "pub_rec":             ask_number("Public records (e.g. bankruptcy)", 0, 20),
        "revol_bal":           ask_number("Credit card balance", 0, 10000000),
        "revol_util":          ask_number("Credit card limit used (%)", 0, 150),
        "total_acc":           ask_number("Total credit accounts ever", 0, 200),
        "application_type":    ask_choice("Application type", "application_type"),
    }
    # FICO range is always a 5-point band, so the high end is derived
    applicant["fico_range_high"] = applicant["fico_range_low"] + 4
    return applicant


def save_record(applicant, proba):
    decision = "FLAG" if proba >= THRESHOLD else "LOW"
    record = {**applicant, "risk_probability": round(proba, 4), "decision_support": decision}
    pd.DataFrame([record]).to_csv("applications_log.csv", mode="a",
                                  header=False, index=False)
    print("\nApplication saved to applications_log.csv")


if __name__ == "__main__":
    applicant = fill_form()
    assess(applicant)
    proba, _ = explain(applicant)
    save_record(applicant, proba)