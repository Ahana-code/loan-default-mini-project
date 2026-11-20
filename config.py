from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "accepted_2007_to_2018Q4.csv"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_RECALL = 0.60          # business rule: catch at least 60% of defaulters
SAMPLE_SIZE = None            # set to e.g. 200_000 for a quick test run; None = full data

# Application-time features only (known when the loan is applied for, so no leakage)
APPLICATION_FEATURES = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "delinq_2yrs", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "application_type",
]