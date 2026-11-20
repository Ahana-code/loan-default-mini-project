import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import *

# Objective 1: explore the raw data
raw = pd.read_csv(DATA_RAW, low_memory=False, nrows=5000)
print("Raw data shape (sample):", raw.shape)
print("Missing values (top 10 columns):")
print(raw.isnull().sum().sort_values(ascending=False).head(10))


def load_data():
    # Read only the columns we need
    df = pd.read_csv(DATA_RAW, low_memory=False,
                     usecols=APPLICATION_FEATURES + ["loan_status"])
    # Keep only loans with a known final outcome
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    # Target: 1 = Charged Off (default), 0 = Fully Paid
    df["loan_default"] = (df["loan_status"] == "Charged Off").astype(int)
    if SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return df
    print("Loan status distribution:")
    print(df["loan_status"].value_counts())


def split_data(df):
    X = df[APPLICATION_FEATURES].copy()
    y = df["loan_default"]
    # Stratified split keeps the Charged Off share the same in both sets
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


def clean_and_encode(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Fit on training data only, then apply the same rules to test data
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    train_cat = encoder.fit_transform(X_train[cat_cols])
    test_cat = encoder.transform(X_test[cat_cols])
    columns = encoder.get_feature_names_out(cat_cols)

    X_train_final = pd.concat([
        pd.DataFrame(scaler.fit_transform(X_train[num_cols]),
                     columns=num_cols, index=X_train.index),
        pd.DataFrame(train_cat, columns=columns, index=X_train.index)
    ], axis=1)

    X_test_final = pd.concat([
        pd.DataFrame(scaler.transform(X_test[num_cols]),
                     columns=num_cols, index=X_test.index),
        pd.DataFrame(test_cat, columns=columns, index=X_test.index)
    ], axis=1)

    preprocessors = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
        "scaler": scaler,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    return X_train_final, X_test_final, preprocessors


def quality_check(X_train, X_test, y_train, y_test):
    checks = {
        "No missing values": X_train.isnull().sum().sum() == 0 and X_test.isnull().sum().sum() == 0,
        "Columns match": X_train.columns.equals(X_test.columns),
        "Split about 80/20": 15 <= len(X_test) / (len(X_train) + len(X_test)) * 100 <= 25,
        "Rows match labels": len(X_train) == len(y_train) and len(X_test) == len(y_test),
        "Class balance similar": abs(y_train.mean() - y_test.mean()) * 100 <= 1.0,
    }
    for name, ok in checks.items():
        print(f"{name:25s}: {'PASS' if ok else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("Quality checks failed. Fix preprocessing before continuing.")

if __name__ == "__main__":
    df = load_data()
    print("Rows after filtering:", len(df))

    print("\nLoan status distribution:")
    print(df["loan_status"].value_counts())

    print("\nMissing values in the 18 input features:")
    print(df[APPLICATION_FEATURES].isnull().sum())

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_final, X_test_final, preprocessors = clean_and_encode(X_train, X_test)
    quality_check(X_train_final, X_test_final, y_train, y_test)

    X_train_final.to_csv(PROCESSED / "X_train.csv", index=False)
    X_test_final.to_csv(PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED / "y_test.csv", index=False)

    joblib.dump(preprocessors, RESULTS / "preprocessors.pkl")
    print("Saved processed data and preprocessors.")