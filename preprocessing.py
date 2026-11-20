import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import *


def load_data():
    # Read only the columns we need; this also avoids loading empty columns
    df = pd.read_csv(DATA_RAW, low_memory=False,
                     usecols=APPLICATION_FEATURES + ["loan_status"])
    # Keep only loans with a known final outcome
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    # Target: 1 = Charged Off (default), 0 = Fully Paid
    df["loan_default"] = (df["loan_status"] == "Charged Off").astype(int)
    if SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return df


def split_data(df):
    X = df[APPLICATION_FEATURES]
    y = df["loan_default"]
    # Stratified split keeps the Charged Off share the same in both sets
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


def clean_and_encode(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    # Fill blanks using TRAINING statistics only
    num_imp = SimpleImputer(strategy="median").fit(X_train[num_cols])
    cat_imp = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols])
    X_train[num_cols] = num_imp.transform(X_train[num_cols])
    X_test[num_cols] = num_imp.transform(X_test[num_cols])
    X_train[cat_cols] = cat_imp.transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imp.transform(X_test[cat_cols])

    # Encoder and scaler also learn from TRAINING data only
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(X_train[cat_cols])
    scaler = StandardScaler().fit(X_train[num_cols])

    def transform(X_part):
        cat = pd.DataFrame(enc.transform(X_part[cat_cols]),
                           index=X_part.index,
                           columns=enc.get_feature_names_out(cat_cols))
        num = pd.DataFrame(scaler.transform(X_part[num_cols]),
                           index=X_part.index, columns=num_cols)
        return pd.concat([num, cat], axis=1)

    return transform(X_train), transform(X_test)


def quality_check(X_train, X_test, y_train, y_test):
    checks = {
        "No blanks": X_train.isnull().sum().sum() == 0 and X_test.isnull().sum().sum() == 0,
        "Columns match": list(X_train.columns) == list(X_test.columns),
        "Split about 80/20": 15 <= len(X_test) / (len(X_train) + len(X_test)) * 100 <= 25,
        "Class balance similar": abs(y_train.mean() - y_test.mean()) * 100 <= 1.0,
    }
    for name, ok in checks.items():
        print(f"{name:25s}: {'PASS' if ok else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit("Quality checks failed. Fix preprocessing before continuing.")


if __name__ == "__main__":
    df = load_data()
    print("Rows after filtering:", len(df))
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_final, X_test_final = clean_and_encode(X_train, X_test)
    quality_check(X_train_final, X_test_final, y_train, y_test)

    # Save so the next files can load them without repeating this step
    X_train_final.to_csv(PROCESSED / "X_train.csv", index=False)
    X_test_final.to_csv(PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED / "y_test.csv", index=False)
    print("Saved processed data to data/processed/")