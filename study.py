import pandas as pd
from config import DATA_RAW, APPLICATION_FEATURES

df = pd.read_csv(DATA_RAW, low_memory=False,
                 usecols=APPLICATION_FEATURES + ["loan_status"])
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

print(df[APPLICATION_FEATURES].describe().T.round(1).to_string())

for col in df[APPLICATION_FEATURES].select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())