import json
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from config import PROCESSED, RESULTS

y_train = pd.read_csv(PROCESSED / "y_train.csv").iloc[:, 0]

counts = y_train.value_counts().sort_index()          # 0 = Fully Paid, 1 = Charged Off
share = y_train.value_counts(normalize=True).sort_index() * 100
weights = compute_class_weight(class_weight="balanced",
                               classes=counts.index.values, y=y_train)

summary = pd.DataFrame({
    "class": ["Fully Paid (0)", "Charged Off (1)"],
    "count": counts.values,
    "share_%": share.values.round(2),
    "balanced_weight": weights.round(3),
})
print(summary.to_string(index=False))

summary.to_csv(RESULTS / "balance_summary.csv", index=False)
print("\nReading: Charged Off loans are rare, so class_weight='balanced'")
print("gives them a larger weight during training. This makes the model")
print("pay attention to defaulters instead of ignoring them.")