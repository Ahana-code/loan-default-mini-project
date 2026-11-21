# Loan Default Risk Prediction (Explainable)

A mini project that predicts whether a loan will be **Charged Off** (default) or **Fully Paid**, using Lending Club accepted-loan data (2007 to 2018 Q4). The model is Logistic Regression, chosen because each feature's effect on the risk score can be explained.

## Objectives

1. **Explore the dataset:** structure, missing values, and loan status distribution.
2. **Prepare a binary classification dataset:** Fully Paid and Charged Off records, with 18 application-time input features.
3. **Preprocess with a stratified split:** train/test split, training-based imputation, categorical encoding, and numerical scaling.
4. **Explain each decision:** show the risk probability and the factors that pushed it up or down.

## Project Structure

```
mini project/
├── data/                       # raw CSV (not included in repo)
├── results/                    # created by the scripts
├── config.py                   # shared settings and the 18 feature names
├── preprocessing.py            # explore, split, impute, encode, scale, save
├── balancing.py                # class imbalance summary and weights
├── training_tuning.py          # Logistic Regression, CV tuning, threshold
├── evaluation.py               # final test-set results (run once)
├── explain.py                  # scores one applicant and explains the result
├── interactive_entry.py        # terminal form for entering a new applicant
├── requirements.txt
└── README.md
```

## Setup

1. Download `accepted_2007_to_2018Q4.csv` from Lending Club (Kaggle also hosts it) and place it in the `data/` folder.
2. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Run these in order from the project folder:

```bash
python preprocessing.py
python balancing.py
python training_tuning.py
python evaluation.py
```

To score and explain one applicant:

```bash
python explain.py
```

To enter a new applicant through the terminal form:

```bash
python interactive_entry.py
```

Each script reads the output of the previous one, so run them in this order the first time.

## Results (test set)

| Model | Accuracy | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Tuned Logistic Regression (cut-off 0.50) | 0.642 | 0.632 | 0.307 | 0.692 |

- Best settings: `C = 10`, `class_weight = 'balanced'`
- Decision threshold chosen on training data to reach at least 60% recall.

## Explainability

- Logistic Regression adds up each feature's effect, so each decision can be traced to specific factors.
- `explain.py` lists the top factors pushing risk up and down.
- Impact values are in model units. Use them to rank factors, not as percentages.

## Key Points

- About 80% of loans are Fully Paid, so accuracy alone is misleading. Recall and ROC-AUC are used to judge the model.
- The test set is used only once, in `evaluation.py`. All tuning happens on training data.
- The model supports a human decision. It does not replace bank policy or KYC checks.

## Future Work

- Gradient boosting models for higher ROC-AUC
- Feature engineering (for example, loan amount relative to income)
- Dashboard for exploring results
- Web interface for applicant entry