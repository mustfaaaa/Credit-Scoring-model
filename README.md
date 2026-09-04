# Credit Scoring Model — Predicting Creditworthiness

**Task 1 — Machine Learning Internship**

A complete, reproducible machine-learning project that predicts whether a credit-card
customer will **default on their next payment**, using six months of real billing,
payment and delinquency history.

---

## 1. Problem statement

> Given an individual's past financial behaviour, predict their creditworthiness.

This is a **binary classification** problem:

| Label | Meaning | Business reading |
|---|---|---|
| `0` | Customer paid the next bill | **GOOD CREDIT** — creditworthy |
| `1` | Customer defaulted on the next bill | **BAD CREDIT** — not creditworthy |

Classification (rather than regression) is the correct framing because the outcome is a
discrete yes/no event. The models output a **probability of default**, which is what a
lender actually needs in order to set an approval cut-off.

---

## 2. Dataset

| | |
|---|---|
| **Name** | Default of Credit Card Clients |
| **Source** | UCI Machine Learning Repository, dataset ID 350 — <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients> |
| **Origin** | A Taiwanese bank, April – September 2005 |
| **Rows (raw)** | 30,000 |
| **Rows (after removing 35 exact duplicates)** | 29,965 |
| **Raw predictors** | 23 |
| **Predictors after feature engineering** | 52 |
| **Target** | `default` — default payment in October 2005 |
| **Class balance** | 77.88% repaid / 22.12% defaulted (≈ 3.52 : 1) |
| **Missing values in the source** | None |
| **Licence** | CC BY 4.0 — free to use with attribution |

The dataset downloads automatically on first run; nothing needs to be fetched by hand.

### Why this dataset?

Two credit-risk datasets were downloaded and inspected before choosing:

| Criterion | **UCI Default of Credit Card Clients** ✅ | UCI Statlog German Credit |
|---|---|---|
| Rows | **30,000** | 1,000 |
| Payment-history time series | **Yes — 6 months of repayment status, bills and payments** | None |
| Feature-engineering scope | Utilisation, payment ratios, delinquency trends | Static attributes only |
| Stability of CV estimates | High | Poor at n = 1,000 |
| Target clarity | Explicit default / no-default | Good / bad credit |

German Credit is the more famous dataset and the easier choice, but it has **no
financial time series**, which would make the assignment's core requirement —
*"feature engineering from financial history"* — impossible. The Taiwan dataset provides
`PAY_0…PAY_6` (repayment delay), `BILL_AMT1…6` (monthly bills) and `PAY_AMT1…6` (monthly
payments), enabling genuine credit-risk features.

### Column meanings

| Column | Meaning |
|---|---|
| `LIMIT_BAL` | Credit limit granted (NT$) |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Demographics |
| `PAY_0`, `PAY_2` … `PAY_6` | Repayment status Sept → April. `-2` = no consumption, `-1` = paid in full, `0` = revolving credit, `1…9` = months of delay |
| `BILL_AMT1` … `BILL_AMT6` | Bill statement amount, Sept → April |
| `PAY_AMT1` … `PAY_AMT6` | Amount actually paid, Sept → April |
| `default` | **Target** — 1 if the customer defaulted in October |

---

## 3. Project structure

```
Credit-Scoring-model/
├── data/
│   ├── raw/                          # auto-downloaded UCI .xls (git-ignored)
│   └── processed/                    # cleaned + engineered dataset (git-ignored)
├── notebooks/
│   └── credit_scoring_analysis.ipynb # full narrative walkthrough
├── src/
│   ├── config.py                     # paths, seeds, cost assumptions
│   ├── data_loader.py                # download + clean
│   ├── feature_engineering.py        # financial feature construction
│   ├── preprocessing.py              # ColumnTransformer pipeline
│   ├── models.py                     # model zoo + search spaces
│   ├── eda.py                        # exploratory figures
│   ├── evaluate.py                   # metrics, threshold search, plots
│   ├── train.py                      # end-to-end training orchestrator
│   └── predict.py                    # prediction interface for new applicants
├── scripts/
│   └── build_notebook.py             # generates + executes the notebook
├── tests/
│   └── verify_pipeline.py            # 12-check end-to-end verification suite
├── models/
│   └── final_model.pkl               # pipeline + threshold + metadata
├── outputs/
│   ├── figures/                      # all PNG figures
│   └── results/                      # all CSV/JSON result tables + training log
├── requirements.txt
├── README.md
└── report.md                         # detailed analysis & viva notes
```

---

## 4. How to run

```bash
pip install -r requirements.txt
```

```bash
python -m src.train
```

Downloads the data, runs EDA, engineers features, trains and tunes every model, and
writes all figures, tables and the saved model. Takes roughly 20–30 minutes.

```bash
python -m src.train --quick
```

Same code path on a 4,000-row subsample with smaller searches (~2 minutes) — for
smoke-testing only, not for reportable results.

```bash
python -m src.predict
```

Scores two example applicants using the saved model.

```bash
python -m tests.verify_pipeline
```

Runs the 12-check verification suite (loading, features, leakage, split,
preprocessing, training, CV, tuning, metrics, save/reload, prediction, artefacts).

```bash
python -m scripts.build_notebook --run
```

Regenerates the notebook and executes every cell to confirm it works.

---

## 5. Methodology

### 5.1 Cleaning

1. **`ID` dropped** — a row identifier has no predictive meaning and would let a tree
   memorise individual customers.
2. **35 exact duplicates removed** — otherwise the same customer could appear in both
   the training and test sets, inflating the score.
3. **Undocumented category codes folded into `others`** — `EDUCATION` contained codes
   `0`, `5`, `6` (345 rows) and `MARRIAGE` contained code `0` (54 rows) that the UCI
   documentation does not define.
4. Negative bills (overpayment, 590 rows) and over-limit bills (2,115 rows) were
   **kept** — both are legitimate, informative financial states.

### 5.2 Leakage prevention

| Risk | Control |
|---|---|
| Identifier shortcut | `ID` dropped before modelling |
| Test data influencing preprocessing | All imputation / scaling / encoding lives **inside** the `Pipeline`, so statistics are fitted per CV fold |
| Test data influencing model choice | Model selected on **cross-validated** training scores, before the test set is scored |
| Test data influencing the threshold | Threshold chosen from **out-of-fold training** predictions |
| Look-ahead in time | Target is October 2005; every predictor covers April–September 2005 |
| Duplicate rows across the split | De-duplicated before splitting |

### 5.3 Feature engineering

All engineered features are computed **row-wise from a single customer's own record**,
so this step is leakage-free and safely precedes the split.

| Group | Features | Financial meaning |
|---|---|---|
| **Credit utilisation** | `util_1…6`, `avg_utilization`, `max_utilization`, `utilization_trend`, `over_limit_months` | Share of the granted limit consumed. Sustained high utilisation is the classic stress signal. |
| **Repayment effort** | `pay_ratio_1…5`, `avg_pay_ratio`, `min_pay_ratio`, `zero_payment_months` | How much of last month's bill was actually paid off. |
| **Delinquency history** | `n_delinquent_months`, `max_delinquency`, `recent_delinquency`, `delinquency_trend`, `months_paid_full`, `months_no_consumption` | Frequency, severity and *direction* of late payment. |
| **Financial burden** | `avg_bill_amt`, `avg_pay_amt`, `bill_volatility`, `payment_to_limit`, `remaining_credit` | Absolute debt load and the capacity to service it. |

**A timing detail that matters:** `PAY_AMT_t` is money paid *during* month *t*, which
settles the bill raised in month *t+1*. The payment ratio is therefore
`PAY_AMT_t / BILL_AMT_{t+1}`, not `PAY_AMT_t / BILL_AMT_t`. Where the previous bill was
zero or negative the ratio is genuinely undefined and is left as `NaN` for the
pipeline's imputer to handle.

### 5.4 Preprocessing

| Feature type | Steps |
|---|---|
| Numeric | Median imputation → standard scaling *(scaling applied only for Logistic Regression; tree ensembles are invariant to monotone rescaling)* |
| Categorical | Most-frequent imputation → one-hot encoding with `handle_unknown="ignore"` so an unseen category cannot crash scoring |

### 5.5 Validation strategy

| Split | Size | Used for |
|---|---|---|
| **Train** | 80% (23,972 rows) | Fitting **and** all hyper-parameter search |
| **Validation** | 5-fold stratified CV *inside* the training set | Model comparison, tuning, threshold selection |
| **Test** | 20% (5,993 rows) | Scored **once**, for the final report only |

Stratification preserves the 22.12% default rate in every split and every fold.

---

## 6. Results

All figures below come from a single executed run of `python -m src.train`
(1,356 s, exit code 0). Full log: `outputs/results/training_log.txt`.

### Model comparison — held-out test set (5,993 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (Majority) | 0.7787 | 0.0000 | 0.0000 | 0.0000 | — | — |
| Logistic Regression | 0.7542 | 0.4573 | 0.5943 | 0.5169 | 0.7613 | 0.5150 |
| Decision Tree | 0.7587 | 0.4653 | 0.6063 | 0.5265 | 0.7697 | 0.5348 |
| Random Forest | 0.7651 | 0.4759 | 0.6094 | 0.5344 | 0.7782 | 0.5592 |
| **HistGradientBoosting** | 0.7545 | 0.4600 | 0.6282 | 0.5311 | **0.7818** | **0.5631** |
| **HistGradientBoosting @ threshold 0.405** | 0.6738 | 0.3796 | **0.7481** | 0.5037 | 0.7818 | 0.5631 |

> Note the baseline: predicting "nobody defaults" scores **77.87% accuracy** while
> catching **zero** defaulters. It beats three of the four real models on accuracy —
> which is exactly why this project is judged on ROC-AUC, recall and PR-AUC.

### Selected model

**`HistGradientBoostingClassifier`** — chosen on cross-validated ROC-AUC (**0.7899**)
*before* the test set was opened, and it held that lead on unseen data (**0.7818**).
The decision threshold **0.405** was cost-optimised on out-of-fold *training*
predictions under an assumed 5:1 false-negative:false-positive cost ratio, lifting
recall from 0.6282 to **0.7481** — the model catches **992 of 1,326 defaulters (74.8%)**.

### What drives credit risk

| Rank | Feature | ROC-AUC drop when shuffled |
|---|---|---|
| 1 | `n_delinquent_months` *(engineered)* | 0.03050 |
| 2 | `PAY_0` — most recent repayment status | 0.02654 |
| 3 | `remaining_credit` *(engineered)* | 0.01183 |
| 4 | `max_delinquency` *(engineered)* | 0.00503 |
| 5 | `BILL_AMT1` | 0.00417 |

Payment behaviour dominates; demographics contribute almost nothing. **8 of the top 12
predictors are engineered features**, and the strongest one does not exist in the raw
data — the feature engineering earned its place.

### Fairness

Dropping the protected attribute `SEX` changes test ROC-AUC by **0.0002**
(0.7818 → 0.7816) — indistinguishable from zero. A production deployment should exclude
it: compliance benefit, no measurable accuracy cost.

See **[report.md](report.md)** for the full analysis — imbalance study, tuning detail,
confusion matrices, error analysis, limitations and viva notes.

---

## 7. Licence and attribution

Dataset: Yeh, I.-C. & Lien, C.-H. (2009), *The comparisons of data mining techniques for
the predictive accuracy of probability of default of credit card clients*, Expert Systems
with Applications. Distributed by the UCI Machine Learning Repository under CC BY 4.0.
