# Credit Scoring Model — Results & Analysis Report

**Task 1 — Predicting creditworthiness from past financial data**

Every number in this report was produced by executing `python -m src.train` and is
stored in `outputs/results/`. Nothing is typed by hand or estimated.

- Full console log: [`outputs/results/training_log.txt`](outputs/results/training_log.txt)
- Total runtime: **1,356 seconds (~23 minutes)**, exit code 0
- Random seed: **42** throughout (`src/config.py`)

---

## 1. Dataset summary

| Property | Value |
|---|---|
| Dataset | UCI *Default of Credit Card Clients* (ID 350) |
| Rows after cleaning | **29,965** (30,000 − 35 duplicates) |
| Predictors (raw → engineered) | 23 → **52** (49 numeric + 3 categorical) |
| Columns after one-hot encoding | 58 |
| Target | `default` — default payment next month |
| Class distribution | 23,335 repaid (77.87%) / 6,630 defaulted (**22.13%**) |
| Imbalance ratio | **3.52 : 1** |
| Missing values in source | 0 |
| Train / test | 23,972 / 5,993 (both at a 0.2213 default rate) |

---

## 2. Cross-validated comparison (training set only)

5-fold stratified CV. This is what the model choice was based on — the test set was
still sealed at this point.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | ± std | PR-AUC |
|---|---|---|---|---|---|---|---|
| Baseline (Majority) | 0.7787 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.2213 |
| Logistic Regression | 0.7563 | 0.4618 | 0.6139 | 0.5271 | 0.7666 | 0.0100 | 0.5053 |
| Decision Tree | 0.7528 | 0.4617 | 0.6354 | 0.5326 | 0.7766 | 0.0052 | 0.5290 |
| Random Forest | 0.7857 | 0.5142 | 0.5756 | 0.5431 | 0.7840 | 0.0097 | 0.5543 |
| **HistGradientBoosting** | 0.7611 | 0.4704 | 0.6348 | 0.5403 | **0.7869** | 0.0083 | **0.5627** |

> **The single most important line in this table is the baseline.** A model that
> predicts "nobody defaults" achieves **77.87% accuracy** — higher than Logistic
> Regression, the Decision Tree *and* HistGradientBoosting — while catching **zero**
> defaulters (recall = 0). This is why accuracy is the wrong headline metric for this
> problem, and why the project is judged on ROC-AUC, recall and PR-AUC instead.

---

## 3. Class imbalance — quantified and mitigated

At 22.13% positives the data is moderately imbalanced (3.52 : 1). Left untreated, every
model optimises for the majority class and under-predicts default.

Measured effect of `class_weight="balanced"` (5-fold CV on the training set,
`outputs/results/imbalance_study.csv`):

| Model | class_weight | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | none | 0.6303 | 0.3039 | 0.4099 | 0.7643 |
| Logistic Regression | **balanced** | 0.4618 | **0.6139** | **0.5271** | 0.7666 |
| Random Forest | none | 0.6604 | 0.3729 | 0.4766 | 0.7832 |
| Random Forest | **balanced** | 0.5142 | **0.5756** | **0.5431** | 0.7840 |
| HistGradientBoosting | none | 0.6723 | 0.3597 | 0.4685 | 0.7861 |
| HistGradientBoosting | **balanced** | 0.4704 | **0.6348** | **0.5403** | 0.7869 |

**Interpretation.** Re-weighting roughly **doubles recall** (HistGradientBoosting:
0.3597 → 0.6348) at the cost of precision, and improves F1 for every model. ROC-AUC
barely moves — as expected, since re-weighting changes the *operating point*, not the
underlying ranking quality. Because a missed defaulter costs a lender far more than a
wrongly declined applicant, this trade is the correct one, so `class_weight="balanced"`
is used in the final model.

**Why not SMOTE?** Three reasons:
1. It requires the extra `imbalanced-learn` dependency for no measured gain here.
2. It must be applied **inside** each CV fold — applying it before the split leaks
   synthetic neighbours of test points into training, a classic and severe error.
3. At 3.52 : 1 the imbalance is moderate; `class_weight` achieves the same re-balancing
   analytically, without inventing customers who never existed.

---

## 4. Hyper-parameter tuning

All searches used 5-fold stratified CV on the **training set only**, scored on ROC-AUC
(`outputs/results/tuning_results.csv`).

| Model | Search | Combos tested | Best CV ROC-AUC | Time | Best parameters |
|---|---|---|---|---|---|
| **HistGradientBoosting** | RandomizedSearchCV | 15 / 216 | **0.7899** | 368s | `learning_rate=0.03`, `max_leaf_nodes=15`, `max_iter=150`, `min_samples_leaf=50`, `l2_regularization=0.5` |
| Random Forest | RandomizedSearchCV | 12 / 144 | 0.7889 | 671s | `n_estimators=400`, `max_depth=16`, `min_samples_leaf=20`, `max_features='sqrt'` |
| Decision Tree | GridSearchCV | 48 / 48 | 0.7767 | 109s | `criterion='gini'`, `max_depth=6`, `min_samples_leaf=200` |
| Logistic Regression | GridSearchCV | 12 / 12 | 0.7666 | 19s | `C=1.0`, `solver='lbfgs'` |

Tuning lifted HistGradientBoosting from 0.7869 to **0.7899** CV ROC-AUC. Note the
tuned tree is *shallower and more regularised* than the default (15 leaf nodes,
learning rate 0.03) — the search pushed back against overfitting.

---

## 5. Final comparison on the held-out test set

The test set (5,993 rows) was scored **once**, after the model and threshold were fixed.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (Majority) | 0.7787 | 0.0000 | 0.0000 | 0.0000 | — | — |
| Logistic Regression | 0.7542 | 0.4573 | 0.5943 | 0.5169 | 0.7613 | 0.5150 |
| Decision Tree | 0.7587 | 0.4653 | 0.6063 | 0.5265 | 0.7697 | 0.5348 |
| Random Forest | 0.7651 | 0.4759 | 0.6094 | 0.5344 | 0.7782 | 0.5592 |
| **HistGradientBoosting** | 0.7545 | 0.4600 | 0.6282 | 0.5311 | **0.7818** | **0.5631** |
| **HistGradientBoosting @ threshold 0.405** | 0.6738 | 0.3796 | **0.7481** | 0.5037 | 0.7818 | 0.5631 |

Figures: `model_comparison.png`, `confusion_matrices.png`, `roc_curves.png`,
`precision_recall_curves.png`.

**CV ranking held on the test set.** HistGradientBoosting was the best model in CV
(0.7899) and remained the best on unseen data (0.7818 ROC-AUC, 0.5631 PR-AUC). The
small drop from CV to test is the expected, healthy gap — not overfitting.

---

## 6. The credit-risk decision: which error costs more?

| Error | What actually happened | Consequence for the lender | Relative cost |
|---|---|---|---|
| **False Negative** | Predicted *repay*, customer **defaulted** | A bad loan was **approved**. The bank loses the outstanding principal. | **High** |
| **False Positive** | Predicted *default*, customer **repaid** | A good customer was **declined**. The bank loses the profit margin and some goodwill. | Lower |

**Metric meanings in this context**

- **Precision** — of everyone we flagged as risky, how many really defaulted?
  Low precision means rejecting good customers.
- **Recall** — of everyone who really defaulted, how many did we catch?
  Low recall means bad loans slip through.
- **F1** — harmonic mean of the two; used when both errors matter.
- **ROC-AUC** — probability the model ranks a random defaulter above a random payer.
  Threshold-independent, which is why it is the right criterion for *choosing a model*.
- **PR-AUC** — average precision across recall levels; more informative than ROC-AUC
  under imbalance because it ignores the easy true negatives.

**Losing money on a defaulter is worse than losing margin on a good customer**, so
recall is weighted more heavily than precision. This is encoded explicitly:

```
total cost = 5 × (false negatives) + 1 × (false positives)
```

The 5:1 ratio is a documented **assumption** in `src/config.py`; a real lender would
substitute its own loss-given-default figures.

### Threshold selection

The threshold was chosen by minimising that cost over **out-of-fold training
predictions** — never on the test set.

| Threshold | Training cost |
|---|---|
| 0.500 (default) | 13,381 |
| **0.405 (selected)** | **12,889** — 3.7% lower |

Effect on the test set: recall rises from **0.6282 → 0.7481** (the model now catches
**75% of all defaulters** instead of 63%), while accuracy falls from 0.7545 to 0.6738.
**That accuracy drop is the intended trade**, not a regression. See
`threshold_analysis.png`.

---

## 7. Final model

| | |
|---|---|
| **Model** | `HistGradientBoostingClassifier` inside a full sklearn `Pipeline` |
| **Why selected** | Highest cross-validated ROC-AUC (**0.7899**) — chosen before the test set was opened — and it held that lead on unseen data (0.7818 ROC-AUC, 0.5631 PR-AUC, the best of all five models on both ranking metrics). |
| **Best parameters** | `learning_rate=0.03`, `max_leaf_nodes=15`, `max_iter=150`, `min_samples_leaf=50`, `l2_regularization=0.5`, `class_weight='balanced'` |
| **Decision threshold** | **0.405** (cost-optimised on out-of-fold training predictions) |
| **Saved to** | `models/final_model.pkl` — contains the *whole* pipeline (imputation → scaling → one-hot → classifier), the threshold, and full metadata |

**Test metrics at the operating threshold (0.405):**

| Metric | Value |
|---|---|
| Accuracy | 0.6738 |
| Precision | 0.3796 |
| Recall | **0.7481** |
| F1 | 0.5037 |
| ROC-AUC | 0.7818 |
| PR-AUC | 0.5631 |

**Confusion matrix on the test set (5,993 customers) at threshold 0.405:**

| | Predicted repay | Predicted default |
|---|---|---|
| **Actually repaid** (4,667) | 3,046 (TN) | 1,621 (FP) |
| **Actually defaulted** (1,326) | 334 (FN) | 992 (TP) |

The model catches **992 of 1,326 defaulters (74.8%)** and misses 334.

**Advantages**
- Best ranking performance of every model tried, on both CV and test.
- Handles the non-linear, interacting effects of payment history natively.
- Fast to train (368s for a 15-point search) and fast to score.
- Ships in scikit-learn — no extra dependency versus XGBoost/LightGBM.

**Limitations**
- Less directly interpretable than Logistic Regression. A regulator asking "why was I
  declined?" needs the permutation-importance analysis in §8, not a coefficient table.
  If explainability outranks accuracy, **Logistic Regression at 0.7613 test ROC-AUC —
  only 0.02 lower — is a defensible alternative.**
- Precision at the chosen threshold is 0.3796: roughly **6 in 10 flagged applicants
  would actually have repaid**. That is the deliberate price of 75% recall.

---

## 8. What financial factors drive credit risk?

Permutation importance measures the real drop in test ROC-AUC when a column is shuffled.
It is model-agnostic and, unlike impurity importance, is not biased toward
high-cardinality features. *(HistGradientBoosting exposes no built-in importance
attribute, so this is the reference measure —
`outputs/figures/feature_importance_permutation.png`.)*

| Rank | Feature | ROC-AUC drop | ± std | Meaning |
|---|---|---|---|---|
| 1 | `n_delinquent_months` | **0.03050** | 0.00224 | **Engineered** — how many of the last 6 months were late |
| 2 | `PAY_0` | **0.02654** | 0.00343 | Most recent repayment status (September) |
| 3 | `remaining_credit` | 0.01183 | 0.00188 | **Engineered** — limit minus current balance |
| 4 | `max_delinquency` | 0.00503 | 0.00097 | **Engineered** — worst delay in 6 months |
| 5 | `BILL_AMT1` | 0.00417 | 0.00063 | Most recent bill |
| 6 | `bill_volatility` | 0.00374 | 0.00069 | **Engineered** — instability of monthly spend |
| 7 | `min_pay_ratio` | 0.00277 | 0.00064 | **Engineered** — worst monthly repayment effort |
| 8 | `EDUCATION` | 0.00199 | 0.00059 | Education level |
| 9 | `PAY_AMT2` | 0.00105 | 0.00079 | Payment made in August |
| 10 | `pay_ratio_2` | 0.00104 | 0.00028 | **Engineered** — August repayment effort |
| 11 | `max_utilization` | 0.00099 | 0.00039 | **Engineered** — peak credit utilisation |
| 12 | `utilization_trend` | 0.00090 | 0.00040 | **Engineered** — is debt growing? |

### The answer, in plain language

> **Credit risk is driven overwhelmingly by payment behaviour, not by who the customer
> is.** How *often* someone has been late (`n_delinquent_months`) and whether they are
> late *right now* (`PAY_0`) are together worth more than every other feature combined.
> After that comes financial headroom (`remaining_credit`, `max_utilization`) and
> repayment effort (`min_pay_ratio`). Demographics — age, marital status, sex —
> contribute almost nothing.

**The feature engineering earned its place: 8 of the top 12 predictors are engineered
features**, and the single strongest predictor (`n_delinquent_months`) does not exist in
the raw data at all.

---

## 9. Error analysis

Test set at threshold 0.405 (`outputs/results/error_analysis.csv`):

| Group | Count | Share | Mean predicted P(default) | Mean `LIMIT_BAL` | Mean `PAY_0` | Mean `n_delinquent_months` |
|---|---|---|---|---|---|---|
| True Negative (correctly approved) | 3,046 | 50.8% | 0.254 | 205,870 | −0.43 | 0.06 |
| **False Positive** (good customer rejected) | 1,621 | 27.0% | 0.584 | 126,255 | +0.14 | **1.35** |
| **False Negative** (defaulter approved) | 334 | 5.6% | 0.294 | 179,790 | −0.42 | **0.08** |
| True Positive (defaulter caught) | 992 | 16.6% | 0.721 | 116,683 | +1.02 | 2.66 |

**What the mistakes have in common**

- **False negatives are "clean-record" defaulters.** Their mean delinquency count is
  0.08 — essentially spotless — and their mean `PAY_0` is −0.42, meaning they were
  paying on time right up to the month they defaulted. Their credit limits (179,790)
  are also well above those of caught defaulters (116,683). **The model cannot see a
  sudden shock** — job loss, illness, a business failure — because nothing in six months
  of card history announces it. This is a limit of the data, not a fixable modelling bug.
- **The model is uncertain, not confidently wrong.** Mean predicted probability for
  false negatives is 0.294 against a 0.405 threshold — they sit just below the line.
  Lowering the threshold further would convert some of them, at the cost of many more
  false positives.
- **False positives look genuinely risky.** They average 1.35 delinquent months and
  positive `PAY_0`. The model is not being arbitrary; these customers really do carry
  warning signs and simply happened to recover.

---

## 10. Fairness note

`SEX`, `EDUCATION` and `MARRIAGE` are protected or proxy attributes, and using them in
lending decisions is restricted in many jurisdictions.

A sensitivity check retrained the winning model **without `SEX`**
(`outputs/results/fairness_sensitivity.csv`):

| Variant | Test ROC-AUC |
|---|---|
| With `SEX` | 0.7818 |
| **Without `SEX`** | **0.7816** |

**Dropping `SEX` costs 0.0002 ROC-AUC — statistically indistinguishable from zero.**
The protected attribute contributes essentially nothing, so a production deployment
should simply exclude it: there is a compliance benefit and no measurable accuracy cost.
`get_feature_lists(include_sex=False)` supports this directly.

---

## 11. Prediction system

`src/predict.py` loads the saved bundle and scores raw applicant data. Real output from
the trained model (`python -m src.predict`):

```
Loaded model : HistGradientBoosting
Trained at   : 2026-09-05T01:03:35
Threshold    : 0.405

----------------------------------------------------------
  Applicant A -- strong payer
----------------------------------------------------------
  Model used                 : HistGradientBoosting
  Prediction                 : GOOD CREDIT (likely to repay)
  Decision                   : APPROVE
  Risk category              : LOW RISK
  Probability of GOOD credit : 88.7%
  Probability of BAD credit  : 11.3%
  Decision threshold         : 0.405
----------------------------------------------------------

----------------------------------------------------------
  Applicant B -- distressed borrower
----------------------------------------------------------
  Model used                 : HistGradientBoosting
  Prediction                 : BAD CREDIT (likely to default)
  Decision                   : DECLINE / REVIEW
  Risk category              : VERY HIGH RISK
  Probability of GOOD credit : 14.9%
  Probability of BAD credit  : 85.1%
  Decision threshold         : 0.405
----------------------------------------------------------
```

Applicant A pays every bill in full on a 350,000 limit; Applicant B is 3 months late on
a maxed-out 20,000 limit. The *inputs* are hand-constructed; **every probability above
comes from the trained model at run time.**

---

## 12. Limitations

1. **One market, one period.** Taiwanese credit-card holders, April–September 2005.
   Consumer behaviour and macroeconomic conditions differ elsewhere and today; the model
   would need revalidation before any other use.
2. **Six-month window only.** No income, employment, other lenders, or long-run credit
   history — all of which real credit bureaus use. This caps achievable performance;
   ~0.78 ROC-AUC is in the normal published range for this dataset.
3. **Card default, not bankruptcy.** The target is one missed credit-card payment, not
   insolvency or write-off.
4. **Assumed cost ratio.** The 5:1 FN:FP ratio drives the threshold. A different ratio
   moves the operating point — the model itself is unchanged.
5. **Low precision at the chosen operating point.** 0.3796 precision means many good
   customers are flagged. Appropriate for a *review-and-escalate* workflow, not for
   fully automated rejection.
6. **Selection bias.** The data contains only customers who were **already approved**
   for a card. It cannot describe applicants the bank had previously declined.
7. **No calibration analysis.** Probabilities are used for ranking and thresholding;
   they have not been checked for calibration (e.g. that predicted 30% really means 30%).

**Recommended next steps:** calibration curves, monotonic constraints for regulatory
explainability, conversion to a WOE/IV scorecard, and population-drift monitoring.

---

## 13. Viva / presentation notes

**Q: What is the problem?**
Predict whether a credit-card customer will default on their next payment — a binary
classification problem — using six months of their financial history.

**Q: Why is this classification and not regression?**
The outcome is a discrete yes/no event. The models output a *probability* of default,
which is what a lender needs to set an approval cut-off.

**Q: What does the target mean?**
`default = 1` means the customer failed to pay their October 2005 bill. 22.13% of
customers did.

**Q: How did you avoid data leakage?**
Six controls: dropped `ID`; removed duplicates *before* splitting; put all preprocessing
inside the `Pipeline` so statistics are fitted per fold; selected the model on
cross-validated scores; chose the threshold on out-of-fold *training* predictions; and
confirmed all features precede the target in time (April–September predict October).

**Q: Why is accuracy not the headline metric?**
Because a model predicting "nobody defaults" scores 77.87% accuracy while catching zero
defaulters. That baseline beats three of my four real models on accuracy — which proves
accuracy is measuring the wrong thing here.

**Q: Which metric did you optimise, and why?**
ROC-AUC to *select the model* (threshold-independent, measures ranking quality), then I
tuned the *threshold* for recall, because a false negative — approving someone who
defaults — costs the bank the whole principal, while a false positive only costs the
profit margin.

**Q: Why did HistGradientBoosting win?**
Highest cross-validated ROC-AUC (0.7899), decided before I opened the test set, and it
kept the lead on unseen data (0.7818). Gradient boosting handles the non-linear,
interacting effects in payment history better than a linear model or a single tree.

**Q: What are the most important financial factors?**
Delinquency dominates: `n_delinquent_months` (how often late) and `PAY_0` (late right
now) outweigh everything else combined. Then financial headroom (`remaining_credit`,
`max_utilization`) and repayment effort (`min_pay_ratio`). Demographics barely matter.

**Q: Did your feature engineering actually help?**
Yes — 8 of the top 12 predictors are engineered, and the single strongest one,
`n_delinquent_months`, does not exist in the raw data.

**Q: How did you handle imbalance?**
Quantified it (3.52:1), used stratified splitting and CV, applied
`class_weight='balanced'`, and measured the effect: recall roughly doubled
(0.3597 → 0.6348 for the winning model). I avoided SMOTE because it adds a dependency
for no gain and is easy to misapply before the split.

**Q: What is the model's biggest weakness?**
It cannot predict sudden shocks. The 334 defaulters it missed had almost spotless
records — 0.08 delinquent months on average — and were paying on time until the month
they defaulted. Nothing in six months of card history announces a job loss.

**Q: How would you score a new applicant?**
`predict_applicant()` takes the raw financial fields, applies the same feature
engineering, and runs the saved pipeline — which includes all preprocessing — returning
a probability of default, a risk band, and an approve/decline decision at the tuned
0.405 threshold.
