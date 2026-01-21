# Credit Card Fraud Detection — Precision-Focused ML System

## Project Overview

Credit card fraud detection is a highly imbalanced, noisy, and time-dependent machine learning problem. Fraudulent transactions often resemble legitimate behavior, making naive threshold-based or accuracy-driven models ineffective.

This project implements a **precision-optimized fraud detection system** that emphasizes:

- Behavioral feature engineering
- Temporal integrity
- Probability ranking instead of hard classification
- Defensive, production-style decision strategies

Rather than optimizing for a single model metric, the project focuses on **how fraud decisions are ultimately made**, which mirrors real-world fraud detection systems.

---

##  Objective

Predict whether a transaction is fraudulent (`TX_FRAUD = 1`) or legitimate (`TX_FRAUD = 0`).

**Primary evaluation metric:**  
**Precision**, reflecting the high cost of false positives in real payment systems.

---

##  Dataset Summary

Transactions are split temporally:

- **Training:** August–December 2021  
- **Test:** January–April 2022  
  - Only **April 2022** predictions are submitted

Fraud rate in training data is approximately **2.2%**, making this a severely imbalanced classification problem.

---

##  Core Design Principles

### Temporal Integrity
- No random splits
- Validation always uses future data
- Prevents information leakage

### Precision-First Framing
- Accuracy is misleading under class imbalance
- Recall is controlled, not maximized
- Precision is optimized at the decision layer

### Model ≠ Decision
- Models output probabilities
- Fraud decisions are made using explicit logic
- Mirrors real-world fraud engines

---

##  Feature Engineering

The goal of feature engineering is to capture **behavioral deviation**, not raw transaction magnitude. Fraud is modeled as a deviation from a customer’s normal behavior, especially when multiple weak signals occur together.

---

###  Customer Spending Behavior Features

These features measure how unusual a transaction amount is **relative to a customer’s historical behavior**.

| Feature | Explanation |
|------|------------|
| `mean_amount` | Average transaction amount for the customer |
| `std_amount` | Variability of customer spending |
| `amt_minus_cust_mean` | Difference between transaction amount and customer mean |
| `amt_zscore` | Standardized deviation of transaction amount |
| `is_amt_unusual` | Flag indicating significantly abnormal spending |

**Interpretation:**  
Fraudulent transactions are rarely extreme globally, but they are often unusual *for the specific customer*. These features normalize spending behavior at the individual level.

---

###  Location Deviation Features

These features capture deviations in transaction location relative to customer norms.

| Feature | Explanation |
|------|------------|
| `cust_term_dist` | Distance between customer and terminal |
| `cust_dist_mean` | Typical transaction distance for customer |
| `cust_dist_std` | Variability of customer transaction distances |
| `dist_zscore` | Standardized distance deviation |
| `is_dist_unusual` | Flag for abnormal transaction location |

**Interpretation:**  
Absolute distance provides little signal due to dataset constraints. Relative distance deviation provides limited but useful context when combined with other signals.

---

###  Temporal Context Features

These features provide time-based context rather than acting as hard rules.

| Feature | Explanation |
|------|------------|
| `tx_hour` | Hour of transaction |
| `tx_weekday` | Day of week |
| `night_weight` | Soft risk weight applied during late-night hours |

**Interpretation:**  
Fraud often occurs during low-monitoring periods. Time alone is weak, but it amplifies other risk signals.

---

###  Velocity Features

These features detect sudden changes in transaction frequency.

| Feature | Explanation |
|------|------------|
| `time_since_last_txn` | Minutes since customer’s previous transaction |
| `txn_count_1h` | Customer transactions in last hour |
| `txn_count_24h` | Customer transactions in last 24 hours |

**Interpretation:**  
Fraud frequently manifests as rapid bursts of activity following account compromise.

---

###  Rolling Behavioral Features

These features capture **sustained abnormal behavior**, not isolated events.

| Feature | Explanation |
|------|------------|
| `rolling_amt_zscore_mean_24h` | Average amount deviation over last 24 hours |
| `rolling_unusual_amt_24h` | Count of abnormal transactions in last 24 hours |

**Interpretation:**  
Repeated abnormal behavior is a stronger fraud signal than a single outlier.

---

###  Terminal Risk Features

These features model merchant-level risk.

| Feature | Explanation |
|------|------------|
| `terminal_fraud_rate` | Historical fraud rate of terminal |
| `high_risk_terminal` | Flag for terminals with elevated fraud history |

**Interpretation:**  
Some terminals are consistently riskier due to compromise or merchant practices. Terminal-level history was one of the strongest predictors.

---

###  Interaction & Compound Risk Features

Fraud typically emerges when **multiple weak risk signals occur together**. These features explicitly model that interaction.

| Feature | Explanation |
|------|------------|
| `amt_night_weighted` | Abnormal spending amplified during high-risk hours |
| `dist_night_weighted` | Unusual transaction location during high-risk hours |
| `burst_and_unusual_amt` | Rapid transaction burst combined with abnormal spending |
| `terminal_burst_risk` | Sudden terminal activity weighted by terminal fraud history |

**Interpretation:**  
A single abnormal signal is often benign. These features capture situations where **context, speed, and abnormality reinforce each other**, which is strongly indicative of fraud.

---

###  Customer–Terminal Novelty Features

These features capture transaction novelty.

| Feature | Explanation |
|------|------------|
| `cust_terminal_txn_count` | Number of prior interactions with terminal |
| `is_first_time_terminal` | Flag for first-time terminal usage |
| `cust_terminal_txn_ratio` | Terminal usage frequency relative to customer activity |

**Interpretation:**  
Fraudulent transactions frequently occur at new or rarely used terminals for a customer.

---

##  Modeling

Two models were evaluated:

- Logistic Regression (baseline, interpretable)
- LightGBM (non-linear, tree-based)

**Finding:**  
Model choice had less impact than feature quality and decision strategy. LightGBM was selected for superior probability ranking.

---

## ❌ What Failed

### Fixed Probability Thresholds
- Achieved perfect precision
- Recall collapsed to near zero
- Not operationally useful

### Hard Rule-Based Gating
- Extremely high precision
- Covered only a handful of transactions
- Brittle and non-scalable

---

##  Final Defensive Strategy

### Soft Rules + Global Top-K Ranking

1. High-confidence rule combinations generate a **soft probability boost**
2. No transactions are discarded
3. Final decisions are made using **global Top-K ranking**

This approach:
- Preserves recall
- Improves precision significantly
- Avoids brittle rules
- Reflects real-world fraud engines

---

##  Validation Results

| Strategy | Precision | Recall |
|--------|----------|--------|
| Baseline Top-K | ~0.56 | ~0.02 |
| Soft-Boosted Top-K | **~0.75** | ~0.016 |

Final submission flags the **top 0.05%** of transactions by risk score.

---

##  Final Decision Logic

```text
1. Train LightGBM on full training data
2. Generate fraud probabilities
3. Apply soft rule-based probability boost
4. Rank transactions globally
5. Flag top 0.05% as fraud
6. Submit April 2022 predictions only

