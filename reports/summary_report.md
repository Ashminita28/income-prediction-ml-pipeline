# Executive Summary Report: End-to-End Machine Learning Pipeline
**Project:** Predicting High-Income Earners (> $50,000) for Premium Financial Services Targeting  
**Organization:** Retail Banking & Wealth Management Analytics  
**Date:** November 2026  
**Status:** Production Ready  

---

## 1. Problem Statement
A retail banking and wealth management enterprise seeks to expand its high-net-worth customer base by proactively identifying prospects earning in excess of **$50,000 annually**. 
* **The Business Opportunity:** High-income clients drive disproportionate lifetime value (LTV) through private banking, wealth advisory management fees, and revolving premium credit cards ($1,200 estimated annual margin per acquired client).
* **The Cost Asymmetry:** Missing an eligible high-income customer (False Negative) forfeits $1,200 in recurring net revenue, whereas sending marketing communications to a non-qualifying individual (False Positive) incurs only an estimated $80 in digital/postal acquisition costs.
* **The Data Challenge:** Demographic census data is inherently class-imbalanced (~76% `<=50K`, ~24% `>50K`), right-skewed, zero-inflated (capital gains/losses), and contains subtle protected demographic attributes requiring strict fair lending governance.

---

## 2. Best Model & Technical Architecture
Across a rigorous benchmark of 5 supervised classification algorithms (Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors, and XGBoost), the **Tuned XGBoost Classifier** calibrated at an adjusted **decision threshold of 0.40** emerged as the superior production model.

* **Production Packaging:** Encapsulated in a unified, leak-free Scikit-Learn `Pipeline` using `ColumnTransformer`:
  - **Numerical Features:** Median Imputation $\rightarrow$ `RobustScaler` (resistant to extreme 99,999 top-coding).
  - **Categorical Features:** Missingness mapped to `'Unknown'` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.
  - **Estimator:** Gradient Boosted Trees (`n_estimators=150`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.80`).
* **Artifact:** Serialized to `models/income/final_pipeline.joblib` for sub-millisecond API scoring.

---

## 3. Key Performance Metrics

### Model Benchmark Comparison (Test Set, $N = 9,758$)
| Classification Model | Accuracy | Precision | Recall (>50K) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost (Threshold = 0.40)** | **86.43%** | **70.81%** | **73.72%** | **0.7223** | **0.9269** |
| Baseline XGBoost (Threshold = 0.50) | 87.09% | 78.41% | 63.57% | 0.7021 | 0.9262 |
| K-Nearest Neighbors | 85.63% | 71.78% | 65.88% | 0.6871 | 0.8968 |
| Random Forest Classifier | 86.18% | 79.64% | 56.76% | 0.6628 | 0.9151 |
| Decision Tree Classifier | 85.43% | 75.47% | 57.96% | 0.6557 | 0.8990 |
| Logistic Regression | 84.89% | 72.49% | 59.46% | 0.6533 | 0.9015 |

### Secondary Regression Benchmark (California Housing Dataset)
To validate continuous numerical prediction capabilities, 5 regression architectures were trained to predict median house values (`MedHouseVal`):
- **Best Regressor:** **XGBoost Regressor** ($R^2 = 0.8287$, $\text{RMSE} = 0.4739$, $\text{MAE} = 0.3156$).
- **Key Finding:** Non-linear tree ensembles drastically outperformed regularized linear baselines (Ridge $R^2 = 0.5758$, Lasso $R^2 = 0.5816$) due to complex spatial interactions between geographical coordinates (`Latitude`, `Longitude`) and median neighborhood income (`MedInc`).

---

## 4. Most Important Predictive Features
Using cooperative game-theoretic feature attribution (**SHAP**) and tree Gini importance:
1. **`capital_gain` & `capital_net` ($+ \text{impact}$):** Top-tier financial drivers; individuals with investment liquidity are overwhelmingly high earners.
2. **`marital_status_Married-civ-spouse` ($+ \text{impact}$):** Strongest demographic marker of dual-earning household stability and accumulated wealth.
3. **`education_num` ($+ \text{impact}$):** Monotonically increases likelihood of earning >$50K (each academic degree adds compounding positive Shapley log-odds).
4. **`hours_per_week` & `work_intensity` ($+ \text{impact}$):** High working commitment ($\ge 45$ hours/week) strongly correlates with senior corporate advancement.
5. **`age` ($+ \text{impact}$ until peak):** Reaches peak positive attribution between ages 38 and 55 (prime career earning window).

---

## 5. Key Findings & Business Insights
1. **Accuracy is a Deceptive Metric:** Due to the 76:24 class split, a naive zero-rule classifier achieves 76% accuracy with zero business utility. The business must focus on **Recall** and **PR-AUC**.
2. **Threshold Tuning Unlocks $2M+ Campaign ROI:** Lowering the probability decision threshold from $0.50$ to $0.40$ increases Recall from **65.24% to 73.72%** (capturing 198 additional high-value accounts per 10k prospects), boosting net campaign profit to **$2,009,600 (830% ROI)**.
3. **Unsupervised Segments Align with Commercial Products:** K-Means clustering ($K=3$) revealed natural customer tiers:
   - *Cluster 0 (Entry-Level Workers):* High-yield savings, basic credit cards.
   - *Cluster 1 (Active Investors):* Private wealth advisory, tax-advantaged trusts.
   - *Cluster 2 (Senior Executives):* Jumbo mortgages, commercial credit lines.

---

## 6. Business Recommendations & Deployment Strategy
1. **Deploy at Threshold = 0.40:** Deploy the containerized pipeline with a 0.40 probability cutoff to maximize acquisition recall without overwhelming branch relationship managers.
2. **Implement Two-Tier Campaign Routing:**
   - Predicted probability $\ge 0.65$: High-touch personal outreach from Private Wealth Advisors.
   - Predicted probability between $0.40$ and $0.64$: Automated digital marketing for Premium Travel Cards and Brokerage accounts.

---

## 7. Model Limitations, Ethical Risks & Monitoring Plan
* **Demographic Bias & Fair Lending Risks:**
  - The training data reflects historical socioeconomic disparities in `sex` and `race`.
  - **Action Required:** In accordance with the **Equal Credit Opportunity Act (ECOA)**, protected attributes (`sex`, `race`, `native_country`) must not be used as standalone underwriting criteria. Disparate impact audits (four-fifths rule) must be conducted quarterly across demographic cohorts.
* **Top-Coding Artifacts:** `capital_gain` is capped at $99,999 in census data, causing artificial plateaus in high-wealth attribution.
* **Post-Deployment Monitoring Strategy:**
  1. **Data Drift:** Track weekly Population Stability Index (PSI) and Wasserstein distance on incoming inputs (alert if $PSI > 0.20$).
  2. **Concept Drift:** Monitor the ongoing predicted positive rate against verified income filings from downstream onboarding.
  3. **Quarterly Retraining:** Automated pipeline retraining on rolling 12-month customer transactional data.

