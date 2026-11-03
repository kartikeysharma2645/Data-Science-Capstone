# Final Project Report

## Credit Risk Analytics — Predicting Credit Card Default

**Project Category:** Data Science / Machine Learning
**Internship:** YuvaIntern — AI/ML Research Specialist
**Dataset:** Default of Credit Card Clients
**Repository:** Data-Science-Capstone

---

# 1. Executive Summary

Credit risk assessment is an important component of financial decision-making. Financial institutions need methods to identify customers who may fail to meet their future payment obligations.

This project investigates the application of machine learning classification techniques to predict credit card payment defaults using historical customer information.

The analysis uses the Default of Credit Card Clients dataset, containing 30,000 observations and 23 predictor variables.

An end-to-end data science workflow was implemented, including data understanding, exploratory data analysis, data preprocessing, model development, evaluation, model interpretability, business insights, and robustness validation.

Five classification approaches were evaluated. The Random Forest Classifier was selected as the final model, using a decision threshold of 0.55.

On the held-out test set, the selected model achieved:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 80.15% |
| Precision | 55.32% |
| Recall    | 53.28% |
| F1-Score  | 54.28% |
| ROC-AUC   | 77.62% |

The analysis identified repayment history as an important source of predictive information.

The project demonstrates the potential of machine learning for credit-risk analysis. However, the model requires further calibration, fairness assessment, and external validation before any real-world deployment.

---

# 2. Introduction

Credit card default occurs when a customer fails to meet their required payment obligations.

For financial institutions, identifying potential default risk is important for monitoring credit portfolios, managing exposure, and supporting risk assessment.

Traditional credit-risk assessment methods can be supplemented with machine learning techniques capable of identifying patterns across multiple customer attributes and historical financial records.

This project applies supervised machine learning to historical credit card customer data to predict whether a customer will default on their payment in the following month.

## 2.1 Problem Statement

Develop a binary classification model that predicts whether a credit card customer will default on their payment in the next month based on historical repayment information, credit limits, billing statements, payment amounts, and demographic variables.

## 2.2 Project Objectives

1. Understand the dataset and its variables.
2. Identify patterns associated with credit card defaults.
3. Prepare the dataset for machine learning.
4. Train and evaluate multiple classification models.
5. Select a suitable model using relevant performance metrics.
6. Interpret important predictive features.
7. Develop illustrative customer risk segments.
8. Evaluate model robustness and subgroup performance.
9. Translate analytical findings into business recommendations.

---

# 3. Dataset Description

## 3.1 Dataset Source

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

Dataset URL:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains historical credit card customer information from Taiwan.

## 3.2 Dataset Dimensions

| Attribute           | Description           |
| ------------------- | --------------------- |
| Total observations  | 30,000                |
| Predictor variables | 23                    |
| Target variable     | 1                     |
| Problem type        | Binary classification |
| File format         | Excel                 |

## 3.3 Feature Categories

The dataset includes the following major feature groups:

* **Demographic information:** Age, sex, education, and marital status.
* **Credit information:** Credit limit assigned to the customer.
* **Repayment history:** Historical repayment status across multiple months.
* **Billing information:** Monthly bill statement amounts.
* **Payment information:** Previous payment amounts.

## 3.4 Target Variable

The target variable is:

`default payment next month`

| Value | Meaning    |
| ----- | ---------- |
| 0     | No default |
| 1     | Default    |

## 3.5 Target Distribution

| Class      | Records | Percentage |
| ---------- | ------: | ---------: |
| No Default |  23,364 |     77.88% |
| Default    |   6,636 |     22.12% |
| Total      |  30,000 |       100% |

The dataset has an imbalanced target distribution, making accuracy alone insufficient for evaluating model performance.

---

# 4. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the distribution of customer characteristics, repayment behavior, billing information, and payment patterns.

The analysis focused on identifying potentially useful relationships between customer attributes and the default target.

## 4.1 Target Distribution

The dataset contains approximately 77.88% non-default observations and 22.12% default observations.

This imbalance influenced the choice of evaluation metrics and model-selection strategy.

## 4.2 Demographic Variables

Demographic attributes such as age, sex, education, and marital status were examined to understand the composition of the dataset.

These variables were also considered during preprocessing and exploratory subgroup analysis.

## 4.3 Repayment History

Repayment status variables provide information about customers' historical payment behavior.

Repayment history emerged as an important predictive feature group during subsequent model interpretation.

## 4.4 Billing and Payment Behavior

Monthly billing statements and previous payment amounts were examined to identify financial behavior patterns.

Negative billing values were retained because they may represent credits or adjustments rather than invalid records.

## 4.5 Outlier Analysis

Potential outliers were examined in numerical financial variables.

Instead of automatically removing extreme observations, the preprocessing strategy retained the underlying records and applied transformations to improve the handling of skewed distributions.

---

# 5. Data Preprocessing

Data preprocessing was performed to prepare the dataset for machine learning.

## 5.1 Data Cleaning

The dataset was inspected for:

* Incorrect data types.
* Missing values.
* Unusual categorical codes.
* Duplicate or irrelevant identifiers.
* Numerical distribution issues.

The ID column was excluded from model training because it does not represent a meaningful predictive customer characteristic.

## 5.2 Categorical Variables

Categorical variables were grouped into appropriate categories.

Undocumented or unknown education codes were grouped with the other/unknown category. The undocumented marriage category was also grouped with the other category.

Nominal demographic features and repayment-status features were encoded using One-Hot Encoding.

## 5.3 Numerical Transformation

The preprocessing pipeline used:

* StandardScaler for age and credit limit.
* Yeo-Johnson transformation for billing and payment amount variables.
* StandardScaler after the power transformation.

This approach helped manage skewed financial variables while retaining their observations.

## 5.4 Data Splitting

The dataset was divided using stratified sampling:

| Dataset  | Observations | Percentage |
| -------- | -----------: | ---------: |
| Training |       24,000 |        80% |
| Testing  |        6,000 |        20% |

The split preserved the target-class distribution.

## 5.5 Leakage Prevention

Preprocessing transformations were fitted using training data rather than the complete dataset.

The preprocessing pipeline was also incorporated into model evaluation so that transformations could be fitted independently within the relevant training folds.

---

# 6. Model Development

Five classification approaches were evaluated.

| Model               | Purpose                        |
| ------------------- | ------------------------------ |
| Dummy Classifier    | Baseline comparison            |
| Logistic Regression | Linear classification baseline |
| Decision Tree       | Rule-based classification      |
| Random Forest       | Ensemble classification        |
| Gradient Boosting   | Boosting-based classification  |

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## 6.1 Threshold Optimization

The default classification threshold of 0.50 was not assumed to be optimal.

Alternative thresholds were evaluated using validation data.

The selected thresholds were:

| Model               | Selected Threshold |
| ------------------- | -----------------: |
| Logistic Regression |               0.55 |
| Random Forest       |               0.55 |
| Gradient Boosting   |               0.25 |

The held-out test set was not used to select these thresholds.

## 6.2 Final Model Selection

Random Forest was selected because it offered a balanced combination of precision, recall, F1-score, and ROC-AUC at the chosen threshold.

The selected configuration used 200 trees, a maximum depth of 12, balanced class weights, and a random state of 42.

---

# 7. Final Model Evaluation

The selected Random Forest model was evaluated on the held-out test set using a threshold of 0.55.

## 7.1 Performance Metrics

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8015 |
| Precision | 0.5532 |
| Recall    | 0.5328 |
| F1-Score  | 0.5428 |
| ROC-AUC   | 0.7762 |

## 7.2 Confusion Matrix

|                   | Predicted No Default | Predicted Default |
| ----------------- | -------------------: | ----------------: |
| Actual No Default |                4,102 |               571 |
| Actual Default    |                  620 |               707 |

## 7.3 Error Analysis

The model produced:

* True Negatives: 4,102
* False Positives: 571
* False Negatives: 620
* True Positives: 707

The false-positive cases represent customers classified as potential defaulters who did not default in the observed test data.

The false-negative cases represent actual defaulters that the model failed to identify.

The false-negative rate was approximately 46.72%.

This indicates that the model misses a meaningful proportion of actual defaults and should not be treated as a standalone credit-risk decision system.

---

# 8. Model Interpretability

Permutation feature importance was used to examine the contribution of individual features to model performance.

The leading predictive features included:

1. PAY_0
2. PAY_2
3. PAY_AMT2
4. PAY_AMT3
5. PAY_AMT1

Repayment-status variables were particularly influential.

This suggests that historical repayment behavior contains useful predictive information for identifying future default risk.

Feature importance indicates predictive relevance, not causation. It does not establish that changing an individual feature would directly change a customer's probability of default.

---

# 9. Customer Risk Segmentation

The final analysis grouped customers into illustrative low-, medium-, and high-risk segments.

| Risk Segment | Customers | Population | Observed Default Rate |
| ------------ | --------: | ---------: | --------------------: |
| Low          |     1,407 |     23.45% |                 5.76% |
| Medium       |     3,066 |     51.10% |                15.46% |
| High         |     1,527 |     25.45% |                50.56% |

The observed default rate increases substantially across the three segments.

This indicates that the segmentation provides a useful way to summarize differences in observed risk within the test population.

These groupings are illustrative analytical segments, not validated lending-policy thresholds.

The model's predicted scores should not be interpreted as calibrated default probabilities.

---

# 10. Robustness and Validation

The selected Random Forest pipeline was evaluated using stratified five-fold cross-validation.

## 10.1 Cross-Validation Results

| Metric    |   Mean | Standard Deviation |
| --------- | -----: | -----------------: |
| Accuracy  | 0.7838 |             0.0032 |
| Precision | 0.5100 |             0.0064 |
| Recall    | 0.5818 |             0.0088 |
| F1-Score  | 0.5435 |             0.0054 |
| ROC-AUC   | 0.7825 |             0.0065 |

The relatively low standard deviations indicate that the model's performance was reasonably consistent across the evaluated folds.

## 10.2 Out-of-Fold Evaluation

Additional out-of-fold evaluation produced:

| Metric            | Result |
| ----------------- | -----: |
| Average Precision | 0.5554 |
| Brier Score       | 0.1697 |

A calibration curve was also examined.

These results do not establish that the model's predicted probabilities are calibrated.

## 10.3 Gender Subgroup Analysis

| Group  | Customers | Accuracy | Precision | Recall | F1-Score |
| ------ | --------: | -------: | --------: | -----: | -------: |
| Female |     3,598 |   0.8118 |    0.5614 | 0.5313 |   0.5459 |
| Male   |     2,402 |   0.7860 |    0.5425 | 0.5348 |   0.5386 |

The results show differences in subgroup performance.

They do not constitute a comprehensive fairness assessment or establish the absence of discriminatory outcomes.

## 10.4 Age Subgroup Analysis

| Age Group | Customers | Accuracy | Precision | Recall | F1-Score |
| --------- | --------: | -------: | --------: | -----: | -------: |
| ≤25       |       767 |   0.7640 |    0.5346 | 0.5918 |   0.5617 |
| 26–35     |     2,492 |   0.8186 |    0.5408 | 0.5143 |   0.5272 |
| 36–45     |     1,775 |   0.8090 |    0.5792 | 0.5340 |   0.5557 |
| 46–55     |       788 |   0.7754 |    0.5561 | 0.5253 |   0.5403 |
| 56+       |       178 |   0.7640 |    0.5476 | 0.5000 |   0.5227 |

The results indicate variation across age groups.

The smaller sample size of some subgroups, particularly the 56+ group, should be considered when interpreting these metrics.

## 10.5 Validation Conclusion

The model demonstrated relatively consistent performance across the five cross-validation folds.

However, cross-validation does not replace evaluation on more recent or external datasets.

Further validation is required before considering production deployment.

---

# 11. Business Recommendations

Based on the findings, the following recommendations are proposed.

## 11.1 Risk-Based Monitoring

The model could support the prioritization of accounts for additional review and monitoring.

## 11.2 Manual Review

Predictions should be used as supporting information for analysts rather than automatic lending decisions.

## 11.3 Threshold Optimization

The classification threshold should be evaluated against the actual financial costs of false positives and false negatives.

## 11.4 Repayment Behavior Analysis

Historical repayment patterns deserve particular attention because repayment-status features were among the strongest predictive variables.

## 11.5 Model Monitoring

If the model is eventually deployed, its performance should be monitored for changes in customer behavior and data distributions.

## 11.6 Further Validation

External validation, probability calibration, and more comprehensive fairness analysis should be completed before real-world adoption.

---

# 12. Project Limitations

The project has several limitations.

1. **Historical data:** The dataset may not reflect current customer behavior or financial conditions.
2. **Geographic limitations:** The dataset represents a particular historical customer population in Taiwan.
3. **Class imbalance:** Default observations represent a minority of the dataset.
4. **False negatives:** The selected model misses a meaningful proportion of actual defaults.
5. **Calibration:** Probability calibration has not been established.
6. **Fairness:** Subgroup comparisons are exploratory and do not constitute comprehensive fairness validation.
7. **External validation:** The model has not been evaluated on an independent external dataset.
8. **Temporal validation:** Performance stability over time has not been established.
9. **Production readiness:** The project does not include a production lending integration or operational monitoring system.

---

# 13. Future Scope

The following improvements could strengthen the project:

* Validate the model using newer and independent datasets.
* Perform temporal validation.
* Apply probability calibration techniques.
* Optimize thresholds using business-specific cost functions.
* Explore additional feature engineering techniques.
* Perform more extensive hyperparameter optimization.
* Expand fairness and subgroup analysis.
* Develop model monitoring and drift detection.
* Assess deployment requirements and operational constraints.

These are proposed future improvements and have not been implemented as part of the current project.

---

# 14. Final Conclusion

This project demonstrates an end-to-end application of data science and machine learning to credit card default prediction.

The workflow covered dataset understanding, exploratory analysis, preprocessing, model development, performance evaluation, interpretability, customer risk segmentation, and robustness validation.

Among the evaluated classification models, the Random Forest Classifier was selected as the final model with a decision threshold of 0.55.

It achieved an accuracy of 80.15%, an F1-score of 54.28%, and an ROC-AUC of 77.62% on the held-out test set.

The analysis identified historical repayment behavior as an important source of predictive information and demonstrated differences in observed default rates across illustrative customer risk segments.

Although the results demonstrate the potential of machine learning for supporting credit-risk analysis, the model is not production-ready. Additional external validation, calibration, fairness evaluation, and operational testing are necessary before real-world use.

The project concludes with a documented analytical workflow and a set of evidence-based recommendations for future development.

---

# 15. Project Completion

The capstone consists of eight notebooks:

| Notebook | Title                                        |
| -------- | -------------------------------------------- |
| 01       | Data Understanding                           |
| 02       | Exploratory Data Analysis                    |
| 03       | Data Preprocessing                           |
| 04       | Model Building and Evaluation                |
| 05       | Model Interpretability and Business Insights |
| 06       | Final Credit Risk Analysis                   |
| 07       | Model Robustness and Validation              |
| 08       | Final Project Report and Executive Summary   |

**Project Status:** Analytical notebook series completed.

**Final Model:** Random Forest Classifier.

**Decision Threshold:** 0.55.
