# Credit Risk Analytics — Predicting Credit Card Default

A machine learning project focused on predicting credit card payment defaults using historical customer information, repayment behavior, billing statements, and payment records.

The project follows an end-to-end data science workflow, covering data understanding, exploratory data analysis, preprocessing, model development, evaluation, interpretability, business insights, and model robustness validation.

## Project Overview

Credit card default is a significant risk for financial institutions. Identifying customers who may fail to make future payments can help institutions improve risk monitoring and make more informed credit-risk assessments.

This project develops and evaluates machine learning classification models to predict whether a customer will default on their credit card payment in the following month.

The primary objective is to identify meaningful patterns in historical customer data and evaluate how machine learning can support credit-risk analysis.

## Objectives

* Understand and analyze the credit card default dataset.
* Perform exploratory data analysis to identify relevant patterns.
* Clean and preprocess data for machine learning.
* Develop and compare multiple classification models.
* Evaluate model performance using appropriate classification metrics.
* Interpret important predictive features.
* Develop illustrative customer risk segments.
* Assess model robustness through cross-validation.
* Examine model performance across demographic subgroups.
* Translate analytical findings into business recommendations.

## Dataset

**Dataset:** Default of Credit Card Clients

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains 30,000 customer records and 23 predictor variables, along with a binary target variable.

### Target Variable

`default payment next month`

| Value | Meaning    |
| ----- | ---------- |
| 0     | No default |
| 1     | Default    |

### Target Distribution

| Class          |    Records | Percentage |
| -------------- | ---------: | ---------: |
| No Default (0) |     23,364 |     77.88% |
| Default (1)    |      6,636 |     22.12% |
| **Total**      | **30,000** |   **100%** |

The dataset includes demographic information, credit limits, repayment status, bill statements, and previous payment amounts.

## Technology Stack

* **Programming Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Notebook Environment:** Jupyter Notebook
* **Version Control:** Git and GitHub

## Repository Structure

```text
Data-Science-Capstone/
│
├── data/
│   └── default of credit card clients.xls
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_model_building_and_evaluation.ipynb
│   ├── 05_model_interpretability_and_business_insights.ipynb
│   ├── 06_final_credit_risk_analysis.ipynb
│   ├── 07_model_robustness_and_validation.ipynb
│   └── 08_final_project_report_and_executive_summary.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
├── FINAL_REPORT.md
└── requirements.txt
```

**Note:** The dataset is required to execute the notebooks but is excluded from version control. Place the downloaded dataset inside the `data/` directory using the filename shown above.

## Project Workflow

### 1. Data Understanding

* Dataset inspection
* Data dimensions and structure
* Feature descriptions
* Data types and missing-value analysis
* Target variable examination

### 2. Exploratory Data Analysis

* Target class distribution
* Demographic analysis
* Repayment behavior
* Credit limit distribution
* Billing and payment patterns
* Correlation analysis

### 3. Data Preprocessing

* Data cleaning
* Categorical feature handling
* Feature encoding
* Numerical scaling
* Yeo-Johnson transformation
* Stratified train-test splitting
* Prevention of data leakage

### 4. Model Building and Evaluation

The following classification models were evaluated:

* Dummy Classifier
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

Model performance was assessed using accuracy, precision, recall, F1-score, and ROC-AUC.

### 5. Model Interpretability and Business Insights

* Permutation feature importance
* Identification of influential repayment variables
* Customer risk segmentation
* Business interpretation of model findings

### 6. Final Credit Risk Analysis

* Final model evaluation
* Confusion matrix analysis
* False-positive and false-negative examination
* Business recommendations

### 7. Model Robustness and Validation

* Stratified five-fold cross-validation
* Out-of-fold evaluation
* Brier score and calibration analysis
* Gender-based subgroup evaluation
* Age-based subgroup evaluation
* Held-out test performance verification

### 8. Final Project Report and Executive Summary

* Consolidated project findings
* Model performance summary
* Business implications
* Limitations
* Future improvements
* Final conclusion

## Final Model

The selected model is the **Random Forest Classifier** with a decision threshold of **0.55**.

The threshold was selected using validation data rather than the held-out test set.

### Held-Out Test Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8015 |
| Precision | 0.5532 |
| Recall    | 0.5328 |
| F1-Score  | 0.5428 |
| ROC-AUC   | 0.7762 |

### Confusion Matrix

|                   | Predicted No Default | Predicted Default |
| ----------------- | -------------------: | ----------------: |
| Actual No Default |                4,102 |               571 |
| Actual Default    |                  620 |               707 |

The results indicate that the model can identify meaningful credit-risk patterns. However, it still misses a considerable proportion of actual defaults, so its predictions should not be treated as definitive lending decisions.

## Cross-Validation Results

The selected Random Forest model was evaluated using stratified five-fold cross-validation.

| Metric    |   Mean | Standard Deviation |
| --------- | -----: | -----------------: |
| Accuracy  | 0.7838 |             0.0032 |
| Precision | 0.5100 |             0.0064 |
| Recall    | 0.5818 |             0.0088 |
| F1-Score  | 0.5435 |             0.0054 |
| ROC-AUC   | 0.7825 |             0.0065 |

The relatively small variation across folds indicates consistent performance under the cross-validation procedure. This does not replace external or temporal validation.

## Key Findings

* Repayment history variables were among the strongest predictive features.
* `PAY_0` and `PAY_2` were particularly influential in the permutation importance analysis.
* Payment amount variables also contributed to prediction.
* The high-risk segment had a substantially higher observed default rate than the low-risk segment.
* Cross-validation results showed relatively low variation across folds.
* Additional calibration, fairness, and external validation are needed before considering real-world deployment.

## Business Applications

Potential applications include:

* Prioritizing accounts for manual credit-risk review.
* Supporting targeted risk monitoring.
* Identifying customers who may require additional attention.
* Supporting data-driven credit-risk assessment.
* Helping financial institutions investigate repayment behavior.

The model is an analytical decision-support prototype. It is not designed to automatically approve or reject credit applications.

## Installation and Setup

### Prerequisites

* Python 3.13 or a compatible Python environment
* Git
* Jupyter Notebook or JupyterLab

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

```bash
cd Data-Science-Capstone
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the actual repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the Dataset

Download the Default of Credit Card Clients dataset from the UCI Machine Learning Repository.

Place the Excel file inside:

```text
data/default of credit card clients.xls
```

### 6. Launch Jupyter

```bash
jupyter lab
```

Open the notebooks in numerical order, beginning with Notebook 01.

## Important Limitations

* The project uses a historical dataset and may not represent current credit markets.
* The dataset is geographically and temporally limited.
* The target classes are imbalanced.
* The selected model still produces false positives and false negatives.
* Probability calibration has not been established.
* Subgroup analysis is exploratory and does not constitute comprehensive fairness validation.
* External and temporal validation have not been completed.
* The model has not been deployed in a production lending environment.

## Future Improvements

* External validation using newer credit datasets.
* Temporal validation to assess performance over time.
* Probability calibration.
* Cost-sensitive threshold optimization.
* Advanced hyperparameter tuning.
* Expanded fairness evaluation.
* Model monitoring and drift detection.
* Development of a deployment-ready prediction interface.

## Conclusion

This project demonstrates an end-to-end machine learning approach to credit card default prediction, from raw data analysis to model evaluation and business interpretation.

The Random Forest model achieved an ROC-AUC of 0.7762 and an F1-score of 0.5428 on the held-out test set at the selected decision threshold.

The findings demonstrate the potential of machine learning for supporting credit-risk analysis while highlighting the need for further validation before real-world application.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

* YuvaIntern — Internship project context.
* UCI Machine Learning Repository — Dataset source.
* Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn — Open-source tools used in the project.
