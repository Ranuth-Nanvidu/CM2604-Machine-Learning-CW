# Telco Customer Churn Prediction

![Last Commit](https://img.shields.io/github/last-commit/Ranuth-Nanvidu/REPO-NAME?style=for-the-badge&logo=github&logoColor=white)

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/Managed%20with-uv-6F42C1?style=for-the-badge&logo=uv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/JupyterLab-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

Predicting whether a telecom customer will leave (churn) using **K-Nearest Neighbours**, **Logistic Regression** and **Random Forest**, with exploratory data analysis, preprocessing, hyperparameter tuning and model comparison.

Coursework for **CM2604 Machine Learning** (Part 01, Individual Assignment), BSc (Hons) Artificial Intelligence and Data Science, Informatics Institute of Technology (IIT), in collaboration with Robert Gordon University (RGU), Aberdeen.

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data, via Kaggle)

- 7,043 customers, 21 columns: 19 features, a customer ID and the target (`Churn`)
- **Target:** `Churn` (Yes / No), imbalanced at **26.5% churn** vs 73.5% stayed
- **Features:** demographics (gender, senior citizen, partner, dependents), account details (tenure, contract, billing, payment method, charges) and subscribed services (phone, internet, security, support, streaming)

---

## Repository structure

```
.
├── data/
│   ├── raw/                    # original Kaggle CSV
│   └── processed/              # cleaned data and the train/test split
├── figures/                    # plots saved by the notebooks
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory data analysis
│   ├── 02_preprocessing.ipynb  # cleaning and train/test split
│   └── 03_modelling.ipynb      # tuning, comparison and evaluation
├── src/
│   └── cm2604_ml_cw/
│       ├── __init__.py
│       ├── preprocessing.py    # encode(): one-hot encoding + optional scaling
│       └── evaluation.py       # CV setup, metrics, result tables, plots
├── pyproject.toml
└── README.md
```

---

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) to manage the environment.

```bash
# 1. Clone the repository
git clone https://github.com/Ranuth-Nanvidu/CM2604-Machine-Learning-CW.git
cd CM2604-Machine-Learning-CW

# 2. Create the environment and install dependencies
uv sync

# 3. Start Jupyter
uv run jupyter lab
```

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle into `data/raw/`, then run the notebooks **in order**:

1. `01_eda.ipynb`
2. `02_preprocessing.ipynb` (creates `data/processed/train.csv` and `test.csv`)
3. `03_modelling.ipynb`

All random processes use a fixed `random_state`, so results are reproducible.

**Main libraries:** pandas, NumPy, Matplotlib, seaborn, scikit-learn

---

## Workflow

### 1. Exploratory data analysis (`01_eda.ipynb`)

- Data quality checks: types, missing values, duplicates
- Class balance, distributions of numerical features, churn rate per category
- Feature interactions (tenure × contract, contract × internet service)
- Correlation analysis

### 2. Preprocessing (`02_preprocessing.ipynb`)

- `TotalCharges` converted to numeric. Its 11 blank values all belong to customers with `tenure = 0` (never billed), so they are set to 0
- `customerID` dropped; `SeniorCitizen` mapped from 0/1 to Yes/No; `Churn` encoded as 1/0
- Duplicate, outlier and redundant-category checks
- Stratified 80/20 train/test split (5,634 / 1,409 customers)

### 3. Modelling (`03_modelling.ipynb`)

- **Encoding and scaling** with `src/cm2604_ml_cw/preprocessing.py`, fitted on the training set only:
  - one-hot encoding for all models (19 features become 40)
  - standard scaling for KNN and Logistic Regression only; Random Forest does not need it
- **Class imbalance:** `class_weight="balanced"` for Logistic Regression and Random Forest, compared against no weighting
- **Hyperparameter tuning:** `GridSearchCV` with stratified 5-fold cross-validation on the training set, optimising **F1 for the churn class**
- **Final evaluation:** a single run on the held-out test set, compared with a majority-class baseline

---

## Results

Test-set performance of the tuned models (churn = positive class):

| Model | Best hyperparameters | CV F1 | Test F1 | Test recall | Test precision | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 200 trees, no depth limit, min leaf 20, log2, balanced | **0.632** | **0.652** | 0.802 | 0.548 | **0.858** | **0.669** |
| Logistic Regression | C = 10, L2, balanced | 0.628 | 0.637 | **0.810** | 0.525 | 0.856 | 0.665 |
| K-Nearest Neighbours | k = 71, uniform, Manhattan | 0.600 | 0.620 | 0.610 | **0.630** | 0.850 | 0.641 |
| Baseline (majority class) | – | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |

### Key findings

- **Accuracy is misleading on this data.** Always predicting "no churn" scores 73.5% accuracy but catches no churners, so models were compared on F1, recall and AUC.
- **Class weighting mattered most.** `class_weight="balanced"` raised recall by about 0.26–0.28 and improved F1 for Logistic Regression and Random Forest. KNN cannot use class weights, so its recall stays lower.
- **Tuning controlled overfitting.** Random Forest needed a large minimum leaf size, and KNN needed a large k. Logistic Regression barely overfits at any setting.
- **Random Forest performed best** in both cross-validation and on the test set, catching 300 of 374 churners. Logistic Regression is a close, more interpretable alternative.
- Churn is driven mainly by **contract type, tenure, charges, internet service type, payment method and the lack of security/support add-ons**.

---

## Author

**Ranuth Nanvidu**[(@Ranuth-Nanvidu)](https://github.com/Ranuth-Nanvidu),

BSc (Hons) Artificial Intelligence and Data Science, IIT / Robert Gordon University.