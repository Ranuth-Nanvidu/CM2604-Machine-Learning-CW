"""Encoding and scaling for the Telco Customer Churn models.

* Encoding (one-hot) is needed by all three models: scikit-learn only accepts numbers.
* Scaling (StandardScaler) is needed by KNN (distance-based) and Logistic
  Regression (regularisation). Random Forest splits on thresholds, so it does
  not need scaling -> use scale=False.

Input: the unencoded train.csv / test.csv from 02_preprocessing.ipynb.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def build_preprocessor(scale=True):
    """Create an (unfitted) ColumnTransformer.

    drop="if_binary" keeps one 0/1 column for Yes/No features.
    handle_unknown="ignore" avoids errors if an unseen category appears.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if scale else "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False), CAT_COLS),
        ],
        verbose_feature_names_out=False,
    )


def encode(X_train, X_test, scale=True, verbose=True):
    """Fit the preprocessor on the training set ONLY, then apply it to the test set.

    The encoded sets are DataFrames with readable column names such as "Contract_Two year".
    """
    preprocessor = build_preprocessor(scale=scale).set_output(transform="pandas")

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    if verbose:
        print(f"Scaling applied         : {scale}")
        print(f"Features before encoding: {X_train.shape[1]}")
        print(f"Features after encoding : {X_train_prep.shape[1]}")
        print()
        print(list(preprocessor.get_feature_names_out()))

    return X_train_prep, X_test_prep, preprocessor