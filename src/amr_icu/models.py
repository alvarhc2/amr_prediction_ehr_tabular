"""Logistic regression and random forest, as used in the study.

Logistic regression is the baseline. Its predictors are standardized on the
training window only. Random forest is the second classifier. It receives
the unscaled matrix. Neither model rebalances the classes: the paper leaves
oversampling as later work, because the class ratio itself moves with time.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from amr_icu.config import RANDOM_STATE, RF_TREES


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    method: str,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if method == "lr":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
    elif method == "rf":
        model = RandomForestClassifier(
            n_estimators=RF_TREES,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    else:
        raise KeyError(f"Unknown method {method!r}. Use 'lr' or 'rf'.")
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model.predict(x_test)
