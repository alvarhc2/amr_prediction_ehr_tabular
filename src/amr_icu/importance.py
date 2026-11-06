"""Feature rankings used in the paper: mutual information and random forest.

Mutual information scores each column on its own. Random-forest importance
is computed from one forest fitted on every culture before January 2016,
which is the first training window of the incremental scheme. The two
rankings disagree on purpose: mutual information highlights the culture date
and the ICU-wide features, while the forest, which sees the columns together,
puts the patient's own recent antibiogram first.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from amr_icu.config import RANDOM_STATE, RF_TREES, TEST_START
from amr_icu.schema import META_COLUMNS


def rank_features(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return mutual-information and random-forest scores, highest first."""
    columns = [column for column in frame.columns if column not in META_COLUMNS]
    before = pd.to_datetime(frame["culture_date"]) < pd.Timestamp(TEST_START)
    train = frame.loc[before]
    if train["target"].nunique() < 2:
        raise ValueError("Feature ranking needs both classes before the first test month.")
    matrix = train.loc[:, columns].to_numpy(dtype=float)
    target = train["target"].to_numpy(dtype=int)
    mutual = pd.Series(
        mutual_info_classif(matrix, target, random_state=RANDOM_STATE),
        index=columns,
    ).sort_values(ascending=False)
    forest = RandomForestClassifier(n_estimators=RF_TREES, random_state=RANDOM_STATE, n_jobs=1)
    forest.fit(matrix, target)
    importance = pd.Series(forest.feature_importances_, index=columns).sort_values(ascending=False)
    return mutual, importance


def top_table(scores: pd.Series, k: int = 5) -> pd.DataFrame:
    """Small table of the highest scores, for printing and for the README."""
    head = scores.head(k)
    return pd.DataFrame({"feature": head.index, "score": np.round(head.to_numpy(), 4)})
