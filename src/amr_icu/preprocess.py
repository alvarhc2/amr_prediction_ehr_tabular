"""Turn a cohort into the numeric table used by the classifiers.

The steps follow the study. Cultures from stays shorter than 48 hours are
removed. Calendar fields are derived from the admission and culture dates.
Categorical fields become integer codes. The two date fields become the
number of days since the earliest date of that field. Missing past-resistance
features are filled with 0, because an absent test in the look-back window
was read as no evidence of a resistant bacterium. Features with Pearson
correlation above 0.9 or below -0.9 are reduced to one column per group.
"""

import pandas as pd

from amr_icu.config import CORRELATION_THRESHOLD, FAMILIES, MIN_STAY_HOURS, PREFERRED_FEATURES
from amr_icu.features import parse_resistance
from amr_icu.schema import GOI_COLUMNS, TARGETS

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "gender",
    "origin",
    "reason_admission",
    "patient_category",
    "culture_type",
    "culture_type_group1",
    "culture_type_group2",
)


def prepare_family(cohort: pd.DataFrame, family: str) -> pd.DataFrame:
    """Build the modeling frame for one antibiotic family.

    The returned frame keeps ``patient_id``, ``culture_id`` and
    ``culture_date`` for the temporal split, a binary ``target``
    (1 resistant, 0 susceptible), and the numeric predictors.
    """
    target_column = f"c&{family}"
    if target_column not in cohort.columns:
        raise KeyError(f"Cohort has no target column {target_column}.")
    missing_past = [f"{kind}&{name}" for kind in ("p", "r") for name in FAMILIES if f"{kind}&{name}" not in cohort.columns]
    if missing_past:
        raise ValueError(
            "Past-resistance columns are missing ("
            + ", ".join(missing_past[:4])
            + "). Call add_temporal_features before prepare_family."
        )

    frame = cohort.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["culture_id"] = frame["culture_id"].astype(str)
    frame["culture_date"] = pd.to_datetime(frame["culture_date"])
    frame["admission_start"] = pd.to_datetime(frame["admission_start"])
    frame["admission_end"] = pd.to_datetime(frame["admission_end"])

    stay = frame["admission_end"] - frame["admission_start"]
    long_enough = stay >= pd.Timedelta(hours=MIN_STAY_HOURS)
    frame = frame.loc[long_enough].copy()

    frame["target"] = frame[target_column].map(parse_resistance)
    frame = frame.loc[frame["target"].notna()].copy()
    frame["target"] = frame["target"].astype(int)
    if frame.empty:
        raise ValueError(f"No cultures with a tested {family.upper()} result and a stay of at least 48 hours.")

    frame = frame.sort_values(["culture_date", "culture_id"]).reset_index(drop=True)
    frame = _derive_fields(frame)
    frame = _encode(frame)
    feature_columns = _feature_columns(frame)
    kept = _drop_correlated(frame[feature_columns])
    model = frame[["patient_id", "culture_id", "culture_date", "target"]].copy()
    for column in kept.columns:
        model[column] = kept[column].to_numpy()
    return model


def _derive_fields(frame: pd.DataFrame) -> pd.DataFrame:
    culture = frame["culture_date"]
    admission = frame["admission_start"]
    frame["days_to_culture"] = (culture.dt.normalize() - admission.dt.normalize()).dt.days.astype(float)
    frame["date_culture"] = culture
    frame["day_week_culture"] = culture.dt.dayofweek.astype(float)
    frame["day_month_culture"] = culture.dt.day.astype(float)
    frame["month_culture"] = culture.dt.month.astype(float)
    frame["year_culture"] = culture.dt.year.astype(float)
    frame["start_date"] = admission
    frame["day_week_admission"] = admission.dt.dayofweek.astype(float)
    frame["day_month_admission"] = admission.dt.day.astype(float)
    frame["month_admission"] = admission.dt.month.astype(float)
    frame["year_admission"] = admission.dt.year.astype(float)
    frame["pluripathology"] = frame.loc[:, GOI_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    for column in frame.columns:
        if column.startswith("p&") or column.startswith("r&"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def _encode(frame: pd.DataFrame) -> pd.DataFrame:
    for column in CATEGORICAL_COLUMNS:
        frame[column] = _integer_codes(frame[column])
    for column in ("date_culture", "start_date"):
        stamps = pd.to_datetime(frame[column])
        origin = stamps.min()
        frame[column] = (stamps.dt.normalize() - origin.normalize()).dt.days.astype(float)
    for column in GOI_COLUMNS + ("age",):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def _integer_codes(series: pd.Series) -> pd.Series:
    """Encode categories by sorted label, so the codes do not depend on row order."""
    labels = series.astype(str)
    ordered = sorted(labels.unique())
    mapping = {label: float(index) for index, label in enumerate(ordered)}
    return labels.map(mapping).astype(float)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = set(TARGETS) | {
        "patient_id",
        "culture_id",
        "culture_date",
        "admission_id",
        "admission_start",
        "admission_end",
        "target",
    }
    return [column for column in frame.columns if column not in blocked]


def _drop_correlated(features: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant predictors, keeping one column from each correlated group."""
    correlation = features.corr(numeric_only=True).fillna(0.0)
    parent = {column: column for column in features.columns}

    def find(column: str) -> str:
        while parent[column] != column:
            parent[column] = parent[parent[column]]
            column = parent[column]
        return column

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    columns = list(features.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            coefficient = correlation.loc[left, right]
            if abs(coefficient) > CORRELATION_THRESHOLD:
                union(left, right)

    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(find(column), []).append(column)

    dropped: set[str] = set()
    for members in groups.values():
        if len(members) == 1:
            continue
        preferred = [column for column in PREFERRED_FEATURES if column in members]
        keeper = preferred[0] if preferred else members[0]
        dropped.update(column for column in members if column != keeper)
    return features.drop(columns=sorted(dropped))
