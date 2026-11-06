"""Own-patient and ICU-wide past-resistance features.

For a culture collected on day ``d``, only earlier cultures in the closed
interval of 21 days to 48 hours before ``d`` are visible. That lag is the
time an antibiogram needs to come back. A resistant culture contributes
``exp(-lambda * days)`` and a susceptible culture contributes 0. The
own-patient feature ``p&`` is the maximum contribution. The ICU feature
``r&`` sums, over every other patient, that patient's maximum contribution.
"""

import math

import numpy as np
import pandas as pd

from amr_icu.config import ANTIBIOGRAM_LAG_DAYS, DECAY_LAMBDA, FAMILIES, PAST_WINDOW_DAYS

_RESISTANT = {"r", "resistant"}
_SUSCEPTIBLE = {"s", "susceptible", "sensitive", "sensible"}


def parse_resistance(value) -> int | None:
    """Map a cell to 1 (resistant), 0 (susceptible), or None when untested."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "zero", "na"}:
        return None
    if text in _RESISTANT:
        return 1
    if text in _SUSCEPTIBLE:
        return 0
    raise ValueError(
        f"Unrecognised resistance value {value!r}. Use 'r' or 's'."
    )


def negative_exponential(days: float, resistant: bool, lam: float = DECAY_LAMBDA) -> float:
    """Contribution of one past culture. Susceptible cultures contribute nothing."""
    if not resistant:
        return 0.0
    return math.exp(-lam * days)


def add_temporal_features(frame: pd.DataFrame, lam: float = DECAY_LAMBDA) -> pd.DataFrame:
    """Add ``p&`` and ``r&`` columns for every antibiotic family.

    A family that was not tested on a past culture is ignored for that
    feature. If the window contains no tested culture, the feature is left
    missing; preprocessing later replaces that missing value with 0.
    """
    out = frame.copy()
    dates = pd.to_datetime(out["culture_date"]).dt.normalize().to_numpy()
    patients = out["patient_id"].astype(str).to_numpy()
    n_rows = len(out)
    families = [family for family in FAMILIES if f"c&{family}" in out.columns]

    for family in families:
        labels = [parse_resistance(value) for value in out[f"c&{family}"].tolist()]
        own = np.full(n_rows, np.nan)
        icu = np.full(n_rows, np.nan)
        for index in range(n_rows):
            own_best: float | None = None
            per_patient: dict[str, float] = {}
            for other in range(n_rows):
                label = labels[other]
                if other == index or label is None:
                    continue
                days = _days_between(dates[index], dates[other])
                if days < ANTIBIOGRAM_LAG_DAYS or days > PAST_WINDOW_DAYS:
                    continue
                contribution = negative_exponential(days, resistant=label == 1, lam=lam)
                if patients[other] == patients[index]:
                    own_best = contribution if own_best is None else max(own_best, contribution)
                else:
                    previous = per_patient.get(patients[other])
                    per_patient[patients[other]] = (
                        contribution if previous is None else max(previous, contribution)
                    )
            if own_best is not None:
                own[index] = own_best
            if per_patient:
                icu[index] = float(sum(per_patient.values()))
        out[f"p&{family}"] = own
        out[f"r&{family}"] = icu
    return out


def _days_between(later, earlier) -> int:
    delta = (later - earlier) / np.timedelta64(1, "D")
    return int(delta)
