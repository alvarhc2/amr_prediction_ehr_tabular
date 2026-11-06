"""Monthly evaluation windows.

Every test window is one calendar month, starting in January 2016 and
stepping forward without overlap. A sliding training window covers the four
years placed 0, 2 or 4 years before that month. An incremental training
window contains every earlier culture and grows by one month at each step.

When the training window touches the test window, cultures from the final
48 hours are left out of training: their antibiogram would not yet be known.
A patient who has a culture in the test month is removed from training, so
the model is not scored on someone it has already seen.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amr_icu.config import ANTIBIOGRAM_LAG_DAYS, TEST_START, TRAIN_YEARS, ModelSpec


@dataclass(frozen=True)
class Window:
    train_index: np.ndarray
    test_index: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp
    # Last culture inside the training interval, before patients are removed.
    # Instance weights are measured against this date.
    reference_date: pd.Timestamp | None


def iter_windows(
    culture_dates: pd.Series,
    patient_ids: pd.Series,
    spec: ModelSpec,
) -> list[Window]:
    """Build every non-empty test month for one model."""
    dates = pd.to_datetime(culture_dates).dt.normalize().reset_index(drop=True)
    patients = patient_ids.astype(str).reset_index(drop=True)
    if dates.empty:
        return []

    windows: list[Window] = []
    test_start = pd.Timestamp(TEST_START)
    last_month = pd.Timestamp(year=dates.max().year, month=dates.max().month, day=1)
    while test_start <= last_month:
        test_end = test_start + pd.DateOffset(months=1)
        train_start, train_end = _training_bounds(test_start, spec)
        test_mask = (dates >= test_start) & (dates < test_end)
        train_mask = dates < train_end
        if train_start is not None:
            train_mask = train_mask & (dates >= train_start)

        test_index = np.flatnonzero(test_mask.to_numpy())
        if len(test_index):
            in_test = set(patients.iloc[test_index])
            candidate = np.flatnonzero(train_mask.to_numpy())
            train_index = np.asarray(
                [index for index in candidate if patients.iloc[index] not in in_test],
                dtype=int,
            )
            reference = dates.loc[train_mask].max() if bool(train_mask.any()) else None
            if reference is not None and pd.isna(reference):
                reference = None
            windows.append(
                Window(
                    train_index=train_index,
                    test_index=test_index,
                    test_start=test_start,
                    test_end=test_end,
                    train_start=train_start,
                    train_end=train_end,
                    reference_date=None if reference is None else pd.Timestamp(reference),
                )
            )
        test_start = test_end
    return windows


def instance_weights(culture_dates: pd.Series, window: Window, lam: float) -> np.ndarray:
    """Weight each training culture by ``exp(-lam * days before the last one)``."""
    if window.reference_date is None or len(window.train_index) == 0:
        return np.ones(len(window.train_index), dtype=float)
    dates = pd.to_datetime(culture_dates).dt.normalize().reset_index(drop=True)
    days = (window.reference_date - dates.iloc[window.train_index]).dt.days.to_numpy(dtype=float)
    return np.exp(-lam * days)


def _training_bounds(test_start: pd.Timestamp, spec: ModelSpec) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    if spec.mode == "incremental":
        return None, test_start - pd.Timedelta(days=ANTIBIOGRAM_LAG_DAYS)
    nominal_end = test_start - pd.DateOffset(years=spec.gap_years)
    train_start = nominal_end - pd.DateOffset(years=TRAIN_YEARS)
    if spec.gap_years == 0:
        train_end = test_start - pd.Timedelta(days=ANTIBIOGRAM_LAG_DAYS)
    else:
        train_end = nominal_end
    return train_start, train_end
