"""A synthetic ICU cohort with the same columns as the study.

The rows are invented. Resistance becomes more common in later years, and a
patient who was recently resistant is more likely to be resistant again, so
the past-culture features have something to learn. Polymyxin resistance stays
uncommon, and two families follow the testing calendar of the paper: polymyxins
are absent before 2007 and quinolones are absent from 2018 on. Nothing here
is a patient, and the metrics are not the published ones.
"""

import numpy as np
import pandas as pd

from amr_icu.config import FAMILIES
from amr_icu.schema import GOI_COLUMNS

SEED = 7

_ORIGINS = ("surgery", "emergency", "internal_medicine", "urology")
_REASONS = ("serious_infection", "respiratory_failure", "hypovolaemia", "other")
_CATEGORIES = ("medical", "surgical")
_CULTURES = {
    "sputum": ("respiratory", "clinical"),
    "bronchoaspirate": ("respiratory", "clinical"),
    "blood": ("liquids", "clinical"),
    "urine": ("liquids", "clinical"),
    "exudate": ("surface", "surface"),
}


def make_cohort(n_patients: int = 140, seed: int = SEED) -> pd.DataFrame:
    """Build a fake cohort spanning July 2004 to May 2019."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2004-07-01")
    end = pd.Timestamp("2019-05-15")
    span_days = int((end - start).days)
    rows: list[dict] = []

    for index in range(n_patients):
        admit = start + pd.Timedelta(days=int(rng.integers(0, span_days - 40)))
        stay = int(rng.integers(5, 22))
        discharge = admit + pd.Timedelta(days=stay)
        if discharge > end + pd.Timedelta(days=5):
            continue
        goi = (rng.random(len(GOI_COLUMNS)) < 0.28).astype(int)
        patient = {
            "patient_id": f"P{index:04d}",
            "admission_id": f"A{index:04d}",
            "admission_start": admit,
            "admission_end": discharge,
            "age": int(rng.integers(40, 87)),
            "gender": "m" if rng.random() < 0.7 else "f",
            "origin": rng.choice(_ORIGINS),
            "reason_admission": rng.choice(_REASONS),
            "patient_category": rng.choice(_CATEGORIES),
        }
        for letter, value in zip("ABCDEFG", goi):
            patient[f"goi_{letter}"] = int(value)

        culture_name = str(rng.choice(list(_CULTURES)))
        group1, group2 = _CULTURES[culture_name]
        n_cultures = int(rng.integers(1, 4))
        day_offsets = np.arange(2, stay)
        offsets = sorted(int(value) for value in rng.choice(day_offsets, size=n_cultures, replace=False))
        recent_resistant: dict[str, pd.Timestamp | None] = {family: None for family in FAMILIES}
        for culture_number, offset in enumerate(offsets):
            culture_date = admit + pd.Timedelta(days=int(offset))
            row = dict(patient)
            row["culture_id"] = f"C{index:04d}-{culture_number}"
            row["culture_date"] = culture_date
            row["culture_type"] = culture_name
            row["culture_type_group1"] = group1
            row["culture_type_group2"] = group2
            year_frac = (culture_date - start).days / span_days
            for family in FAMILIES:
                row[f"c&{family}"] = _family_result(
                    rng, family, culture_date, year_frac, recent_resistant[family]
                )
                if row[f"c&{family}"] == "r":
                    recent_resistant[family] = culture_date
            rows.append(row)

    cohort = pd.DataFrame(rows)
    return cohort.sort_values(["culture_date", "culture_id"]).reset_index(drop=True)


def _family_result(rng, family: str, culture_date: pd.Timestamp, year_frac: float, last_resistant) -> str:
    if family == "pol" and culture_date < pd.Timestamp("2007-01-01"):
        return ""
    if family == "qui" and culture_date >= pd.Timestamp("2018-01-01"):
        return ""
    recent = (
        last_resistant is not None
        and 0 < (culture_date - last_resistant).days <= 21
    )
    if family == "pol":
        probability = 0.12 + 0.08 * year_frac
    elif family == "car":
        probability = 0.20 + 0.65 * year_frac + (0.25 if recent else 0.0)
    else:
        probability = 0.15 + 0.55 * year_frac + (0.30 if recent else 0.0)
    return "r" if rng.random() < min(probability, 0.95) else "s"
