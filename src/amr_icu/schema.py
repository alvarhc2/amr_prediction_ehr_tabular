"""Column contract for a cohort file.

A cohort is one row per culture. Resistance columns use ``r`` and ``s``.
An empty cell means that family was not tested on that culture. Past-resistance
features are computed by the package, so a cohort file does not include them.

The hospital extract behind the paper is not part of this repository.
"""

from pathlib import Path

import pandas as pd

from amr_icu.config import FAMILIES

TARGETS: tuple[str, ...] = tuple(f"c&{family}" for family in FAMILIES)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "culture_id",
    "culture_date",
    "admission_id",
    "admission_start",
    "admission_end",
    "age",
    "gender",
    "origin",
    "reason_admission",
    "patient_category",
    "goi_A",
    "goi_B",
    "goi_C",
    "goi_D",
    "goi_E",
    "goi_F",
    "goi_G",
    "culture_type",
    "culture_type_group1",
    "culture_type_group2",
    *TARGETS,
)

GOI_COLUMNS: tuple[str, ...] = tuple(f"goi_{letter}" for letter in "ABCDEFG")

# Carried through the modeling frame but never shown to a classifier.
# culture_date is the timestamp used to build windows. date_culture is the
# numeric copy that a model may use as a feature.
META_COLUMNS: tuple[str, ...] = ("patient_id", "culture_id", "culture_date", "target")


def read_cohort(path: str | Path) -> pd.DataFrame:
    """Load a cohort CSV and check that the published feature set can be built."""
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "The cohort file is missing columns required by the study: "
            + ", ".join(missing)
        )
    return frame
