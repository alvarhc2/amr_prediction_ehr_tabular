"""Constants and the fourteen models reported in the paper."""

from dataclasses import dataclass


FAMILIES: tuple[str, ...] = ("amg", "car", "cf4", "pap", "pol", "qui")

FAMILY_NAMES: dict[str, str] = {
    "amg": "Aminoglycosides",
    "car": "Carbapenems",
    "cf4": "4th generation cephalosporins",
    "pap": "Extended-spectrum penicillins",
    "pol": "Polymyxins",
    "qui": "Quinolones",
}

# Negative exponential used to decay a past resistant culture.
# The study set lambda experimentally to 0.095. The original scripts used
# 1.1 ** (-days), and ln(1.1) is approximately 0.0953.
DECAY_LAMBDA: float = 0.095

# A past culture counts only when its antibiogram would already be back,
# and only while clinicians still treat a positive result as current.
ANTIBIOGRAM_LAG_DAYS: int = 2
PAST_WINDOW_DAYS: int = 21

# Patients who stayed in the ICU for less than this are not modeled.
MIN_STAY_HOURS: int = 48

# |Pearson r| above this marks a redundant feature. The study used 0.9.
CORRELATION_THRESHOLD: float = 0.9

# First month predicted in every experiment. Training never uses this month
# or anything after it in the first step.
TEST_START: str = "2016-01-01"
TRAIN_YEARS: int = 4

# Grid searched for the instance-weight decay. lambda = 0 leaves every
# training culture with weight 1, which recovers the unweighted window.
WEIGHT_LAMBDAS: tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)

RANDOM_STATE: int = 0
RF_TREES: int = 100

# When several date features exceed the correlation threshold, keep the
# culture date. It is the feature the study retained.
PREFERRED_FEATURES: tuple[str, ...] = ("date_culture",)


@dataclass(frozen=True)
class ModelSpec:
    """One of the models M1–M14.

    Sliding models use a training window of ``TRAIN_YEARS`` placed
    ``gap_years`` before the test month. Incremental models grow from the
    first culture up to the test month. ``use_p`` and ``use_r`` switch the
    own-patient and ICU-wide past-resistance features. Weighted models
    down-weight training cultures that sit far from the test month.
    """

    name: str
    mode: str
    gap_years: int
    use_p: bool
    use_r: bool
    weighted: bool

    @property
    def window(self) -> str:
        if self.mode == "incremental":
            return "incremental-weighted" if self.weighted else "incremental"
        return f"sliding-{self.gap_years}y"


def _build_specs() -> tuple[ModelSpec, ...]:
    specs: list[ModelSpec] = []
    # M1–M3 neither past feature, M4–M6 ICU past only, M7–M9 both.
    # Inside each block the gap is 0, then 2, then 4 years.
    feature_sets = ((False, False), (False, True), (True, True))
    number = 1
    for use_p, use_r in feature_sets:
        for gap in (0, 2, 4):
            specs.append(ModelSpec(f"M{number}", "sliding", gap, use_p, use_r, False))
            number += 1
    for use_p, use_r in feature_sets:
        specs.append(ModelSpec(f"M{number}", "incremental", 0, use_p, use_r, False))
        number += 1
    specs.append(ModelSpec("M13", "incremental", 0, False, True, True))
    specs.append(ModelSpec("M14", "incremental", 0, True, True, True))
    return tuple(specs)


MODEL_SPECS: tuple[ModelSpec, ...] = _build_specs()


def get_specs(names: list[str] | None) -> list[ModelSpec]:
    """Return the requested models. ``None`` or ``['all']`` returns every model."""
    if not names or names == ["all"]:
        return list(MODEL_SPECS)
    by_name = {spec.name: spec for spec in MODEL_SPECS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        known = ", ".join(spec.name for spec in MODEL_SPECS)
        raise KeyError(f"Unknown model(s): {', '.join(unknown)}. Known models: {known}.")
    return [by_name[name] for name in names]
