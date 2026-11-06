"""Decay, window and metric tests for the published method."""

import math

import numpy as np
import pandas as pd
import pytest

from amr_icu.config import DECAY_LAMBDA, MODEL_SPECS, get_specs
from amr_icu.experiment import run_experiment
from amr_icu.features import add_temporal_features, negative_exponential
from amr_icu.metrics import Counts
from amr_icu.preprocess import _drop_correlated, prepare_family
from amr_icu.synthetic import make_cohort
from amr_icu.windows import iter_windows


def test_decay_matches_the_paper_equation():
    assert negative_exponential(10, resistant=False) == 0.0
    assert negative_exponential(10, resistant=True) == pytest.approx(math.exp(-DECAY_LAMBDA * 10))
    assert negative_exponential(2, resistant=True) > negative_exponential(21, resistant=True)


def test_own_feature_is_a_max_and_icu_feature_sums_patients():
    frame = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B", "A", "C", "C"],
            "culture_date": [
                "2016-01-01",
                "2016-01-10",
                "2016-01-12",
                "2016-01-20",
                "2016-03-01",
                "2016-03-12",
            ],
            "c&amg": ["r", "s", "r", "s", "s", "s"],
        }
    )
    featured = add_temporal_features(frame)
    # 20 Jan looks back over [30 Dec, 18 Jan]. Patient A's resistant culture
    # on 1 Jan (19 days earlier) beats the susceptible culture on 10 Jan.
    # Patient B's resistant culture on 12 Jan is 8 days earlier.
    own = math.exp(-DECAY_LAMBDA * 19)
    other = math.exp(-DECAY_LAMBDA * 8)
    assert featured.loc[3, "p&amg"] == pytest.approx(own)
    assert featured.loc[3, "r&amg"] == pytest.approx(other)
    # 12 Mar has only a susceptible culture in the window, so p& is observed
    # and zero. 1 Jan has no earlier culture, so p& stays missing.
    assert featured.loc[5, "p&amg"] == 0.0
    assert math.isnan(featured.loc[0, "p&amg"])


def test_cultures_inside_the_48_hour_lag_are_invisible():
    frame = pd.DataFrame(
        {
            "patient_id": ["A", "A"],
            "culture_date": ["2016-01-01", "2016-01-02"],
            "c&amg": ["r", "s"],
        }
    )
    featured = add_temporal_features(frame)
    assert math.isnan(featured.loc[1, "p&amg"])


def test_sliding_window_drops_the_margin_and_the_test_patient():
    dates = pd.Series(pd.to_datetime(["2012-06-01", "2015-12-29", "2015-12-31", "2016-01-15", "2016-02-10"]))
    patients = pd.Series(["A", "B", "A", "B", "A"])
    spec = get_specs(["M1"])[0]
    windows = iter_windows(dates, patients, spec)
    january = windows[0]
    assert january.test_start == pd.Timestamp("2016-01-01")
    assert list(january.test_index) == [3]
    # A on 2012-06-01 is inside 2012–2015. B is in the January test, so B is
    # removed from training. 31 Dec is inside the 48-hour margin.
    assert list(january.train_index) == [0]


def test_incremental_window_grows_and_keeps_the_gap():
    dates = pd.Series(pd.to_datetime(["2012-06-01", "2015-12-29", "2015-12-31", "2016-01-15", "2016-02-10"]))
    patients = pd.Series(["A", "B", "A", "B", "A"])
    spec = get_specs(["M12"])[0]
    windows = iter_windows(dates, patients, spec)
    february = next(window for window in windows if window.test_start == pd.Timestamp("2016-02-01"))
    assert list(february.test_index) == [4]
    # Training may use January, and it may not use patient A.
    assert list(february.train_index) == [1, 3]
    assert february.train_start is None


def test_fourteen_models_follow_the_paper():
    assert [spec.name for spec in MODEL_SPECS] == [f"M{number}" for number in range(1, 15)]
    assert [spec.gap_years for spec in MODEL_SPECS[:3]] == [0, 2, 4]
    assert MODEL_SPECS[0].use_p is False and MODEL_SPECS[0].use_r is False
    assert MODEL_SPECS[3].use_r is True and MODEL_SPECS[3].use_p is False
    assert MODEL_SPECS[6].use_p is True and MODEL_SPECS[6].use_r is True
    assert MODEL_SPECS[11].name == "M12" and MODEL_SPECS[11].weighted is False
    assert MODEL_SPECS[13].name == "M14" and MODEL_SPECS[13].weighted is True


def test_accuracy_is_accumulated_not_averaged_across_months():
    counts = Counts()
    counts.add(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    counts.add(np.array([1, 1]), np.array([1, 1]))
    accuracies = counts.accuracies()
    assert accuracies["resistant_accuracy"] == pytest.approx(100.0 * 3 / 4)
    assert accuracies["susceptible_accuracy"] == pytest.approx(100.0)
    assert accuracies["total_accuracy"] == pytest.approx(100.0 * 5 / 6)


def test_correlated_dates_keep_the_culture_date():
    frame = pd.DataFrame(
        {
            "date_culture": [0.0, 10.0, 20.0, 30.0],
            "year_culture": [0.0, 10.0, 20.0, 30.0],
            "age": [1.0, 0.0, 1.0, 0.0],
        }
    )
    kept = _drop_correlated(frame)
    assert "date_culture" in kept.columns
    assert "year_culture" not in kept.columns
    assert "age" in kept.columns


def test_synthetic_cohort_runs_the_incremental_model():
    cohort = make_cohort(n_patients=50, seed=3)
    featured = add_temporal_features(cohort)
    prepared = prepare_family(featured, "amg")
    assert set(prepared["target"].unique()) == {0, 1}
    result = run_experiment(prepared, family="amg", models=["M12"], methods=("lr",), use_date=(False,))
    assert len(result) == 1
    assert result.loc[0, "n_steps"] > 0
    assert 0.0 <= result.loc[0, "total_accuracy"] <= 100.0
