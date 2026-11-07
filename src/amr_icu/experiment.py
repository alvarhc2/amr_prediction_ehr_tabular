"""Run the published models on one prepared family table."""

import numpy as np
import pandas as pd

from amr_icu.config import TRAIN_YEARS, WEIGHT_LAMBDAS, ModelSpec, get_specs
from amr_icu.metrics import Counts
from amr_icu.models import fit_predict
from amr_icu.schema import META_COLUMNS
from amr_icu.windows import instance_weights, iter_windows


def run_experiment(
    frame: pd.DataFrame,
    family: str,
    models: list[str] | None = None,
    methods: tuple[str, ...] = ("lr", "rf"),
    use_date: tuple[bool, ...] = (False,),
    weight_grid: tuple[float, ...] = WEIGHT_LAMBDAS,
    verbose: bool = False,
) -> pd.DataFrame:
    """Evaluate the requested models.

    ``frame`` is the output of :func:`amr_icu.preprocess.prepare_family`.
    Accuracies are percentages accumulated across test months. For a weighted
    model the decay is the grid value with the highest mean of the defined
    accuracies. Ties keep the smaller decay, which uses more of the history.
    """
    specs = get_specs(models)
    rows: list[dict] = []
    for spec in specs:
        windows = iter_windows(frame["culture_date"], frame["patient_id"], spec)
        for date_flag in use_date:
            columns = feature_columns(frame, spec, use_date=date_flag)
            matrix = frame.loc[:, columns].to_numpy(dtype=float)
            target = frame["target"].to_numpy(dtype=int)
            for method in methods:
                if verbose:
                    print(_model_banner(family, spec, method, date_flag), flush=True)
                if spec.weighted:
                    if verbose:
                        grid = ", ".join(format_lambda(lam) for lam in weight_grid)
                        print(
                            "    Trying every weight decay on the full sequence of test months: "
                            f"{grid}. The decay with the highest mean of the three accuracies is kept. "
                            "A tie keeps the smaller decay.",
                            flush=True,
                        )
                    chosen_lam, counts = _best_weight(
                        frame, matrix, target, windows, method, weight_grid, verbose=verbose
                    )
                else:
                    if verbose:
                        print(
                            "    Fitting one classifier per test month and accumulating the counts. "
                            "This model has no weight decay.",
                            flush=True,
                        )
                    chosen_lam = float("nan")
                    counts = _accumulate(frame, matrix, target, windows, method, lam=None)
                    if verbose:
                        print(
                            f"    Done. {counts.steps} test months contributed to the accuracies.",
                            flush=True,
                        )
                rows.append(_row(family, spec, date_flag, method, chosen_lam, counts))
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame, spec: ModelSpec, use_date: bool) -> list[str]:
    """Predictors for one model, after the temporal features have been built."""
    columns = [column for column in frame.columns if column not in META_COLUMNS]
    if not spec.use_p:
        columns = [column for column in columns if not column.startswith("p&")]
    if not spec.use_r:
        columns = [column for column in columns if not column.startswith("r&")]
    if not use_date:
        columns = [column for column in columns if column != "date_culture"]
    if not columns:
        raise ValueError(f"{spec.name} has no predictors left.")
    return columns


def _best_weight(
    frame, matrix, target, windows, method, weight_grid, verbose: bool = False
) -> tuple[float, Counts]:
    best_lam = weight_grid[0]
    best_counts = None
    best_score = -1.0
    for lam in weight_grid:
        if verbose:
            print(f"    lambda = {format_lambda(lam)}", flush=True)
        counts = _accumulate(frame, matrix, target, windows, method, lam=lam)
        score = counts.selection_score()
        if best_counts is None or score > best_score + 1e-12:
            best_lam = lam
            best_counts = counts
            best_score = score
    if best_counts is None:
        raise ValueError("The weight grid is empty.")
    if verbose:
        print(
            f"    Selected lambda = {format_lambda(best_lam)}. "
            f"{best_counts.steps} test months contributed to the accuracies.",
            flush=True,
        )
    return best_lam, best_counts


def format_lambda(lam: float) -> str:
    """Render a grid value as 0, 1e-05, or 0.1."""
    if lam == 0:
        return "0"
    if lam >= 0.01:
        return f"{lam:g}"
    return f"{lam:.0e}"


def _model_banner(family: str, spec: ModelSpec, method: str, use_date: bool) -> str:
    method_name = "logistic regression" if method == "lr" else "random forest"
    if spec.mode == "sliding":
        window = (
            f"sliding window of {TRAIN_YEARS} years, "
            f"placed {spec.gap_years} years before the test month"
        )
    elif spec.weighted:
        window = "incremental window with a weight on each training culture"
    else:
        window = "incremental window, from the first culture up to the test month"
    if spec.use_p and spec.use_r:
        past = "p& and r&"
    elif spec.use_r:
        past = "r& only"
    elif spec.use_p:
        past = "p& only"
    else:
        past = "none"
    date = "included" if use_date else "excluded"
    return (
        f"  {family.upper()} {spec.name}, {method_name} ({method}). "
        f"{window}. Culture date {date}. Past-resistance features: {past}."
    )


def _accumulate(frame, matrix, target, windows, method, lam: float | None) -> Counts:
    counts = Counts()
    for window in windows:
        if len(window.train_index) == 0 or len(window.test_index) == 0:
            continue
        y_train = target[window.train_index]
        if len(np.unique(y_train)) < 2:
            continue
        weights = None
        if lam is not None:
            weights = instance_weights(frame["culture_date"], window, lam)
        prediction = fit_predict(
            matrix[window.train_index],
            y_train,
            matrix[window.test_index],
            method,
            sample_weight=weights,
        )
        counts.add(target[window.test_index], prediction)
    return counts


def _row(family, spec: ModelSpec, use_date: bool, method: str, lam: float, counts: Counts) -> dict:
    accuracies = counts.accuracies()
    return {
        "family": family,
        "model": spec.name,
        "window": spec.window,
        "gap_years": spec.gap_years if spec.mode == "sliding" else pd.NA,
        "use_p": spec.use_p,
        "use_r": spec.use_r,
        "weighted": spec.weighted,
        "date_culture": use_date,
        "method": method,
        "lambda": lam,
        "total_accuracy": accuracies["total_accuracy"],
        "resistant_accuracy": accuracies["resistant_accuracy"],
        "susceptible_accuracy": accuracies["susceptible_accuracy"],
        "n_resistant": counts.resistant,
        "n_susceptible": counts.susceptible,
        "n_steps": counts.steps,
    }
