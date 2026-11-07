"""Run the published models on one prepared family table."""

import numpy as np
import pandas as pd

from amr_icu.config import WEIGHT_LAMBDAS, ModelSpec, get_specs
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
                    date = "on" if date_flag else "off"
                    print(f"  {spec.name} {method}  {spec.window}  date {date}", flush=True)
                if spec.weighted:
                    chosen_lam, counts = _best_weight(
                        frame, matrix, target, windows, method, weight_grid, verbose=verbose
                    )
                else:
                    chosen_lam = float("nan")
                    counts = _accumulate(
                        frame, matrix, target, windows, method, lam=None, verbose=verbose
                    )
                    if verbose:
                        print(f"    {counts.steps} months", flush=True)
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
    for index, lam in enumerate(weight_grid):
        if verbose:
            print(f"    lambda {format_lambda(lam)}", flush=True)
        counts = _accumulate(
            frame, matrix, target, windows, method, lam=lam, verbose=verbose and index == 0
        )
        score = counts.selection_score()
        if best_counts is None or score > best_score + 1e-12:
            best_lam = lam
            best_counts = counts
            best_score = score
    if best_counts is None:
        raise ValueError("The weight grid is empty.")
    if verbose:
        print(f"    selected {format_lambda(best_lam)}, {best_counts.steps} months", flush=True)
    return best_lam, best_counts


def format_lambda(lam: float) -> str:
    """Render a grid value as 0, 1e-05, or 0.1."""
    if lam == 0:
        return "0"
    if lam >= 0.01:
        return f"{lam:g}"
    return f"{lam:.0e}"


def _accumulate(frame, matrix, target, windows, method, lam: float | None, verbose: bool = False) -> Counts:
    counts = Counts()
    shown = 0
    for window in windows:
        if len(window.train_index) == 0 or len(window.test_index) == 0:
            continue
        y_train = target[window.train_index]
        if len(np.unique(y_train)) < 2:
            continue
        if verbose:
            if shown % 6 == 0:
                print("   ", end="", flush=True)
            print(f" {window.test_start:%Y-%m}", end="", flush=True)
            shown += 1
            if shown % 6 == 0:
                print(flush=True)
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
    if verbose and shown % 6:
        print(flush=True)
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
