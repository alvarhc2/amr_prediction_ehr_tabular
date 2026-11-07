"""Command line for the synthetic cohort and for a local cohort file."""

import argparse
from pathlib import Path

import pandas as pd

from amr_icu.config import FAMILIES
from amr_icu.experiment import run_experiment
from amr_icu.features import add_temporal_features
from amr_icu.importance import rank_features, top_table
from amr_icu.preprocess import prepare_family
from amr_icu.schema import read_cohort
from amr_icu.synthetic import make_cohort


def main() -> None:
    args = _parser().parse_args()
    if args.data and args.synthetic:
        raise SystemExit("Pass either --data or --synthetic.")
    if not args.data and not args.synthetic:
        args.synthetic = True

    if args.synthetic:
        print(f"cohort    synthetic, {args.patients} patients", flush=True)
        cohort = make_cohort(n_patients=args.patients)
    else:
        print(f"cohort    {args.data}", flush=True)
        cohort = read_cohort(args.data)

    dates = pd.to_datetime(cohort["culture_date"])
    print(
        f"          {len(cohort)} cultures, {cohort['patient_id'].nunique()} patients, "
        f"{dates.min():%Y-%m-%d} .. {dates.max():%Y-%m-%d}",
        flush=True,
    )

    if args.export_synthetic:
        if not args.synthetic:
            raise SystemExit("--export-synthetic only applies to --synthetic.")
        destination = Path(args.export_synthetic)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(destination, index=False)
        print(f"Wrote {destination}")

    print("features  p&, r&", flush=True)
    featured = add_temporal_features(cohort)
    families = list(FAMILIES) if args.family == "all" else [args.family]
    use_date = {"on": (True,), "off": (False,), "both": (True, False)}[args.date_culture]
    models = ["all"] if args.full else args.models
    if args.full:
        families = list(FAMILIES)
        use_date = (True, False)

    frames = []
    for family in families:
        prepared = prepare_family(featured, family)
        print(
            f"{family.upper():<9} {len(prepared)} cultures, "
            f"{int((prepared['target'] == 1).sum())} resistant, "
            f"{int((prepared['target'] == 0).sum())} susceptible",
            flush=True,
        )
        result = run_experiment(
            prepared,
            family=family,
            models=models,
            methods=tuple(args.methods),
            use_date=use_date,
            verbose=True,
        )
        frames.append(result)
        if args.importance:
            mutual, forest = rank_features(prepared)
            print(f"\n{family.upper()} mutual information")
            print(top_table(mutual).to_string(index=False))
            print(f"\n{family.upper()} random-forest importance")
            print(top_table(forest).to_string(index=False))

    table = pd.concat(frames, ignore_index=True)
    _print_table(table, synthetic=args.synthetic)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        print(f"Wrote {destination}")


def _print_table(table: pd.DataFrame, synthetic: bool) -> None:
    view = table.copy()
    for column in ("total_accuracy", "resistant_accuracy", "susceptible_accuracy", "lambda"):
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    columns = [
        "family",
        "model",
        "window",
        "date_culture",
        "method",
        "lambda",
        "total_accuracy",
        "resistant_accuracy",
        "susceptible_accuracy",
        "n_steps",
    ]
    print()
    print(view.loc[:, columns].to_string(index=False))
    print()
    print("n_steps: test months used. lambda: selected weight decay, blank when unused.")
    if synthetic:
        print("Synthetic smoke test. Published numbers are in results/.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict Pseudomonas aeruginosa resistance with the temporal models from the study."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=str, help="CSV cohort with the columns documented in data/README.md")
    source.add_argument("--synthetic", action="store_true", help="Run on an invented cohort")
    parser.add_argument("--patients", type=int, default=140, help="Patients in the synthetic cohort")
    parser.add_argument("--family", default="amg", choices=[*FAMILIES, "all"])
    parser.add_argument("--models", nargs="+", default=["M7", "M12", "M14"], help="Model names, or 'all'")
    parser.add_argument("--methods", nargs="+", default=["lr", "rf"], choices=["lr", "rf"])
    parser.add_argument("--date-culture", default="off", choices=["on", "off", "both"])
    parser.add_argument("--full", action="store_true", help="All families, all 14 models, with and without the culture date")
    parser.add_argument("--importance", action="store_true", help="Print mutual-information and random-forest rankings")
    parser.add_argument("--export-synthetic", type=str, help="Write the invented cohort to this CSV path")
    parser.add_argument("--output", type=str, help="Write the metric table to this CSV path")
    return parser
