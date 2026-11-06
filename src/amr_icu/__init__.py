"""Temporal prediction of antimicrobial resistance in the ICU.

This package implements the study published as Hernández-Carnerero et al.,
"Antimicrobial Resistance Prediction in Intensive Care Unit for Pseudomonas
aeruginosa using Temporal Data-Driven Models", International Journal of
Interactive Multimedia and Artificial Intelligence, 2021.
DOI: https://doi.org/10.9781/ijimai.2021.02.012
"""

from amr_icu.experiment import run_experiment
from amr_icu.features import add_temporal_features
from amr_icu.synthetic import make_cohort

__all__ = ["add_temporal_features", "make_cohort", "run_experiment"]
__version__ = "1.0.0"
