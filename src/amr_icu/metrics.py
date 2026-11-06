"""Accumulated resistant, susceptible and total accuracy.

Test months do not contain the same number of cultures, so a mean of monthly
accuracies would give a quiet month the same vote as a busy one. The study
adds the counts across months and divides once at the end.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Counts:
    resistant: int = 0
    susceptible: int = 0
    resistant_correct: int = 0
    susceptible_correct: int = 0
    steps: int = 0

    def add(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        resistant = y_true == 1
        susceptible = y_true == 0
        self.resistant += int(resistant.sum())
        self.susceptible += int(susceptible.sum())
        self.resistant_correct += int((resistant & (y_pred == 1)).sum())
        self.susceptible_correct += int((susceptible & (y_pred == 0)).sum())
        self.steps += 1

    def accuracies(self) -> dict[str, float]:
        total = self.resistant + self.susceptible
        return {
            "total_accuracy": _ratio(self.resistant_correct + self.susceptible_correct, total),
            "resistant_accuracy": _ratio(self.resistant_correct, self.resistant),
            "susceptible_accuracy": _ratio(self.susceptible_correct, self.susceptible),
        }

    def selection_score(self) -> float:
        """Mean of the accuracies that are defined. Used to choose lambda."""
        values = [value for value in self.accuracies().values() if not np.isnan(value)]
        if not values:
            return -1.0
        return float(np.mean(values))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return 100.0 * numerator / denominator
