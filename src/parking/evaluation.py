"""Évaluation des modèles et comparaison à la baseline.

Semaine 3 : remplir. Le modèle doit être jugé *contre* la baseline naïve.
"""

from __future__ import annotations

import pandas as pd


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcule les métriques de performance.

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.

    Returns:
        Dictionnaire de métriques (ex. ``{"f1": ..., "accuracy": ..., "mae": ...}``).
    """
    raise NotImplementedError("À implémenter en semaine 3.")


def compare_to_baseline(
    y_true: pd.Series, y_pred_model: pd.Series, y_pred_baseline: pd.Series
) -> dict[str, dict[str, float]]:
    """Compare les métriques du modèle à celles de la baseline.

    Args:
        y_true: Valeurs réelles.
        y_pred_model: Prédictions du modèle ML.
        y_pred_baseline: Prédictions de la baseline naïve.

    Returns:
        Dictionnaire ``{"model": {...}, "baseline": {...}}`` des métriques comparées.
    """
    raise NotImplementedError("À implémenter en semaine 3.")
