"""Évaluation des modèles et comparaison à la baseline.

Le modèle est jugé sur un split *temporel* (on prédit le futur), avec MAE, RMSE
et R², et toujours *contre* la baseline naïve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def temporal_split(occ: pd.DataFrame, cutoff: str | pd.Timestamp) -> np.ndarray:
    """Masque booléen du jeu d'entraînement (créneaux antérieurs à ``cutoff``).

    Args:
        occ: Table d'occupation contenant la colonne ``slot``.
        cutoff: Date de coupure train/test.

    Returns:
        Tableau booléen ``True`` pour les lignes d'entraînement.
    """
    return (occ["slot"] < pd.Timestamp(cutoff)).to_numpy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule MAE, RMSE et R².

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.

    Returns:
        Dictionnaire ``{"MAE", "RMSE", "R2"}``.
    """
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def compare_to_baseline(
    y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_baseline: np.ndarray
) -> dict[str, dict[str, float]]:
    """Compare les métriques du modèle à celles de la baseline.

    Returns:
        ``{"model": {...}, "baseline": {...}}``.
    """
    return {
        "model": evaluate(y_true, y_pred_model),
        "baseline": evaluate(y_true, y_pred_baseline),
    }
