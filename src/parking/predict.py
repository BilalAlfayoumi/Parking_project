"""Inférence : probabilité d'occupation pour une rue, une heure et un jour.

Utilise le modèle entraîné par :mod:`parking.model` (cible ``make train``).
"""

from __future__ import annotations

import pandas as pd

from parking.features import FEATURE_COLS
from parking.model import load_model


def predict_occupancy(rue: str, heure: int, jour: int, minute: int = 0, mois: int = 6) -> float:
    """Prédit le taux d'occupation d'une rue à un moment donné.

    Args:
        rue: Nom de la rue (``street_name``), tel qu'à l'entraînement.
        heure: Heure de la journée (0-23).
        jour: Jour de la semaine (0 = lundi … 6 = dimanche).
        minute: Minute du créneau (0 ou 30).
        mois: Mois (1-12).

    Returns:
        Taux d'occupation prédit, dans ``[0, 1]``.

    Raises:
        ValueError: Si la rue est inconnue du modèle.
    """
    bundle = load_model()
    streets = bundle["streets"]
    if rue not in streets:
        raise ValueError(f"Rue inconnue du modèle : {rue!r}")

    row = {
        "street_code": list(streets).index(rue),
        "hour": heure,
        "minute": minute,
        "day_of_week": jour,
        "is_weekend": int(jour >= 5),
        "month": mois,
    }
    X = pd.DataFrame([row])[FEATURE_COLS]
    return float(bundle["model"].predict(X)[0])


def predict_free_probability(rue: str, heure: int, jour: int, **kwargs: int) -> float:
    """Probabilité de trouver une place libre (= 1 − occupation prédite)."""
    return 1.0 - predict_occupancy(rue, heure, jour, **kwargs)
