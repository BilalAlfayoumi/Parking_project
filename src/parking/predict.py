"""Inférence : charge le modèle et renvoie une probabilité de place libre.

Semaine 4-5 : remplir. Utilisé par l'API FastAPI.
"""

from __future__ import annotations


def predict_probability(rue: str, heure: int, jour: int) -> float:
    """Prédit la probabilité de trouver une place libre dans une rue.

    Args:
        rue: Identifiant ou nom de la rue.
        heure: Heure de la journée (0-23).
        jour: Jour de la semaine (0 = lundi … 6 = dimanche).

    Returns:
        Probabilité de place libre, dans l'intervalle ``[0, 1]``.
    """
    raise NotImplementedError("À implémenter en semaine 4-5.")
