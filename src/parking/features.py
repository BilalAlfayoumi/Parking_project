"""Nettoyage des capteurs et construction des features pour le ML.

Semaine 2 : remplir le nettoyage, la reconstruction d'occupation et les features.
"""

from __future__ import annotations

import pandas as pd


def clean_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige les erreurs de capteurs (arrivées après départs, valeurs manquantes).

    Args:
        df: Événements de capteurs bruts issus de :func:`parking.ingestion.load_raw`.

    Returns:
        DataFrame nettoyé, sans incohérences temporelles.
    """
    raise NotImplementedError("À implémenter en semaine 2.")


def build_occupancy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit l'état occupé/libre et agrège le taux d'occupation par rue.

    Args:
        df: Événements de capteurs nettoyés.

    Returns:
        Table du taux d'occupation par tranche de
        :data:`parking.config.AGG_FREQ_MINUTES` minutes et par rue.
    """
    raise NotImplementedError("À implémenter en semaine 2.")


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construit la matrice de features X et la cible y.

    Args:
        df: Table d'occupation issue de :func:`build_occupancy_table`.

    Returns:
        Tuple ``(X, y)`` prêt pour l'entraînement.
    """
    raise NotImplementedError("À implémenter en semaine 2.")
