"""Chargement des données brutes de capteurs de stationnement (Melbourne).

Semaine 1-2 : remplir `load_raw`.
"""

from __future__ import annotations

import pandas as pd


def load_raw(year: int) -> pd.DataFrame:
    """Charge une année de données brutes de capteurs depuis ``data/raw``.

    Args:
        year: Année du jeu de données à charger (ex. ``2019``).

    Returns:
        DataFrame des événements de capteurs bruts (arrivées/départs).
    """
    raise NotImplementedError("À implémenter en semaine 1-2.")
