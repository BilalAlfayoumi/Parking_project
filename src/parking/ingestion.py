"""Chargement des données brutes de capteurs de stationnement (Melbourne).

Semaine 1-2 : on prototype d'abord le chargement dans le notebook d'exploration
(``notebooks/01_exploration.ipynb``). Une fois la logique stabilisée, on la
remonte ici dans ``load_raw`` (industrialisation, semaine 4).
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
    raise NotImplementedError("À stabiliser dans le notebook puis remonter ici (semaine 4).")
