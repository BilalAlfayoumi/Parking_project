"""Fixtures partagées pour les tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def events() -> pd.DataFrame:
    """Mini jeu d'événements au schéma de ``parking.ingestion.load_events``.

    Contient volontairement deux lignes invalides (départ avant arrivée, durée
    nulle) pour tester :func:`parking.features.clean_events`.
    """
    return pd.DataFrame(
        {
            "street_name": ["RUE A", "RUE A", "RUE A", "RUE A"],
            "street_marker": ["A1", "A2", "A1", "A2"],
            "arrival_time": pd.to_datetime(
                [
                    "2019-01-01 08:00",
                    "2019-01-01 08:10",  # ligne invalide : départ avant arrivée
                    "2019-01-01 09:00",
                    "2019-01-01 09:05",  # ligne invalide : durée nulle
                ]
            ),
            "departure_time": pd.to_datetime(
                [
                    "2019-01-01 08:40",
                    "2019-01-01 07:50",
                    "2019-01-01 09:30",
                    "2019-01-01 09:05",
                ]
            ),
        }
    )
