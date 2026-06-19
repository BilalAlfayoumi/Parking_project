"""Fixtures partagées pour les tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_sensor_events() -> pd.DataFrame:
    """Mini jeu d'événements de capteurs factices.

    Contient volontairement une incohérence (départ avant arrivée) pour tester
    :func:`parking.features.clean_sensors`.
    """
    return pd.DataFrame(
        {
            "bay_id": [1, 1, 2],
            "rue": ["Rue A", "Rue A", "Rue B"],
            "arrivee": pd.to_datetime(["2019-01-01 08:00", "2019-01-01 10:00", "2019-01-01 09:00"]),
            "depart": pd.to_datetime(["2019-01-01 09:00", "2019-01-01 09:30", "2019-01-01 11:00"]),
            # ligne 2 : depart (09:30) < arrivee (10:00) -> erreur capteur à nettoyer
        }
    )
