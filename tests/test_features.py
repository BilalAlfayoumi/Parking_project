"""Tests du nettoyage et de la construction de features (semaine 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from parking import features


@pytest.mark.skip(reason="À implémenter en semaine 2")
def test_clean_sensors_supprime_incoherences(raw_sensor_events: pd.DataFrame) -> None:
    """clean_sensors doit retirer les lignes où depart < arrivee."""
    cleaned = features.clean_sensors(raw_sensor_events)
    assert (cleaned["depart"] >= cleaned["arrivee"]).all()


@pytest.mark.skip(reason="À implémenter en semaine 2")
def test_make_features_renvoie_X_et_y(raw_sensor_events: pd.DataFrame) -> None:
    """make_features doit renvoyer un couple (X, y) aligné."""
    occ = features.build_occupancy_table(features.clean_sensors(raw_sensor_events))
    X, y = features.make_features(occ)
    assert len(X) == len(y)
