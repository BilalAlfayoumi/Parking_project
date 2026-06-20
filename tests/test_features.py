"""Tests du nettoyage, de la reconstruction d'occupation et des features."""

from __future__ import annotations

import pandas as pd

from parking.features import (
    FEATURE_COLS,
    add_time_features,
    clean_events,
    make_features,
    occupancy_for_street,
)


def test_clean_events_removes_invalid(events: pd.DataFrame) -> None:
    """Les durées <= 0 et les départs avant arrivées sont retirés."""
    cleaned = clean_events(events)
    assert len(cleaned) == 2
    assert (cleaned["duration_min"] > 0).all()


def test_occupancy_rate_in_bounds(events: pd.DataFrame) -> None:
    """Le taux d'occupation reste dans [0, 1] et la table a la bonne forme."""
    occ = occupancy_for_street(clean_events(events))
    assert {"slot", "occupancy_rate", "n_bays"} <= set(occ.columns)
    assert occ["occupancy_rate"].between(0, 1).all()


def test_make_features_shapes(events: pd.DataFrame) -> None:
    """make_features renvoie X (FEATURE_COLS) et y alignés."""
    occ = occupancy_for_street(clean_events(events))
    occ["street_name"] = "RUE A"
    X, y = make_features(add_time_features(occ))
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y)
