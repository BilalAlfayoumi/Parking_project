"""Tests de la logique du dashboard (sans lancer Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from parking.dashboard.app import predict_all


class _FakeModel:
    def predict(self, X) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.full(len(X), 0.4)


def test_predict_all_shape_and_values() -> None:
    """predict_all renvoie une ligne par rue, avec coords et proba libre = 1−occ."""
    bundle = {"model": _FakeModel(), "streets": ["RUE A", "RUE B"]}
    geo = pd.DataFrame(
        {"street_name": ["RUE A", "RUE B"], "lat": [-37.8, -37.81], "lon": [144.9, 144.95]}
    )
    df = predict_all(bundle, geo, jour=2, heure=18)
    assert len(df) == 2
    assert {"street_name", "lat", "lon", "occupancy_rate", "probabilite_libre"} <= set(df.columns)
    assert np.allclose(df["probabilite_libre"], 0.6)
    assert df["lat"].notna().all()
