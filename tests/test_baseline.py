"""Tests de la baseline naïve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from parking.baseline import BaselineModel


def _toy() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "street_code": [0, 0, 1, 1],
            "hour": [8, 8, 8, 8],
            "day_of_week": [1, 1, 1, 1],
        }
    )
    y = pd.Series([0.2, 0.4, 0.8, 1.0])
    return X, y


def test_baseline_predicts_group_mean() -> None:
    """La baseline renvoie la moyenne du groupe (rue, heure, jour)."""
    X, y = _toy()
    model = BaselineModel().fit(X, y)
    pred = model.predict(X)
    assert np.isclose(pred[0], 0.3)  # moyenne de la rue 0 à 8h, jour 1
    assert np.isclose(pred[2], 0.9)  # moyenne de la rue 1


def test_baseline_unknown_combo_falls_back() -> None:
    """Une combinaison absente retombe sur une valeur connue (pas de NaN)."""
    X, y = _toy()
    model = BaselineModel().fit(X, y)
    unseen = pd.DataFrame({"street_code": [0], "hour": [22], "day_of_week": [6]})
    pred = model.predict(unseen)
    assert not np.isnan(pred).any()
