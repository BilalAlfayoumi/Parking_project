"""Tests de l'entraînement, de la persistance et de l'inférence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from parking import predict as predict_mod
from parking.evaluation import evaluate, temporal_split
from parking.features import FEATURE_COLS
from parking.model import load_model, save_model, train


def _toy_xy(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "street_code": rng.integers(0, 3, n),
            "hour": rng.integers(0, 24, n),
            "minute": rng.choice([0, 30], n),
            "day_of_week": rng.integers(0, 7, n),
            "is_weekend": rng.integers(0, 2, n),
            "month": rng.integers(1, 13, n),
        }
    )[FEATURE_COLS]
    y = pd.Series((X["hour"] / 24).to_numpy())
    return X, y


def test_save_then_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Un modèle sauvegardé se recharge et prédit à l'identique."""
    X, y = _toy_xy()
    model = train(X, y)
    path = tmp_path / "model.joblib"
    save_model(model, ["RUE A", "RUE B", "RUE C"], path)
    bundle = load_model(path)
    assert bundle["streets"] == ["RUE A", "RUE B", "RUE C"]
    assert np.allclose(bundle["model"].predict(X), model.predict(X))


def test_evaluate_keys() -> None:
    """evaluate renvoie MAE, RMSE, R²."""
    m = evaluate(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert set(m) == {"MAE", "RMSE", "R2"}
    assert m["MAE"] == 0.0


def test_temporal_split_mask() -> None:
    """Le split temporel sépare bien avant/après la coupure."""
    occ = pd.DataFrame({"slot": pd.to_datetime(["2019-05-01", "2019-11-01"])})
    mask = temporal_split(occ, "2019-10-01")
    assert mask.tolist() == [True, False]


def test_predict_occupancy(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """predict_occupancy renvoie un taux dans [0, 1] pour une rue connue."""
    X, y = _toy_xy()
    model = train(X, y)
    path = tmp_path / "model.joblib"
    save_model(model, ["RUE A", "RUE B", "RUE C"], path)
    monkeypatch.setattr(predict_mod, "load_model", lambda: load_model(path))
    val = predict_mod.predict_occupancy("RUE A", heure=8, jour=2)
    assert 0.0 <= val <= 1.0
