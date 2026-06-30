"""Tests du modèle de prévision des parkings Bordeaux (sur données factices)."""

from __future__ import annotations

import pandas as pd

from parking import bordeaux_model as bm


def _fake_history(n_days: int = 8) -> pd.DataFrame:
    """Historique synthétique : occupation = fonction de l'heure (signal net)."""
    rows = []
    base = pd.Timestamp("2026-06-01T00:00Z")
    for day in range(n_days):
        for hour in range(0, 24, 2):
            ts = base + pd.Timedelta(days=day, hours=hour)
            for ident in ["P1", "P2"]:
                # occupation forte en journée, faible la nuit
                occ = 0.2 + 0.6 * (1 if 8 <= hour <= 18 else 0)
                rows.append({"ident": ident, "mdate": ts, "occupancy_rate": occ})
    return pd.DataFrame(rows)


def test_make_features_park_code() -> None:
    """make_features ajoute park_code et renvoie la liste ordonnée des parkings."""
    feats, parks = bm.make_features(_fake_history())
    assert parks == ["P1", "P2"]
    assert "park_code" in feats.columns


def test_train_forecast_beats_baseline(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Sur un signal horaire net, le modèle bat (ou égale) la baseline."""
    monkeypatch.setattr(bm, "HISTORY_FILE", tmp_path / "h.parquet")
    monkeypatch.setattr(bm, "BX_MODEL_FILE", tmp_path / "m.joblib")
    _fake_history().to_parquet(bm.HISTORY_FILE, index=False)

    bundle = bm.train_forecast()
    assert set(bundle) == {"model", "parks", "scores"}
    assert bundle["scores"]["model"]["MAE"] <= bundle["scores"]["baseline"]["MAE"] + 0.05
    assert (tmp_path / "m.joblib").exists()


def test_predict_occupancy_bounds(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """predict_occupancy renvoie un taux dans [0, 1] pour un parking connu."""
    monkeypatch.setattr(bm, "HISTORY_FILE", tmp_path / "h.parquet")
    monkeypatch.setattr(bm, "BX_MODEL_FILE", tmp_path / "m.joblib")
    _fake_history().to_parquet(bm.HISTORY_FILE, index=False)
    bm.train_forecast()
    val = bm.predict_occupancy("P1", hour=12, day_of_week=2)
    assert 0.0 <= val <= 1.0
