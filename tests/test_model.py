"""Tests de l'entraînement et de la persistance du modèle (semaine 4)."""

from __future__ import annotations

import pytest

from parking import model


@pytest.mark.skip(reason="À implémenter en semaine 4")
def test_save_then_load_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Un modèle sauvegardé doit pouvoir être rechargé à l'identique."""
    path = tmp_path / "model.joblib"
    trained = model.train(X=None, y=None)  # type: ignore[arg-type]
    model.save_model(trained, path)
    reloaded = model.load_model(path)
    assert reloaded is not None
