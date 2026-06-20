"""Tests de l'API FastAPI (modèle simulé via dependency_overrides)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from parking.api.main import app, get_bundle


class _FakeModel:
    def predict(self, X) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.full(len(X), 0.3)


_FAKE_BUNDLE = {"model": _FakeModel(), "streets": ["RUE A", "RUE B"]}
app.dependency_overrides[get_bundle] = lambda: _FAKE_BUNDLE

client = TestClient(app)


def test_health() -> None:
    """La sonde /health répond 200 sans modèle."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_streets() -> None:
    """/streets renvoie la liste des rues connues."""
    resp = client.get("/streets")
    assert resp.status_code == 200
    assert resp.json()["streets"] == ["RUE A", "RUE B"]


def test_predict_ok() -> None:
    """/predict renvoie une réponse cohérente (occupation 0.3 -> libre 0.7)."""
    resp = client.get("/predict", params={"rue": "RUE A", "heure": 8, "jour": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert abs(body["occupancy_rate"] - 0.3) < 1e-6
    assert abs(body["probabilite_libre"] - 0.7) < 1e-6
    assert body["niveau"] == "élevée"


def test_predict_unknown_street() -> None:
    """Une rue inconnue renvoie 404."""
    resp = client.get("/predict", params={"rue": "RUE Z", "heure": 8, "jour": 1})
    assert resp.status_code == 404


def test_predict_invalid_hour() -> None:
    """Une heure hors bornes renvoie 422 (validation)."""
    resp = client.get("/predict", params={"rue": "RUE A", "heure": 99, "jour": 1})
    assert resp.status_code == 422
