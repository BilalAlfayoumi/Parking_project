"""Tests de l'API FastAPI (actifs dès le scaffold : le stub est fonctionnel)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from parking.api.main import app

client = TestClient(app)


def test_health() -> None:
    """La sonde /health répond 200 avec un statut ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_renvoie_structure_attendue() -> None:
    """La route /predict renvoie une réponse typée valide."""
    resp = client.get("/predict", params={"rue": "Rue A", "heure": 8, "jour": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probabilite"] <= 1.0
    assert body["label"] in {"libre", "incertain", "occupé"}
    assert body["rue"] == "Rue A"
