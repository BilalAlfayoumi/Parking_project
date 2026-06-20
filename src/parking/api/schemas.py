"""Schémas Pydantic pour les requêtes et réponses de l'API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Réponse de la route ``/predict``."""

    rue: str
    heure: int
    jour: int
    occupancy_rate: float = Field(..., ge=0.0, le=1.0, description="Taux d'occupation prédit")
    probabilite_libre: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilité de trouver une place (1 − occupation)"
    )
    niveau: str = Field(..., description="Chance de se garer : élevée / moyenne / faible")


class StreetsResponse(BaseModel):
    """Réponse de la route ``/streets`` : rues connues du modèle."""

    count: int
    streets: list[str]
