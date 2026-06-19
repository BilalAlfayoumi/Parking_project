"""Schémas Pydantic pour les requêtes et réponses de l'API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Réponse de la route ``/predict``."""

    rue: str
    heure: int
    jour: int
    probabilite: float = Field(..., ge=0.0, le=1.0, description="Probabilité de place libre")
    label: str = Field(..., description="Interprétation lisible : libre / incertain / occupé")
