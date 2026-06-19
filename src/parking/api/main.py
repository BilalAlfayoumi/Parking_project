"""Application FastAPI exposant la prédiction d'occupation.

Stub fonctionnel : ``/predict`` renvoie une probabilité factice (0.5) tant que le
modèle n'est pas branché (semaine 5). La structure des routes est définitive.
"""

from __future__ import annotations

from fastapi import FastAPI

from parking.api.schemas import PredictResponse
from parking.config import FREE_THRESHOLD

app = FastAPI(
    title="Parking Occupancy Predictor",
    description="Prédit la probabilité de trouver une place de stationnement libre.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Sonde de disponibilité."""
    return {"status": "ok"}


@app.get("/predict", response_model=PredictResponse)
def predict(rue: str, heure: int, jour: int) -> PredictResponse:
    """Renvoie la probabilité de place libre pour une rue, une heure et un jour.

    Stub : renvoie 0.5 en attendant le branchement du modèle (semaine 5).
    """
    probabilite = 0.5  # TODO(semaine 5): appeler parking.predict.predict_probability
    if probabilite >= FREE_THRESHOLD + 0.2:
        label = "libre"
    elif probabilite <= FREE_THRESHOLD - 0.2:
        label = "occupé"
    else:
        label = "incertain"
    return PredictResponse(rue=rue, heure=heure, jour=jour, probabilite=probabilite, label=label)
