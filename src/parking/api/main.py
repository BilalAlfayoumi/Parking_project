"""Application FastAPI exposant le prédicteur d'occupation.

Le modèle (entraîné via ``make train``) est chargé à la demande et mis en cache.
``GET /predict`` renvoie le taux d'occupation prédit et la probabilité de trouver
une place pour une rue, une heure et un jour donnés.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from parking.api.schemas import PredictResponse, StreetsResponse
from parking.features import FEATURE_COLS
from parking.model import load_model

app = FastAPI(
    title="Parking Occupancy Predictor",
    description="Prédit la probabilité de trouver une place de stationnement libre.",
    version="0.1.0",
)


@lru_cache
def _cached_bundle() -> dict[str, Any]:
    return load_model()


def get_bundle() -> dict[str, Any]:
    """Dépendance FastAPI : fournit le bundle {model, streets}.

    Renvoie 503 si le modèle n'a pas encore été entraîné (``make train``).
    """
    try:
        return _cached_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="Modèle indisponible — lancez `make train`."
        ) from exc


def _niveau(prob_libre: float) -> str:
    """Traduit la probabilité de place libre en niveau lisible."""
    if prob_libre >= 0.5:
        return "élevée"
    if prob_libre >= 0.25:
        return "moyenne"
    return "faible"


@app.get("/health")
def health() -> dict[str, str]:
    """Sonde de disponibilité."""
    return {"status": "ok"}


@app.get("/streets", response_model=StreetsResponse)
def streets(bundle: dict[str, Any] = Depends(get_bundle)) -> StreetsResponse:
    """Liste les rues connues du modèle (valeurs valides pour ``rue``)."""
    names = bundle["streets"]
    return StreetsResponse(count=len(names), streets=names)


@app.get("/predict", response_model=PredictResponse)
def predict(
    rue: str = Query(..., description="Nom de la rue (voir /streets)"),
    heure: int = Query(..., ge=0, le=23),
    jour: int = Query(..., ge=0, le=6, description="0 = lundi … 6 = dimanche"),
    minute: int = Query(0, ge=0, le=59),
    mois: int = Query(6, ge=1, le=12),
    bundle: dict[str, Any] = Depends(get_bundle),
) -> PredictResponse:
    """Prédit l'occupation et la probabilité de trouver une place."""
    streets_list = bundle["streets"]
    if rue not in streets_list:
        raise HTTPException(status_code=404, detail=f"Rue inconnue : {rue!r}. Voir /streets.")

    row = {
        "street_code": list(streets_list).index(rue),
        "hour": heure,
        "minute": minute,
        "day_of_week": jour,
        "is_weekend": int(jour >= 5),
        "month": mois,
    }
    occ = float(bundle["model"].predict(pd.DataFrame([row])[FEATURE_COLS])[0])
    occ = min(max(occ, 0.0), 1.0)
    libre = 1.0 - occ
    return PredictResponse(
        rue=rue,
        heure=heure,
        jour=jour,
        occupancy_rate=occ,
        probabilite_libre=libre,
        niveau=_niveau(libre),
    )
