"""Ingestion des parkings de Bordeaux Métropole (temps réel).

Source : dataset ``st_park_p`` du portail Opendatasoft de Bordeaux Métropole,
qui expose pour chaque parking hors voirie sa capacité et ses **places libres**
en temps réel (pour les parkings raccordés au flux, ~rafraîchi toutes les 2-3 min).

Ce flux ne fournit que l'instantané courant : pour entraîner un modèle de
prévision, on accumule l'historique nous-mêmes (voir ``parking.collect``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import requests

BASE_URL = (
    "https://opendata.bordeaux-metropole.fr/api/explore/v2.1/catalog/datasets/st_park_p/records"
)
_HEADERS = {"User-Agent": "parking-project/1.0 (research)"}
_SELECT = "ident,nom,secteur,exploit,total,libres,geo_point_2d,mdate"


def fetch_live(timeout: int = 30) -> pd.DataFrame:
    """Récupère l'état temps réel des parkings de Bordeaux raccordés au flux.

    Args:
        timeout: Délai maximal de la requête HTTP, en secondes.

    Returns:
        DataFrame, une ligne par parking, avec colonnes :
        ``ident, nom, secteur, exploit, total, libres, lat, lon, mdate,
        occupancy_rate, free_rate, fetched_at``.
    """
    params = {
        "where": "libres IS NOT NULL AND total > 0",
        "select": _SELECT,
        "limit": "100",
    }
    resp = requests.get(BASE_URL, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    rows = resp.json().get("results", [])

    df = pd.json_normalize(rows)
    if df.empty:
        return df

    df = df.rename(columns={"geo_point_2d.lat": "lat", "geo_point_2d.lon": "lon"})
    df["mdate"] = pd.to_datetime(df["mdate"], errors="coerce", utc=True)
    df["occupancy_rate"] = (1 - df["libres"] / df["total"]).clip(0, 1)
    df["free_rate"] = 1 - df["occupancy_rate"]
    df["fetched_at"] = datetime.now(UTC)

    cols = [
        "ident",
        "nom",
        "secteur",
        "exploit",
        "total",
        "libres",
        "lat",
        "lon",
        "mdate",
        "occupancy_rate",
        "free_rate",
        "fetched_at",
    ]
    return df[cols].sort_values("nom").reset_index(drop=True)


def fetch_live_fresh(max_age_hours: int = 6, timeout: int = 30) -> pd.DataFrame:
    """Comme :func:`fetch_live`, mais ne garde que les parkings réellement à jour.

    Certains parkings restent dans le flux avec une donnée figée (capteur non
    raccordé) : on les écarte en filtrant sur l'ancienneté de ``mdate``.

    Args:
        max_age_hours: Ancienneté maximale de la dernière mise à jour, en heures.
        timeout: Délai maximal de la requête HTTP, en secondes.

    Returns:
        DataFrame filtré aux parkings frais.
    """
    df = fetch_live(timeout=timeout)
    if df.empty:
        return df
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=max_age_hours)
    return df[df["mdate"] >= cutoff].reset_index(drop=True)
