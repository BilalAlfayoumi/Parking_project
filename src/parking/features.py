"""Nettoyage des capteurs, reconstruction de l'occupation et features ML.

Pipeline : événements bruts -> nettoyage -> taux d'occupation par rue et par
tranche de 30 min -> matrice de features ``X`` + cible ``y``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from parking.config import AGG_FREQ_MINUTES, DATA_PROCESSED
from parking.ingestion import load_events, street_sizes

#: Durée de stationnement maximale plausible (minutes) ; au-delà = départ manquant.
MAX_DURATION_MIN = 720

#: Fichier de la table features+target consolidée.
OCCUPANCY_FILE = DATA_PROCESSED / "occupancy_2019.parquet"

#: Features utilisées par le modèle (la rue est encodée en code entier).
FEATURE_COLS = ["street_code", "hour", "minute", "day_of_week", "is_weekend", "month"]
TARGET_COL = "occupancy_rate"

_FREQ = f"{AGG_FREQ_MINUTES}min"


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """Retire les incohérences de capteurs (durées <= 0 ou aberrantes, NA).

    Args:
        events: Événements bruts (``arrival_time``, ``departure_time``, ``street_marker``).

    Returns:
        Événements nettoyés, avec une colonne ``duration_min``.
    """
    df = events.dropna(subset=["arrival_time", "departure_time", "street_marker"]).copy()
    df["duration_min"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60
    return df[(df["duration_min"] > 0) & (df["duration_min"] <= MAX_DURATION_MIN)]


def occupancy_for_street(events: pd.DataFrame) -> pd.DataFrame:
    """Taux d'occupation d'une rue sur une grille de 30 min.

    À un instant ``t``, l'occupation = nombre d'intervalles [arrivée, départ]
    actifs, rapporté au nombre de places. Les chevauchements sur une même place
    étant négligeables (~0,1 %), on compte les intervalles et on plafonne à 1.

    Args:
        events: Événements nettoyés d'une seule rue.

    Returns:
        DataFrame ``slot, occupancy_rate, n_bays``.
    """
    n_bays = events["street_marker"].nunique()
    start = events["arrival_time"].min().floor(_FREQ)
    end = events["departure_time"].max().ceil(_FREQ)
    grid = pd.date_range(start, end, freq=_FREQ)
    gt = grid.values.astype("datetime64[ns]")
    arr = np.sort(events["arrival_time"].values)
    dep = np.sort(events["departure_time"].values)
    occupied = np.searchsorted(arr, gt, side="right") - np.searchsorted(dep, gt, side="right")
    return pd.DataFrame(
        {"slot": grid, "occupancy_rate": np.clip(occupied / n_bays, 0, 1), "n_bays": n_bays}
    )


def add_time_features(occ: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables temporelles dérivées de ``slot``."""
    occ = occ.copy()
    occ["hour"] = occ["slot"].dt.hour
    occ["minute"] = occ["slot"].dt.minute
    occ["day_of_week"] = occ["slot"].dt.dayofweek
    occ["is_weekend"] = (occ["day_of_week"] >= 5).astype(int)
    occ["month"] = occ["slot"].dt.month
    return occ


def build_occupancy_dataset(
    year: int = 2019, min_bays: int = 10, n_batches: int = 3, save: bool = True
) -> pd.DataFrame:
    """Construit la table d'occupation pour toutes les rues bien fournies.

    Lit le zip par lots de rues (pour borner la mémoire), reconstruit
    l'occupation de chaque rue, ajoute les features temporelles, et sauvegarde.

    Args:
        year: Année du jeu de données.
        min_bays: Nombre minimal de places pour retenir une rue.
        n_batches: Nombre de lots de rues (compromis mémoire/passes de lecture).
        save: Si vrai, sauvegarde le résultat en parquet (``OCCUPANCY_FILE``).

    Returns:
        Table consolidée ``street_name, slot, occupancy_rate, n_bays`` + features.
    """
    sizes = street_sizes(year)
    streets = sizes[sizes >= min_bays].index.tolist()

    frames: list[pd.DataFrame] = []
    for batch in np.array_split(streets, n_batches):
        events = clean_events(load_events(year, streets=set(batch)))
        for name, grp in events.groupby("street_name"):
            occ = occupancy_for_street(grp)
            occ["street_name"] = name
            frames.append(occ)

    occ = add_time_features(pd.concat(frames, ignore_index=True))
    if save:
        OCCUPANCY_FILE.parent.mkdir(parents=True, exist_ok=True)
        occ.to_parquet(OCCUPANCY_FILE, index=False)
    return occ


def make_features(occ: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construit la matrice de features ``X`` et la cible ``y``.

    Args:
        occ: Table d'occupation (avec ``street_name`` et features temporelles).

    Returns:
        Tuple ``(X, y)`` : ``X`` contient :data:`FEATURE_COLS`, ``y`` le taux
        d'occupation.
    """
    df = occ.copy()
    df["street_code"] = df["street_name"].astype("category").cat.codes
    return df[FEATURE_COLS], df[TARGET_COL]
