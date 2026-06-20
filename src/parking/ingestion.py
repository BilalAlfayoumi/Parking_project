"""Chargement des événements de stationnement bruts (capteurs Melbourne).

Les jeux historiques de Melbourne ne sont disponibles qu'en archive ZIP (ex.
2019 : 684 Mo, ~42,7 M lignes, CSV de 7,45 Go décompressé). On lit donc le CSV
*en streaming* depuis le zip, par chunks, pour ne jamais tout charger en mémoire.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from parking.config import DATA_RAW

#: Archives brutes par année (fichiers gitignorés sous ``data/raw/``).
RAW_ZIP: dict[int, Path] = {
    2019: DATA_RAW / "on-street-parking-2019.zip",
}

#: Format des horodatages Melbourne, ex. ``04/16/2019 02:14:47 PM``.
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

#: Colonnes brutes utiles (avant normalisation snake_case).
_USECOLS = ["StreetName", "StreetMarker", "ArrivalTime", "DepartureTime"]


def _snake(name: str) -> str:
    """Normalise un nom de colonne en snake_case."""
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name.strip())
    return re.sub(r"[\s\-]+", "_", s).lower()


def _find_csv_member(zf: zipfile.ZipFile) -> str:
    """Nom du vrai CSV de l'archive (ignore les fichiers parasites macOS)."""
    members = [
        n
        for n in zf.namelist()
        if n.lower().endswith(".csv") and not n.split("/")[-1].startswith("._")
    ]
    if not members:
        raise FileNotFoundError("Aucun fichier .csv exploitable dans l'archive.")
    return members[0]


def stream_chunks(year: int, chunksize: int = 1_000_000) -> Iterator[pd.DataFrame]:
    """Itère sur le CSV du zip par chunks (colonnes utiles, snake_case).

    Args:
        year: Année du jeu de données.
        chunksize: Nombre de lignes par chunk (contrôle la mémoire).

    Yields:
        Des chunks de DataFrame aux colonnes ``street_name, street_marker,
        arrival_time, departure_time`` (chaînes brutes, dates non parsées).
    """
    zip_path = RAW_ZIP[year]
    with zipfile.ZipFile(zip_path) as zf, zf.open(_find_csv_member(zf)) as f:
        reader = pd.read_csv(f, usecols=_USECOLS, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            chunk.columns = [_snake(c) for c in chunk.columns]
            yield chunk


def load_events(year: int, streets: set[str] | None = None) -> pd.DataFrame:
    """Charge les événements de stationnement d'une année, dates parsées.

    Args:
        year: Année du jeu de données.
        streets: Si fourni, ne garde que ces rues (``street_name``). Indispensable
            pour borner la mémoire : charger toutes les rues d'un coup est lourd.

    Returns:
        DataFrame ``street_name, street_marker, arrival_time, departure_time``
        avec les horodatages convertis en ``datetime``.
    """
    parts = []
    for chunk in stream_chunks(year):
        parts.append(chunk if streets is None else chunk[chunk["street_name"].isin(streets)])
    df = pd.concat(parts, ignore_index=True)
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], format=DATE_FORMAT, errors="coerce")
    df["departure_time"] = pd.to_datetime(df["departure_time"], format=DATE_FORMAT, errors="coerce")
    return df


def street_sizes(year: int, frac: float = 0.05, seed: int = 42) -> pd.Series:
    """Estime le nombre de places (``street_marker`` distincts) par rue.

    Calculé sur un échantillon pour rester rapide ; sert à sélectionner les rues
    suffisamment fournies pour une agrégation fiable.

    Args:
        year: Année du jeu de données.
        frac: Fraction de lignes échantillonnées par chunk.
        seed: Graine de reproductibilité.

    Returns:
        Série indexée par ``street_name`` donnant le nombre de places.
    """
    parts = []
    for i, chunk in enumerate(stream_chunks(year, chunksize=500_000)):
        parts.append(chunk.sample(frac=frac, random_state=seed + i))
    sample = pd.concat(parts, ignore_index=True)
    return sample.groupby("street_name")["street_marker"].nunique().sort_values(ascending=False)
