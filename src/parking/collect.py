"""Collecteur d'historique des parkings de Bordeaux.

Le flux open data ne donne que l'instant présent. Pour entraîner un modèle de
prévision, on accumule nous-mêmes l'historique : à chaque exécution, on relève
l'état courant et on l'ajoute à un fichier parquet versionné.

Exécution : ``python -m parking.collect`` (lancé périodiquement par GitHub
Actions — voir ``.github/workflows/collect.yml``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from parking.bordeaux import fetch_live_fresh
from parking.config import DATA_DIR

#: Fichier d'historique (versionné — voir .gitignore).
HISTORY_FILE: Path = DATA_DIR / "history" / "bordeaux_parkings.parquet"

#: Colonnes conservées dans l'historique.
_COLS = ["ident", "nom", "secteur", "total", "libres", "occupancy_rate", "mdate", "fetched_at"]


def collect_snapshot(history_file: Path = HISTORY_FILE) -> tuple[int, int]:
    """Relève l'état courant des parkings et l'ajoute à l'historique.

    La déduplication se fait sur ``(ident, mdate)`` : si un parking n'a pas été
    rafraîchi à la source depuis la dernière collecte, on ne duplique pas la ligne.

    Args:
        history_file: Fichier parquet d'historique (créé s'il n'existe pas).

    Returns:
        Tuple ``(n_releves, n_total)`` : lignes relevées cette fois, total en base.
    """
    snap = fetch_live_fresh()[_COLS]

    if history_file.exists():
        combined = pd.concat([pd.read_parquet(history_file), snap], ignore_index=True)
    else:
        combined = snap
    combined = combined.drop_duplicates(subset=["ident", "mdate"]).reset_index(drop=True)

    history_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(history_file, index=False)
    return len(snap), len(combined)


def main() -> None:
    """Point d'entrée CLI : effectue une collecte et affiche le bilan.

    Si l'API Bordeaux reste injoignable malgré les tentatives, on sort
    proprement (sans erreur) : une collecte manquée n'est pas critique et on
    évite ainsi les alertes d'échec inutiles. La prochaine collecte réessaiera.
    """
    try:
        n_new, n_total = collect_snapshot()
    except requests.RequestException as exc:
        print(f"API Bordeaux injoignable, collecte ignorée cette fois : {exc}")
        return
    print(f"Relevé : {n_new} parkings · historique total : {n_total:,} lignes -> {HISTORY_FILE}")


if __name__ == "__main__":
    main()
