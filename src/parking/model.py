"""Entraînement et persistance du modèle (Random Forest / XGBoost).

Semaine 3-4 : remplir l'entraînement. ``python -m parking.model`` doit entraîner
et sauvegarder le modèle (jalon `make train`, semaine 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from parking.config import MODEL_FILE


def train(X: pd.DataFrame, y: pd.Series) -> Any:
    """Entraîne le modèle de prédiction d'occupation.

    Args:
        X: Matrice de features.
        y: Cible.

    Returns:
        Le modèle entraîné (estimateur scikit-learn / XGBoost).
    """
    raise NotImplementedError("À implémenter en semaine 3-4.")


def save_model(model: Any, path: Path = MODEL_FILE) -> None:
    """Sérialise le modèle entraîné sur disque via joblib.

    Args:
        model: Modèle entraîné.
        path: Destination du fichier ``.joblib``.
    """
    raise NotImplementedError("À implémenter en semaine 4.")


def load_model(path: Path = MODEL_FILE) -> Any:
    """Charge un modèle sérialisé depuis le disque.

    Args:
        path: Chemin du fichier ``.joblib``.

    Returns:
        Le modèle désérialisé.
    """
    raise NotImplementedError("À implémenter en semaine 4.")


def main() -> None:
    """Point d'entrée CLI : entraîne puis sauvegarde le modèle (`make train`)."""
    raise NotImplementedError("À implémenter en semaine 4.")


if __name__ == "__main__":
    main()
