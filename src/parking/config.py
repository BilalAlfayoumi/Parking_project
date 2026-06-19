"""Configuration centralisée : chemins du projet et constantes.

Tous les modules importent les chemins depuis ce fichier — aucun chemin en dur
ailleurs dans le code.
"""

from __future__ import annotations

from pathlib import Path

# Racine du projet : remonte de src/parking/config.py -> racine
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Données
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_INTERIM: Path = DATA_DIR / "interim"
DATA_PROCESSED: Path = DATA_DIR / "processed"

# Modèles entraînés
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_FILE: Path = MODELS_DIR / "model.joblib"

# Constantes métier
#: Granularité d'agrégation du taux d'occupation, en minutes.
AGG_FREQ_MINUTES: int = 30
#: Seuil de probabilité au-delà duquel une place est jugée "probablement libre".
FREE_THRESHOLD: float = 0.5
