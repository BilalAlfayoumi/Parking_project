"""Entraînement, persistance et CLI d'entraînement du modèle.

``python -m parking.model`` (cible ``make train``) : construit/charge la table
d'occupation, entraîne la baseline et le modèle, évalue en split temporel, et
sauvegarde le modèle (joblib) avec la correspondance rue -> code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from parking.baseline import BaselineModel
from parking.config import MODEL_FILE
from parking.evaluation import compare_to_baseline, temporal_split
from parking.features import (
    OCCUPANCY_FILE,
    build_occupancy_dataset,
    make_features,
)

#: Coupure train/test (entraînement jan-sep, test oct-déc).
CUTOFF = "2019-10-01"
#: Profondeur retenue par la validation curve (semaine 3).
MAX_DEPTH = 12


def train(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    """Entraîne le modèle d'occupation (HistGradientBoosting)."""
    model = HistGradientBoostingRegressor(max_depth=MAX_DEPTH, random_state=42)
    model.fit(X, y)
    return model


def save_model(model: Any, streets: list[str], path: Path = MODEL_FILE) -> None:
    """Sérialise le modèle et la liste des rues (ordre des codes) via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "streets": list(streets)}, path)


def load_model(path: Path = MODEL_FILE) -> dict[str, Any]:
    """Charge le bundle ``{"model", "streets"}`` sérialisé."""
    return joblib.load(path)


def _load_occupancy() -> pd.DataFrame:
    """Charge la table d'occupation, la construit depuis le zip si absente."""
    if OCCUPANCY_FILE.exists():
        return pd.read_parquet(OCCUPANCY_FILE)
    print("Table d'occupation absente -> construction depuis le zip (peut être long)...")
    return build_occupancy_dataset()


def main() -> None:
    """Point d'entrée CLI : entraîne, évalue et sauvegarde le modèle."""
    occ = _load_occupancy()
    streets = list(pd.Categorical(occ["street_name"]).categories)
    X, y = make_features(occ)

    train_mask = temporal_split(occ, CUTOFF)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    baseline = BaselineModel().fit(X_train, y_train)
    model = train(X_train, y_train)

    scores = compare_to_baseline(y_test.to_numpy(), model.predict(X_test), baseline.predict(X_test))
    print(f"Entraînement sur {len(X_train):,} lignes, test sur {len(X_test):,}.")
    print(f"  Baseline : {scores['baseline']}")
    print(f"  Modèle   : {scores['model']}")
    gain = (scores["baseline"]["MAE"] - scores["model"]["MAE"]) / scores["baseline"]["MAE"]
    print(f"  Gain MAE vs baseline : {gain:+.1%}")

    save_model(model, streets)
    print(f"Modèle sauvegardé -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
