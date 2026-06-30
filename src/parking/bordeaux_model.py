"""Prévision du remplissage des parkings de Bordeaux.

Entraîne un modèle sur l'historique collecté (``parking.collect``) pour prédire
le taux d'occupation d'un parking à une heure et un jour donnés. Le modèle est
évalué en split temporel, contre une baseline « moyenne historique ».

``python -m parking.bordeaux_model`` (cible ``make train-bordeaux``) entraîne et
sauvegarde le modèle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from parking.collect import HISTORY_FILE
from parking.config import MODELS_DIR
from parking.evaluation import evaluate

#: Modèle sérialisé des parkings Bordeaux.
BX_MODEL_FILE: Path = MODELS_DIR / "bordeaux_model.joblib"

FEATURES = ["park_code", "hour", "day_of_week", "is_weekend"]
TARGET = "occupancy_rate"
#: Clés de la baseline (moyenne historique par parking × heure × jour).
BASELINE_KEYS = ["park_code", "hour", "day_of_week"]


def load_history() -> pd.DataFrame:
    """Charge l'historique collecté et dérive les variables temporelles."""
    df = pd.read_parquet(HISTORY_FILE)
    df["mdate"] = pd.to_datetime(df["mdate"], utc=True)
    df = df.dropna(subset=["mdate", "occupancy_rate", "ident"])
    df["hour"] = df["mdate"].dt.hour
    df["day_of_week"] = df["mdate"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df.sort_values("mdate").reset_index(drop=True)


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construit X (avec ``park_code``) et renvoie la liste ordonnée des parkings.

    Args:
        df: Historique enrichi (issu de :func:`load_history`).

    Returns:
        Tuple ``(df_avec_park_code, idents)`` où ``idents[i]`` correspond au code i.
    """
    cat = pd.Categorical(df["ident"])
    out = df.copy()
    out["park_code"] = cat.codes
    return out, list(cat.categories)


def _baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Moyenne historique par (parking, heure, jour), avec repli."""
    table = train.groupby(BASELINE_KEYS)[TARGET].mean().rename("pred")
    park_mean = train.groupby("park_code")[TARGET].mean()
    pred = test.merge(table, on=BASELINE_KEYS, how="left")["pred"]
    pred = pred.fillna(test["park_code"].map(park_mean)).fillna(train[TARGET].mean())
    return pred.to_numpy()


def train_forecast(test_fraction: float = 0.25) -> dict[str, Any]:
    """Entraîne et évalue le modèle de prévision (split temporel).

    Args:
        test_fraction: Part finale (dans le temps) réservée au test.

    Returns:
        Bundle ``{"model", "parks", "scores"}``. Sauvegardé dans ``BX_MODEL_FILE``.
    """
    df, parks = make_features(load_history())
    cut = df["mdate"].quantile(1 - test_fraction)
    train, test = df[df["mdate"] < cut], df[df["mdate"] >= cut]

    model = HistGradientBoostingRegressor(max_depth=6, random_state=42)
    model.fit(train[FEATURES], train[TARGET])

    scores = {
        "model": evaluate(test[TARGET].to_numpy(), model.predict(test[FEATURES])),
        "baseline": evaluate(test[TARGET].to_numpy(), _baseline_predict(train, test)),
    }
    bundle = {"model": model, "parks": parks, "scores": scores}

    BX_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BX_MODEL_FILE)
    return bundle


def load_model() -> dict[str, Any]:
    """Charge le bundle ``{model, parks, scores}`` sérialisé."""
    return joblib.load(BX_MODEL_FILE)


def predict_occupancy(ident: str, hour: int, day_of_week: int) -> float:
    """Prédit le taux d'occupation d'un parking à un moment donné.

    Args:
        ident: Identifiant du parking (``ident``, voir l'historique).
        hour: Heure (0-23).
        day_of_week: Jour (0 = lundi … 6 = dimanche).

    Returns:
        Taux d'occupation prédit dans ``[0, 1]``.

    Raises:
        ValueError: Si le parking est inconnu du modèle.
    """
    bundle = load_model()
    parks = bundle["parks"]
    if ident not in parks:
        raise ValueError(f"Parking inconnu du modèle : {ident!r}")
    row = {
        "park_code": list(parks).index(ident),
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
    }
    pred = float(bundle["model"].predict(pd.DataFrame([row])[FEATURES])[0])
    return min(max(pred, 0.0), 1.0)


def main() -> None:
    """Point d'entrée CLI : entraîne, évalue et sauvegarde."""
    bundle = train_forecast()
    s = bundle["scores"]
    gain = (s["baseline"]["MAE"] - s["model"]["MAE"]) / s["baseline"]["MAE"]
    print(f"Parkings : {len(bundle['parks'])}")
    print(f"  Baseline : {s['baseline']}")
    print(f"  Modèle   : {s['model']}")
    print(f"  Gain MAE vs baseline : {gain:+.1%}")
    print(f"Modèle sauvegardé -> {BX_MODEL_FILE}")


if __name__ == "__main__":
    main()
