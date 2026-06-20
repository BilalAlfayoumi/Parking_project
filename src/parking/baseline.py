"""Baseline naïve : moyenne historique d'occupation par rue × heure × jour.

Référence imposée par le plan : tout modèle ML doit battre cette baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Clés d'agrégation de la moyenne historique.
KEYS = ["street_code", "hour", "day_of_week"]


class BaselineModel:
    """Prédit l'occupation comme la moyenne historique (rue × heure × jour).

    Repli hiérarchique pour les combinaisons absentes du train : moyenne de la
    rue, puis moyenne globale.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaselineModel:
        """Mémorise les moyennes par groupe.

        Args:
            X: Features contenant au moins :data:`KEYS`.
            y: Cible (taux d'occupation).

        Returns:
            L'instance ``self``, ajustée.
        """
        d = X[KEYS].copy()
        d["_y"] = np.asarray(y)
        self.table_ = d.groupby(KEYS)["_y"].mean().rename("pred").reset_index()
        self.street_mean_ = d.groupby("street_code")["_y"].mean()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit la moyenne historique apprise pour chaque ligne de ``X``."""
        pred = X[KEYS].merge(self.table_, on=KEYS, how="left")["pred"]
        pred = pred.fillna(X["street_code"].map(self.street_mean_)).fillna(self.global_mean_)
        return pred.to_numpy()
