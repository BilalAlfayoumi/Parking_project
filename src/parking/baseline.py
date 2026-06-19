"""Baseline naïve : moyenne historique d'occupation par rue et par heure.

Référence obligatoire — tout modèle ML doit battre cette baseline (semaine 3).
"""

from __future__ import annotations

import pandas as pd


class BaselineModel:
    """Prédit l'occupation comme la moyenne historique (rue × heure × jour).

    C'est la référence de comparaison imposée par le plan : un modèle ML n'a de
    valeur que s'il bat cette baseline.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaselineModel:
        """Mémorise la moyenne de la cible par groupe (rue, heure, jour).

        Args:
            X: Matrice de features contenant au moins rue, heure et jour.
            y: Cible (taux d'occupation ou état libre/occupé).

        Returns:
            L'instance ``self``, ajustée.
        """
        raise NotImplementedError("À implémenter en semaine 3.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Renvoie la moyenne historique apprise pour chaque ligne de ``X``.

        Args:
            X: Matrice de features.

        Returns:
            Prédictions alignées sur l'index de ``X``.
        """
        raise NotImplementedError("À implémenter en semaine 3.")
