"""Tests de la baseline naïve (semaine 3)."""

from __future__ import annotations

import pytest

from parking.baseline import BaselineModel


@pytest.mark.skip(reason="À implémenter en semaine 3")
def test_baseline_predit_moyenne_historique() -> None:
    """La baseline doit renvoyer la moyenne apprise par groupe."""
    model = BaselineModel()
    # arrange / act / assert à compléter en semaine 3
    assert model is not None
