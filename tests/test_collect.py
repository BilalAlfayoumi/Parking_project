"""Tests du collecteur d'historique (sans appel réseau réel)."""

from __future__ import annotations

import pandas as pd

from parking import collect


def _fake_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ident": ["P1", "P2"],
            "nom": ["Alpha", "Beta"],
            "secteur": ["CENTRE", "PERIPHERIE"],
            "total": [100, 200],
            "libres": [40, 50],
            "occupancy_rate": [0.6, 0.75],
            "mdate": pd.to_datetime(["2026-06-22T10:00Z", "2026-06-22T10:00Z"]),
            "fetched_at": pd.to_datetime(["2026-06-22T10:01Z", "2026-06-22T10:01Z"]),
        }
    )


def test_collect_creates_then_appends(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Première collecte crée le fichier ; une 2ᵉ avec nouveau mdate s'ajoute."""
    hist = tmp_path / "h.parquet"
    monkeypatch.setattr(collect, "fetch_live_fresh", _fake_snapshot)
    n_new, n_total = collect.collect_snapshot(hist)
    assert (n_new, n_total) == (2, 2)

    # Nouveau relevé (mdate différent) → +2 lignes
    later = _fake_snapshot()
    later["mdate"] = pd.to_datetime(["2026-06-22T10:15Z", "2026-06-22T10:15Z"])
    monkeypatch.setattr(collect, "fetch_live_fresh", lambda: later)
    _, n_total2 = collect.collect_snapshot(hist)
    assert n_total2 == 4


def test_collect_dedups_same_mdate(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Relever deux fois le même mdate ne duplique pas les lignes."""
    hist = tmp_path / "h.parquet"
    monkeypatch.setattr(collect, "fetch_live_fresh", _fake_snapshot)
    collect.collect_snapshot(hist)
    _, n_total = collect.collect_snapshot(hist)  # même snapshot
    assert n_total == 2
