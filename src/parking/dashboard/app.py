"""Dashboard Streamlit — prédicteur d'occupation de stationnement (Melbourne).

Sélection rue / jour / heure → probabilité de trouver une place, + carte des rues
colorées (vert = place probable, rouge = plein).

Lancement : ``streamlit run src/parking/dashboard/app.py`` (ou ``make dashboard``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from parking.config import DATA_INTERIM
from parking.features import FEATURE_COLS

JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
GEO_FILE = DATA_INTERIM / "melbourne_street_geo.parquet"


def predict_all(
    bundle: dict[str, Any], geo: pd.DataFrame, jour: int, heure: int, minute: int = 0, mois: int = 6
) -> pd.DataFrame:
    """Prédit l'occupation de toutes les rues connues pour un moment donné.

    Args:
        bundle: Bundle ``{model, streets}`` chargé via ``parking.model.load_model``.
        geo: Table ``street_name, lat, lon`` (coordonnées des rues).
        jour: Jour de la semaine (0 = lundi).
        heure: Heure (0-23).
        minute: Minute du créneau.
        mois: Mois (1-12).

    Returns:
        DataFrame ``street_name, lat, lon, occupancy_rate, probabilite_libre``.
    """
    streets = list(bundle["streets"])
    rows = [
        {
            "street_code": code,
            "hour": heure,
            "minute": minute,
            "day_of_week": jour,
            "is_weekend": int(jour >= 5),
            "month": mois,
        }
        for code in range(len(streets))
    ]
    occ = bundle["model"].predict(pd.DataFrame(rows)[FEATURE_COLS]).clip(0, 1)
    out = pd.DataFrame({"street_name": streets, "occupancy_rate": occ})
    out["probabilite_libre"] = 1 - out["occupancy_rate"]
    return out.merge(geo[["street_name", "lat", "lon"]], on="street_name", how="left")


def _color(prob_libre: float) -> list[int]:
    """Rouge (plein) → jaune → vert (place probable), en RGBA.

    L'occupation étant globalement élevée à Melbourne, la probabilité de place
    libre se concentre dans le bas de l'échelle. On étale donc la rampe sur
    ``[0, 0.5]`` pour rendre les variations heure/jour bien visibles.
    """
    t = min(max(prob_libre / 0.5, 0.0), 1.0)  # 0 = plein, 1 = ≥50 % libre
    red = int(255 * (1 - t))
    green = int(200 * t)
    return [red, green, 50, 180]


def main() -> None:
    """Rend le dashboard Streamlit."""
    import pydeck as pdk
    import streamlit as st

    from parking.model import load_model

    st.set_page_config(page_title="Parking Predictor — Melbourne", page_icon="🅿️", layout="wide")
    st.title("🅿️ Prédicteur de stationnement — Melbourne")

    try:
        bundle = load_model()
    except FileNotFoundError:
        st.error("Modèle introuvable. Lance `make train` d'abord.")
        return
    geo = pd.read_parquet(GEO_FILE)

    col = st.sidebar
    jour = col.selectbox("Jour", range(7), format_func=lambda i: JOURS[i])
    heure = col.slider("Heure", 0, 23, 18)
    rue = col.selectbox("Rue (focus)", sorted(bundle["streets"]))

    df = predict_all(bundle, geo, jour, heure)

    one = df[df["street_name"] == rue].iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Probabilité de trouver une place", f"{one['probabilite_libre']:.0%}")
    c2.metric("Taux d'occupation prédit", f"{one['occupancy_rate']:.0%}")

    df_map = df.dropna(subset=["lat", "lon"]).copy()
    df_map["color"] = df_map["probabilite_libre"].apply(_color)
    # Colonne texte en pourcentage pour le tooltip + rayon proportionnel.
    df_map["libre_pct"] = (df_map["probabilite_libre"] * 100).round().astype(int).astype(str) + " %"
    df_map["radius"] = 40 + df_map["probabilite_libre"] * 160
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=df_map["lat"].mean(), longitude=df_map["lon"].mean(), zoom=12
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius="radius",
                    pickable=True,
                )
            ],
            tooltip={"text": "{street_name}\nPlace libre : {libre_pct}"},
        )
    )
    st.caption("🟢 place probable · 🔴 plein — prédiction du modèle pour le créneau choisi.")

    # Top des rues où se garer maintenant : rend le changement heure/jour évident.
    st.subheader(f"Meilleures rues — {JOURS[jour]} {heure}h")
    top = df.sort_values("probabilite_libre", ascending=False).head(10).copy()
    top["Place libre"] = (top["probabilite_libre"] * 100).round().astype(int).astype(str) + " %"
    st.dataframe(
        top[["street_name", "Place libre"]].rename(columns={"street_name": "Rue"}),
        hide_index=True,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
