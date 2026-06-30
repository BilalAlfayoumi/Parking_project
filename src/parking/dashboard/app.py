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


def _render_melbourne(st: Any, pdk: Any) -> None:
    """Onglet Melbourne : prédiction d'occupation par le modèle."""
    from parking.model import load_model

    st.header("🇦🇺 Melbourne — prédiction d'occupation")
    try:
        bundle = load_model()
    except FileNotFoundError:
        st.error("Modèle introuvable. Lance `make train` d'abord.")
        return
    geo = pd.read_parquet(GEO_FILE)

    c1, c2, c3 = st.columns(3)
    jour = c1.selectbox("Jour", range(7), format_func=lambda i: JOURS[i])
    heure = c2.slider("Heure", 0, 23, 18)
    rue = c3.selectbox("Rue (focus)", sorted(bundle["streets"]))

    df = predict_all(bundle, geo, jour, heure)
    one = df[df["street_name"] == rue].iloc[0]
    m1, m2 = st.columns(2)
    m1.metric("Probabilité de trouver une place", f"{one['probabilite_libre']:.0%}")
    m2.metric("Taux d'occupation prédit", f"{one['occupancy_rate']:.0%}")

    df_map = df.dropna(subset=["lat", "lon"]).copy()
    df_map["color"] = df_map["probabilite_libre"].apply(_color)
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

    st.subheader(f"Meilleures rues — {JOURS[jour]} {heure}h")
    top = df.sort_values("probabilite_libre", ascending=False).head(10).copy()
    top["Place libre"] = (top["probabilite_libre"] * 100).round().astype(int).astype(str) + " %"
    st.dataframe(
        top[["street_name", "Place libre"]].rename(columns={"street_name": "Rue"}),
        hide_index=True,
        use_container_width=True,
    )


def _render_bordeaux_forecast(st: Any, parking: str, ident: str) -> None:
    """Section prévision : remplissage prédit du parking à une heure/jour choisis.

    Le modèle est entraîné à la volée sur l'historique collecté (mis en cache),
    pour rester à jour sans dépendre d'un fichier modèle versionné.
    """
    from parking.bordeaux_model import FEATURES, train_forecast

    @st.cache_resource(ttl=3600)
    def _bundle() -> dict[str, Any]:
        return train_forecast()

    st.divider()
    st.subheader("🔮 Prévision du remplissage")
    try:
        bundle = _bundle()
    except Exception as exc:  # noqa: BLE001 — pas assez d'historique, on informe
        st.info(f"Prévision indisponible (historique insuffisant) : {exc}")
        return
    if ident not in bundle["parks"]:
        st.info("Pas encore assez d'historique pour ce parking.")
        return

    c1, c2 = st.columns(2)
    jour = c1.selectbox("Jour", range(7), format_func=lambda i: JOURS[i], key="bx_jour")
    heure = c2.slider("Heure", 0, 23, 18, key="bx_heure")
    row = {
        "park_code": list(bundle["parks"]).index(ident),
        "hour": heure,
        "day_of_week": jour,
        "is_weekend": int(jour >= 5),
    }
    occ = float(bundle["model"].predict(pd.DataFrame([row])[FEATURES])[0])
    occ = min(max(occ, 0.0), 1.0)
    st.metric(f"Disponibilité prévue — {parking}, {JOURS[jour]} {heure}h", f"{1 - occ:.0%}")
    st.caption(
        f"Modèle entraîné sur l'historique collecté · R² {bundle['scores']['model']['R2']:.2f}"
    )


def _render_bordeaux(st: Any, pdk: Any) -> None:
    """Onglet Bordeaux : disponibilité des parkings en temps réel + prévision."""
    from parking.bordeaux import fetch_live_fresh

    st.header("🇫🇷 Bordeaux — parkings en temps réel")

    @st.cache_data(ttl=120)
    def _live() -> pd.DataFrame:
        return fetch_live_fresh()

    if st.button("🔄 Rafraîchir"):
        _live.clear()
    try:
        df = _live()
    except Exception as exc:  # noqa: BLE001 — on affiche l'erreur réseau à l'utilisateur
        st.error(f"Données Bordeaux indisponibles : {exc}")
        return
    if df.empty:
        st.warning("Aucun parking à jour pour le moment.")
        return

    maj = df["mdate"].max()
    st.caption(f"{len(df)} parkings raccordés · dernière mise à jour {maj:%d/%m %H:%M} UTC")

    parking = st.selectbox("Parking (focus)", sorted(df["nom"]))
    one = df[df["nom"] == parking].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Places libres", f"{int(one['libres'])} / {int(one['total'])}")
    m2.metric("Disponibilité", f"{one['free_rate']:.0%}")
    m3.metric("Secteur", str(one["secteur"]).replace("_", " ").title())

    _render_bordeaux_forecast(st, parking, str(one["ident"]))

    df_map = df.dropna(subset=["lat", "lon"]).copy()
    df_map["color"] = df_map["free_rate"].apply(_color)
    df_map["info"] = df_map["nom"] + " — " + df_map["libres"].astype(int).astype(str) + " libres"
    df_map["radius"] = 60 + df_map["free_rate"] * 200
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
            tooltip={"text": "{info}"},
        )
    )
    st.caption("🟢 beaucoup de place · 🔴 presque plein — données live Bordeaux Métropole.")

    st.subheader("Où se garer maintenant")
    table = df.sort_values("libres", ascending=False)[["nom", "secteur", "libres", "total"]].copy()
    table = table.rename(
        columns={"nom": "Parking", "secteur": "Secteur", "libres": "Libres", "total": "Total"}
    )
    st.dataframe(table.head(15), hide_index=True, use_container_width=True)


def main() -> None:
    """Rend le dashboard Streamlit (2 onglets : Melbourne, Bordeaux)."""
    import pydeck as pdk
    import streamlit as st

    st.set_page_config(page_title="Parking Predictor", page_icon="🅿️", layout="wide")
    st.title("🅿️ Prédicteur de stationnement")
    tab_mel, tab_bdx = st.tabs(["Melbourne (prédiction)", "Bordeaux (temps réel)"])
    with tab_mel:
        _render_melbourne(st, pdk)
    with tab_bdx:
        _render_bordeaux(st, pdk)


if __name__ == "__main__":
    main()
