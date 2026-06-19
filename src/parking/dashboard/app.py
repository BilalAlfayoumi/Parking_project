"""Dashboard Streamlit : carte interactive de probabilité de place libre.

Stub (semaine 6) : page placeholder. La carte et les sélecteurs seront ajoutés
quand l'API renverra de vraies prédictions.

Lancement : ``streamlit run src/parking/dashboard/app.py``
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Rend la page Streamlit."""
    st.set_page_config(page_title="Parking Predictor", page_icon="🅿️")
    st.title("🅿️ Prédicteur d'occupation de stationnement")
    st.info("🚧 En construction — la carte interactive arrive en semaine 6.")

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Rue", ["(à venir)"], disabled=True)
    with col2:
        st.slider("Heure", 0, 23, 8, disabled=True)


if __name__ == "__main__":
    main()
