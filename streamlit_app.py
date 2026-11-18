import streamlit as st

st.set_page_config(
    page_title="Nutri-Score — Rétro-ingénierie & Arbre de décision",
    page_icon=":material/nutrition:",
    layout="wide",
)

pages = [
    st.Page(
        "app_pages/simulator.py",
        title="Simulateur & comparateur",
        icon=":material/tune:",
        help="Simulateur temps réel confrontant le calculateur officiel 2023 à l'arbre de décision CART.",
        default=True,
    ),
    st.Page(
        "app_pages/explorer.py",
        title="Explorateur de produits",
        icon=":material/search:",
        help="Recherche et diagnostic d'exceptions parmi les 14 611 céréales de petit-déjeuner.",
    ),
    st.Page(
        "app_pages/rules.py",
        title="Rétro-ingénierie des seuils",
        icon=":material/balance:",
        help="Confrontation des seuils appris par l'arbre aux tables de l'Arrêté officiel 2025.",
    ),
    st.Page(
        "app_pages/cross_category.py",
        title="Évaluation multi-catégories",
        icon=":material/category:",
        help="Test de généralisation sur les biscuits et illustration de la dérogation fromage.",
    ),
]

nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.caption("Projet Data Mining — EMSI 5IIR")
    st.caption("Taha Ait Baissi & Yassine Amjad")
    st.caption("Enseignante : Pr. Hiba El Balbali")

nav.run()
