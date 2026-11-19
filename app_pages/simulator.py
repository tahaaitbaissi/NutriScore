import streamlit as st
import pandas as pd
import numpy as np
from src.nutriscore import compute, FEATURES
from src.app_utils import load_trained_model, INT_TO_GRADE

clf = load_trained_model()

st.header("Simulateur interactif : calculateur officiel vs arbre de décision")
st.caption("Ajustez les 7 nutriments pour observer la convergence ou la divergence entre la règle officielle 2023 et l'arbre CART.")

# Preset selection
preset = st.selectbox(
    "Profil type prédéfini",
    [
        "Personnalisé",
        "Flocons d'avoine purs (Note attendue A)",
        "Muesli fruits secs (Note attendue B)",
        "Granola chocolat & noisettes (Note attendue C)",
        "Pétales de maïs givrés au sucre (Note attendue D)",
        "Céréales fourrées chocolatées (Note attendue E)",
        "Trink Porridge (OFF applique le barème boisson)"
    ]
)

defaults = {
    "energy_kj": 1600.0,
    "sat_fat": 1.5,
    "sugars": 10.0,
    "salt": 0.20,
    "fibre": 6.0,
    "proteins": 8.0,
    "fvl_pct": 0.0
}

if preset == "Flocons d'avoine purs (Note attendue A)":
    defaults = {"energy_kj": 1500.0, "sat_fat": 1.3, "sugars": 1.0, "salt": 0.02, "fibre": 10.0, "proteins": 12.0, "fvl_pct": 0.0}
elif preset == "Muesli fruits secs (Note attendue B)":
    defaults = {"energy_kj": 1600.0, "sat_fat": 2.0, "sugars": 12.0, "salt": 0.20, "fibre": 7.0, "proteins": 8.0, "fvl_pct": 20.0}
elif preset == "Granola chocolat & noisettes (Note attendue C)":
    defaults = {"energy_kj": 1850.0, "sat_fat": 4.0, "sugars": 18.0, "salt": 0.40, "fibre": 7.0, "proteins": 8.5, "fvl_pct": 0.0}
elif preset == "Pétales de maïs givrés au sucre (Note attendue D)":
    defaults = {"energy_kj": 1650.0, "sat_fat": 0.6, "sugars": 37.0, "salt": 0.85, "fibre": 2.5, "proteins": 5.0, "fvl_pct": 0.0}
elif preset == "Céréales fourrées chocolatées (Note attendue E)":
    defaults = {"energy_kj": 1950.0, "sat_fat": 7.5, "sugars": 42.0, "salt": 0.95, "fibre": 2.0, "proteins": 4.5, "fvl_pct": 0.0}
elif preset == "Trink Porridge (OFF applique le barème boisson)":
    defaults = {"energy_kj": 1515.0, "sat_fat": 1.1, "sugars": 9.3, "salt": 0.02, "fibre": 11.0, "proteins": 11.0, "fvl_pct": 0.0}

if preset == "Trink Porridge (OFF applique le barème boisson)":
    st.info(
        "OFF classe ce produit en **boisson** (`is_beverage = 1`) : sa fiche affiche **D (score 7)** avec le "
        "barème boisson, plus sévère. Le simulateur compare ici le **barème général 2023** (A) et l'arbre (A) : "
        "c'est un conflit de catégorie, pas une erreur de nutriments.",
        icon=":material/info:",
    )

col_neg, col_pos = st.columns(2)

with col_neg:
    with st.container(border=True):
        st.subheader("Nutriments défavorables (Composante N)")
        val_energy = st.slider("Énergie (kJ / 100g)", 0.0, 3500.0, float(defaults["energy_kj"]), step=10.0)
        val_sat_fat = st.slider("Acides gras saturés (g / 100g)", 0.0, 30.0, float(defaults["sat_fat"]), step=0.1)
        val_sugars = st.slider("Sucres simples (g / 100g)", 0.0, 70.0, float(defaults["sugars"]), step=0.5)
        val_salt = st.slider("Sel (g / 100g)", 0.0, 5.0, float(defaults["salt"]), step=0.01)

with col_pos:
    with st.container(border=True):
        st.subheader("Nutriments favorables (Composante P)")
        val_fibre = st.slider("Fibres alimentaires (g / 100g)", 0.0, 30.0, float(defaults["fibre"]), step=0.2)
        val_proteins = st.slider("Protéines (g / 100g)", 0.0, 30.0, float(defaults["proteins"]), step=0.2)
        val_fvl = st.slider("Fruits, légumes, légumineuses (%)", 0.0, 100.0, float(defaults["fvl_pct"]), step=1.0)

nutrients_dict = {
    "energy_kj": val_energy,
    "sat_fat": val_sat_fat,
    "sugars": val_sugars,
    "salt": val_salt,
    "fibre": val_fibre,
    "proteins": val_proteins,
    "fvl_pct": val_fvl
}

calc_res = compute(nutrients_dict, version="2023")
official_score = calc_res["score"]
official_grade = calc_res["grade"]

input_df = pd.DataFrame([nutrients_dict])[list(FEATURES)]
tree_pred_int = clf.predict(input_df)[0]
tree_pred_grade = INT_TO_GRADE[tree_pred_int]
tree_proba = clf.predict_proba(input_df)[0]

res_col1, res_col2 = st.columns(2)

with res_col1:
    with st.container(border=True):
        st.caption("Calculateur réglementaire 2023")
        st.metric(
            label="Note officielle",
            value=official_grade,
            delta=f"Score numérique : {official_score}",
            delta_color="off"
        )
        st.write(f"Points N : `{calc_res['N']}` | Points P : `{calc_res['P']}`")
        detail_txt = (
            f"Fibres : {calc_res['p_fibre']} pts, Protéines : {calc_res['p_protein']} pts "
            f"(comptées : {calc_res['protein_counted']}), Fruits/Légumes : {calc_res['p_fvl']} pts"
        )
        st.caption(detail_txt)

with res_col2:
    with st.container(border=True):
        st.caption("Arbre de décision CART")
        st.metric(
            label="Note prédite",
            value=tree_pred_grade,
            delta=f"Probabilité classe : {tree_proba[tree_pred_int]*100:.1f} %",
            delta_color="off"
        )
        proba_df = pd.DataFrame({
            "Note": ["A", "B", "C", "D", "E"],
            "Probabilité": tree_proba
        })
        st.bar_chart(proba_df, x="Note", y="Probabilité", height=150, alt="Distribution des probabilités par note Nutri-Score")

diff_abs = abs(ord(official_grade) - ord(tree_pred_grade))
if diff_abs == 0:
    st.success("L'arbre de décision retrouve exactement la note officielle issue du barème réglementaire.", title="Accord parfait")
elif diff_abs == 1:
    st.warning(f"Note officielle : {official_grade} vs Arbre : {tree_pred_grade}. Erreur fréquente au voisinage immédiat des frontières de score.", title="Erreur de frontière (1 lettre)")
else:
    st.error(f"Note officielle : {official_grade} vs Arbre : {tree_pred_grade}. Divergence due à l'approximation en escalier de la somme linéaire N - P.", title=f"Divergence forte ({diff_abs} lettres)")
