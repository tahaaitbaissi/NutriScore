import streamlit as st
import pandas as pd
from src.nutriscore import FEATURES
from src.app_utils import load_datasets, load_trained_model, INT_TO_GRADE

train_df, val_df, test_df, clean_df, dis_df = load_datasets()
clf = load_trained_model()

st.header("Explorateur de produits et diagnostic des exceptions")
st.caption("Filtrez les 14 611 céréales propres d'Open Food Facts pour auditer les anomalies de saisie et les désaccords.")

# Fast predictions calculation
clean_view = clean_df.copy()
clean_view["pred_int"] = clf.predict(clean_view[list(FEATURES)])
clean_view["pred_grade"] = clean_view["pred_int"].map(INT_TO_GRADE)
clean_view["tree_error"] = clean_view["pred_grade"] != clean_view["grade"].astype(str)
dis_codes = set(dis_df["code"])
clean_view["calc_disagreement"] = clean_view["code"].isin(dis_codes)

# Search input
search_query = st.text_input(
    "Recherche textuelle",
    type="search",
    placeholder="Rechercher par nom de produit ou par code-barres..."
)

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    filter_grade = st.multiselect(
        "Note officielle Open Food Facts",
        ["A", "B", "C", "D", "E"],
        default=["A", "B", "C", "D", "E"]
    )

with col_f2:
    filter_tree = st.segmented_control(
        "Statut de l'arbre",
        ["Tous", "Erreurs uniquement", "Bien classés"],
        default="Tous"
    )

with col_f3:
    filter_anomaly = st.segmented_control(
        "Type d'anomalie",
        ["Toutes", "Incohérence physique (R7)", "Désaccord calculateur"],
        default="Toutes"
    )

# Filter application
filtered = clean_view[clean_view["grade"].astype(str).isin(filter_grade)]

if search_query:
    q = search_query.strip().lower()
    filtered = filtered[
        filtered["product_name"].str.lower().str.contains(q, na=False) |
        filtered["code"].str.contains(q, na=False)
    ]

if filter_tree == "Erreurs uniquement":
    filtered = filtered[filtered["tree_error"]]
elif filter_tree == "Bien classés":
    filtered = filtered[~filtered["tree_error"]]

if filter_anomaly == "Incohérence physique (R7)":
    filtered = filtered[filtered["suspect_consistency"]]
elif filter_anomaly == "Désaccord calculateur":
    filtered = filtered[filtered["calc_disagreement"]]

st.write(f"**{len(filtered):,} produits** trouvés.")

cols_show = [
    "code", "product_name", "grade", "pred_grade", "off_score",
    "score_v2023", "sugars", "sat_fat", "fibre", "salt",
    "suspect_consistency", "countries"
]

st.dataframe(
    filtered[cols_show].head(250),
    hide_index=True,
    column_config={
        "code": st.column_config.TextColumn("Code-barres"),
        "product_name": st.column_config.TextColumn("Nom du produit"),
        "grade": st.column_config.TextColumn("Note OFF"),
        "pred_grade": st.column_config.TextColumn("Note arbre"),
        "off_score": st.column_config.NumberColumn("Score OFF"),
        "score_v2023": st.column_config.NumberColumn("Score 2023"),
        "sugars": st.column_config.NumberColumn("Sucres (g)"),
        "sat_fat": st.column_config.NumberColumn("Gras sat. (g)"),
        "fibre": st.column_config.NumberColumn("Fibres (g)"),
        "salt": st.column_config.NumberColumn("Sel (g)"),
        "suspect_consistency": st.column_config.CheckboxColumn("Incohérence R7"),
        "countries": st.column_config.TextColumn("Pays"),
    }
)

if len(filtered) > 0:
    st.subheader("Fiche détaillée du produit")
    selected_code = st.selectbox(
        "Sélectionnez un produit à auditer :",
        filtered["code"].head(50),
        format_func=lambda c: f"{filtered.loc[filtered['code'] == c, 'product_name'].values[0]} ({c})"
    )
    if selected_code:
        prod = clean_view[clean_view["code"] == selected_code].iloc[0]
        
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            with st.container(border=True):
                st.write(f"**Produit** : {prod['product_name']}")
                st.caption(f"Code-barres : `{prod['code']}` | Pays : `{prod['countries']}`")
                st.write(f"- Note déclarée OFF : `{prod['grade']}` (Score : `{prod['off_score']}`)")
                st.write(f"- Note prédite par l'arbre : `{prod['pred_grade']}`")
                st.write(f"- Recalcul officiel 2023 : `{prod['grade_v2023']}` (Score : `{prod['score_v2023']}`)")
                st.write(f"- Recalcul officiel 2021 : `{prod['grade_v2021']}` (Score : `{prod['score_v2021']}`)")
                st.write(f"- Millésime identifié : `{prod['nutriscore_version_guess']}`")

        with detail_col2:
            with st.container(border=True):
                st.write("**Valeurs nutritionnelles (pour 100g)**")
                st.write(f"- Énergie : `{prod['energy_kj']:.0f} kJ`")
                st.write(f"- Acides gras saturés : `{prod['sat_fat']:.2f} g`")
                st.write(f"- Sucres : `{prod['sugars']:.2f} g`")
                st.write(f"- Sel : `{prod['salt']:.2f} g`")
                st.write(f"- Fibres : `{prod['fibre']:.2f} g`")
                st.write(f"- Protéines : `{prod['proteins']:.2f} g`")
                st.write(f"- % Fruits/légumes/noix : `{prod['fvl_pct']:.1f} %`")

