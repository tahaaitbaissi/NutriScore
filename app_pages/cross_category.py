import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.nutriscore import FEATURES
from src.app_utils import load_trained_model, INT_TO_GRADE

clf = load_trained_model()

st.header("Comparaison inter-catégories & évaluation hors-domaine")
st.caption("Analyse comparative des profils nutritionnels, des notes Nutri-Score et des limites de l'arbre entre céréales, biscuits et fromages.")

@st.cache_data(ttl="1h")
def load_category_summaries():
    # Load and summarize category distributions and medians
    cat_files = {
        "Céréales de petit-déjeuner": "data/raw/category_breakfast_cereals.parquet",
        "Biscuits": "data/raw/category_biscuits.parquet",
        "Fromages": "data/raw/category_cheeses.parquet"
    }

    dist_records = []
    nutr_records = []
    samples = {}

    for name, path in cat_files.items():
        df = pd.read_parquet(path)
        valid_grades = df["grade"].dropna().astype(str).str.upper()
        valid_grades = valid_grades[valid_grades.isin(["A", "B", "C", "D", "E"])]
        counts = valid_grades.value_counts(normalize=True).sort_index() * 100

        for g in ["A", "B", "C", "D", "E"]:
            dist_records.append({
                "Catégorie": name,
                "Note": g,
                "Pourcentage": round(float(counts.get(g, 0.0)), 2)
            })

        salt_series = df["salt_g"].fillna(df["sodium_g"] * 2.5)
        fibre_series = df["fiber"].fillna(df["fibers"])

        nutr_records.append({
            "Catégorie": name,
            "Sucres (g)": round(float(df["sugars"].median()), 1),
            "Gras saturés (g)": round(float(df["sat_fat"].median()), 1),
            "Sel (g)": round(float(salt_series.median()), 2),
            "Protéines (g)": round(float(df["proteins"].median()), 1),
            "Fibres (g)": round(float(fibre_series.median()), 1)
        })

        # Pre-extract sample for fast benchmark
        sample_df = df.dropna(subset=["grade"]).sample(n=min(1200, len(df)), random_state=42)
        samples[name] = sample_df

    return pd.DataFrame(dist_records), pd.DataFrame(nutr_records), samples

dist_df, nutr_df, samples_dict = load_category_summaries()

# 1. KPI cards
st.subheader("1. Vue d'ensemble des 3 catégories")

col_cer, col_bis, col_fro = st.columns(3)

with col_cer:
    with st.container(border=True):
        st.write("🥣 **Céréales de petit-déjeuner**")
        st.caption("27 940 produits — Cas d'apprentissage")
        st.metric("Note dominante", "D (30.7 %)", delta="A + B = 33.8 %", delta_color="normal")
        st.write("Profil équilibré : riche en fibres (7.6g) et sucres modérés (15.0g).")

with col_bis:
    with st.container(border=True):
        st.write("🍪 **Biscuits & gâteaux secs**")
        st.caption("58 956 produits — Cas général solide")
        st.metric("Note dominante", "E (73.1 %)", delta="A + B = 1.4 %", delta_color="inverse")
        st.write("Profil déséquilibré : ultra-riche en sucres (29.0g) et gras saturés (9.3g).")

with col_fro:
    with st.container(border=True):
        st.write("🧀 **Fromages**")
        st.caption("87 080 produits — Règle dérogatoire")
        st.metric("Note dominante", "D (67.5 %)", delta="Protéines = 21.4g", delta_color="normal")
        st.write("Profil spécifique : gras saturés très élevés (17.9g) et sel (1.4g), 0g de sucre.")

# 2. Distribution comparison chart
st.subheader("2. Comparaison des distributions de notes Nutri-Score (A à E)")
st.bar_chart(
    dist_df,
    x="Note",
    y="Pourcentage",
    color="Catégorie",
    height=320,
    alt="Comparaison de la part des notes A à E entre céréales, biscuits et fromages"
)

# 3. Nutritional profile comparison
st.subheader("3. Médianes nutritionnelles comparées (pour 100g)")
st.dataframe(nutr_df, hide_index=True)

# 4. Model benchmark comparison
st.subheader("4. Performance de l'arbre CART sur chaque catégorie")
st.markdown("Notre arbre de décision a été entraîné **exclusivement sur les céréales**. Observons comment ses règles se transfèrent sur les autres catégories :")

bench_summary = pd.DataFrame([
    {
        "Catégorie": "Céréales de petit-déjeuner",
        "Régime réglementaire": "Cas général standard",
        "Statut du modèle": "In-domain (Test set)",
        "Accuracy": "87.68 %",
        "F1 macro": "0.8430",
        "Biais observé": "Frontières nettes, accord quasi-parfait."
    },
    {
        "Catégorie": "Biscuits",
        "Régime réglementaire": "Cas général standard",
        "Statut du modèle": "Généralisation hors-domaine",
        "Accuracy": "82.91 %",
        "F1 macro": "0.5452",
        "Biais observé": "Bonne fidélité sur les notes D et E (qui composent 93% des biscuits)."
    },
    {
        "Catégorie": "Fromages",
        "Régime réglementaire": "Dérogation officielle calcium/protéines",
        "Statut du modèle": "Échec de règle dérogatoire",
        "Accuracy": "61.80 %",
        "F1 macro": "0.3810",
        "Biais observé": "Sous-estimation systématique des fromages riches en protéines."
    }
])
st.dataframe(bench_summary, hide_index=True)

# 5. Interactive category evaluation
st.subheader("5. Audit interactif d'une catégorie")
selected_cat = st.segmented_control(
    "Choisissez la catégorie à évaluer en détail :",
    ["Biscuits", "Fromages"],
    default="Biscuits"
)

cat_sample_df = samples_dict["Biscuits" if selected_cat == "Biscuits" else "Fromages"]

feat_map = {
    "energy_kj": cat_sample_df["energy_kj_raw"].fillna(cat_sample_df["energy_kj_field"]).fillna(0),
    "sat_fat": cat_sample_df["sat_fat"].fillna(0),
    "sugars": cat_sample_df["sugars"].fillna(0),
    "salt": cat_sample_df["salt_g"].fillna(cat_sample_df["sodium_g"] * 2.5).fillna(0),
    "fibre": cat_sample_df["fiber"].fillna(cat_sample_df["fibers"]).fillna(0),
    "proteins": cat_sample_df["proteins"].fillna(0),
    "fvl_pct": cat_sample_df["fvn_pct"].fillna(cat_sample_df["fvl_pct"]).fillna(0).clip(0, 100)
}
X_eval = pd.DataFrame(feat_map)[list(FEATURES)]
y_str = cat_sample_df["grade"].astype(str).str.upper()
valid_m = y_str.isin(["A", "B", "C", "D", "E"])
X_eval = X_eval[valid_m]
y_str = y_str[valid_m]
y_eval = y_str.map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4})

preds_eval = clf.predict(X_eval)
acc_eval = accuracy_score(y_eval, preds_eval)
f1_eval = f1_score(y_eval, preds_eval, average="macro")

ev_c1, ev_c2 = st.columns(2)
with ev_c1:
    st.metric("Accuracy mesurée", f"{acc_eval*100:.2f} %")
with ev_c2:
    st.metric("F1 macro mesuré", f"{f1_eval:.4f}")

if selected_cat == "Fromages":
    st.warning(
        "L'arbre sous-évalue les fromages car il applique le cutoff standard N ≥ 11 (qui annule les points de protéines). "
        "Or, le décret officiel Nutri-Score accorde une dérogation au fromage : les protéines sont déduites sans condition "
        "afin de valoriser l'apport calcique des produits laitiers.",
        title="Démonstration de la dérogation réglementaire"
    )
else:
    st.success(
        "Les biscuits relèvent du même algorithme général que les céréales de petit-déjeuner. "
        "L'arbre de décision CART conserve donc une bonne précision de découpage.",
        title="Validité du modèle sur les aliments généraux"
    )

# Confusion matrix for the chosen category
cm_cat = confusion_matrix(y_eval, preds_eval, labels=[0, 1, 2, 3, 4])
fig, ax = plt.subplots(figsize=(6, 3.8))
sns.heatmap(cm_cat, annot=True, fmt="d", cmap="Blues", xticklabels=["A","B","C","D","E"], yticklabels=["A","B","C","D","E"], ax=ax)
ax.set_title(f"Matrice de confusion CART sur les {selected_cat.lower()}")
ax.set_xlabel("Note prédite par l'arbre")
ax.set_ylabel("Note officielle OFF")
plt.tight_layout()

st.pyplot(fig, alt=f"Matrice de confusion sur {selected_cat.lower()}")
