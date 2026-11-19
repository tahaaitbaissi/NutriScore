import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.app_utils import ROOT, load_datasets, load_trained_model
from src.nutriscore import FEATURES

clf = load_trained_model()
train_df, val_df, test_df, clean_df, dis_df = load_datasets()

GRADES = ["A", "B", "C", "D", "E"]
CATEGORIES = {
    "Céréales de petit-déjeuner": {
        "file": "category_breakfast_cereals.parquet",
        "regime": "Cas général standard",
        "emoji": "🥣",
    },
    "Biscuits": {
        "file": "category_biscuits.parquet",
        "regime": "Cas général standard",
        "emoji": "🍪",
    },
    "Fromages": {
        "file": "category_cheeses.parquet",
        "regime": "Dérogation fromage (protéines toujours comptées)",
        "emoji": "🧀",
    },
}


def features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Mêmes règles que la Partie 1, sans imputation : les cas incomplets seront écartés."""
    return pd.DataFrame({
        "energy_kj": df["energy_kj_raw"].fillna(df["energy_kj_field"]).fillna(df["energy_kcal_field"] * 4.184),
        "sat_fat": df["sat_fat"],
        "sugars": df["sugars"],
        "salt": df["salt_g"].fillna(df["sodium_g"] * 2.5),
        "fibre": df["fiber"].fillna(df["fibers"]),
        "proteins": df["proteins"],
        "fvl_pct": df["fvn_pct"].fillna(df["fvl_pct"]).clip(0, 100),
    })[list(FEATURES)]


@st.cache_data(ttl="1h")
def load_category_summaries():
    dist_rows, nutr_rows, kpi, samples = [], [], {}, {}

    for name, meta in CATEGORIES.items():
        df = pd.read_parquet(ROOT / "data" / "raw" / meta["file"])
        grade = df["grade"].astype(str).str.upper()
        valid = grade.isin(GRADES)
        counts = grade[valid].value_counts(normalize=True).reindex(GRADES, fill_value=0) * 100

        for g in GRADES:
            dist_rows.append({"Catégorie": name, "Note": g, "Pourcentage": round(float(counts[g]), 2)})

        salt = df["salt_g"].fillna(df["sodium_g"] * 2.5)
        fibre = df["fiber"].fillna(df["fibers"])
        nutr_rows.append({
            "Catégorie": name,
            "Sucres (g)": round(float(df["sugars"].median()), 1),
            "Gras saturés (g)": round(float(df["sat_fat"].median()), 1),
            "Sel (g)": round(float(salt.median()), 2),
            "Protéines (g)": round(float(df["proteins"].median()), 1),
            "Fibres (g)": round(float(fibre.median()), 1),
        })

        kpi[name] = {
            "n_raw": len(df),
            "n_valid": int(valid.sum()),
            "dominant": counts.idxmax(),
            "dominant_pct": round(float(counts.max()), 1),
            "a_b_pct": round(float(counts[["A", "B"]].sum()), 1),
            "regime": meta["regime"],
            "emoji": meta["emoji"],
        }

        X = features_from_raw(df)
        y = grade.map({g: i for i, g in enumerate(GRADES)})
        complete = X.notna().all(axis=1) & valid
        n_complete = int(complete.sum())
        n_sample = min(1200, n_complete)
        idx = X[complete].sample(n=n_sample, random_state=42).index
        samples[name] = {
            "X": X.loc[idx].reset_index(drop=True),
            "y": y.loc[idx].reset_index(drop=True),
            "n_complete": n_complete,
            "n_sample": n_sample,
            "regime": meta["regime"],
        }

    return pd.DataFrame(dist_rows), pd.DataFrame(nutr_rows), kpi, samples


dist_df, nutr_df, kpi, samples = load_category_summaries()

st.header("Comparaison inter-catégories & évaluation hors-domaine")
st.caption(
    "Le modèle est entraîné exclusivement sur les céréales. On mesure ici son transfert sur les biscuits "
    "(même algorithme général) et sur les fromages (dérogation protéines), en n'utilisant que des cas "
    "complets, sans imputation."
)

st.subheader("1. Vue d'ensemble des 3 catégories")
for col, (name, meta) in zip(st.columns(3), CATEGORIES.items()):
    k = kpi[name]
    with col:
        with st.container(border=True):
            st.write(f"**{meta['emoji']} {name}**")
            st.caption(f"{k['n_raw']:,} produits bruts | {k['n_valid']:,} notes valides")
            st.metric(
                "Note dominante",
                f"{k['dominant']} ({k['dominant_pct']} %)",
                delta=f"A+B = {k['a_b_pct']} %",
                delta_color="off",
            )
            st.caption(k["regime"])
            if name == "Céréales de petit-déjeuner":
                st.caption("Entraînement : 10 227 lignes (14 611 nettoyées).")

st.subheader("2. Distribution des notes Nutri-Score (A à E)")
st.bar_chart(
    dist_df, x="Note", y="Pourcentage", color="Catégorie", height=320,
    alt="Comparaison de la part des notes A à E entre céréales, biscuits et fromages",
)

st.subheader("3. Médianes nutritionnelles comparées (pour 100 g)")
st.dataframe(nutr_df, hide_index=True)

st.subheader("4. Performance de l'arbre CART sur chaque catégorie")
st.markdown(
    "Protocole : **cas complets** sur les 7 entrées (mêmes règles que la Partie 1, aucune imputation) ; "
    "1 200 produits tirés au sort (graine 42) pour les catégories hors-domaine. Les céréales sont évaluées "
    "sur le **jeu de test officiel** (2 192 produits), pour rester cohérent avec le notebook 02."
)

bench_rows = []
for name, s in samples.items():
    if name == "Céréales de petit-déjeuner":
        X_eval = test_df[list(FEATURES)]
        y_eval = test_df["y_int"]
        protocol = f"Test officiel ({len(X_eval):,})"
    else:
        X_eval = s["X"]
        y_eval = s["y"]
        protocol = f"1 200 cas complets (sur {s['n_complete']:,})"

    pred = clf.predict(X_eval)
    acc = accuracy_score(y_eval, pred)
    f1m = f1_score(y_eval, pred, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)
    e_share = 100 * (pred == 4).mean()

    if name == "Céréales de petit-déjeuner":
        bias = "In-domain : référence du notebook."
    elif name == "Biscuits":
        bias = f"Transfert correct ; {e_share:.0f} % prédits E (classe dominante)."
    else:
        bias = f"Échec : {e_share:.0f} % prédits E ; la dérogation protéines n'est pas apprise."

    bench_rows.append({
        "Catégorie": name,
        "Régime réglementaire": s["regime"],
        "Protocole": protocol,
        "Accuracy": f"{100 * acc:.2f} %",
        "F1 macro": f"{f1m:.4f}",
        "Biais observé": bias,
    })

st.dataframe(pd.DataFrame(bench_rows), hide_index=True)

st.subheader("5. Audit interactif d'une catégorie")
selected_cat = st.segmented_control(
    "Choisissez la catégorie à évaluer en détail :",
    ["Biscuits", "Fromages"],
    default="Biscuits",
)

s = samples[selected_cat]
X_eval, y_eval = s["X"], s["y"]
preds_eval = clf.predict(X_eval)
acc_eval = accuracy_score(y_eval, preds_eval)
f1_eval = f1_score(y_eval, preds_eval, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)

st.caption(
    f"Échantillon : {s['n_sample']:,} cas complets sur {s['n_complete']:,} disponibles "
    f"(graine 42, aucune imputation)."
)
c1, c2 = st.columns(2)
with c1:
    st.metric("Accuracy mesurée", f"{100 * acc_eval:.2f} %")
with c2:
    st.metric("F1 macro mesuré", f"{f1_eval:.4f}")

if selected_cat == "Fromages":
    st.warning(
        "L'arbre prédit souvent E pour les fromages : il applique la règle générale N ≥ 11, qui annule "
        "les protéines, alors que le barème fromage les compte toujours. C'est exactement le cas "
        "dérogatoire que l'arbre n'a pas appris (il n'a vu que des céréales).",
        title="Démonstration de la dérogation réglementaire",
    )
else:
    st.success(
        f"Les biscuits relèvent du même algorithme général que les céréales : le transfert reste correct "
        f"({100 * acc_eval:.1f} % d'accuracy), même si le F1 macro est plus faible car A et B sont rares.",
        title="Validité du modèle sur les aliments généraux",
    )

cm_cat = confusion_matrix(y_eval, preds_eval, labels=[0, 1, 2, 3, 4])
fig, ax = plt.subplots(figsize=(6, 3.8))
sns.heatmap(
    cm_cat, annot=True, fmt="d", cmap="Blues",
    xticklabels=GRADES, yticklabels=GRADES, ax=ax,
)
ax.set_title(f"Matrice de confusion CART sur les {selected_cat.lower()}")
ax.set_xlabel("Note prédite par l'arbre")
ax.set_ylabel("Note officielle OFF")
plt.tight_layout()
st.pyplot(fig, alt=f"Matrice de confusion sur {selected_cat.lower()}")
