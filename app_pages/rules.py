import streamlit as st
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from src.nutriscore import FEATURES
from src.app_utils import load_datasets, load_trained_model

train_df, val_df, test_df, clean_df, dis_df = load_datasets()
clf = load_trained_model()

st.header("Rétro-ingénierie des règles et seuils officiels")
st.caption("Confrontation entre les découpages appris par l'arbre de décision CART et les tables du décret officiel 2023.")

# Feature importances
st.subheader("Importance relative des 7 nutriments")
fi_series = pd.Series(clf.feature_importances_, index=list(FEATURES)).sort_values(ascending=False)
fi_df = pd.DataFrame({
    "Nutriment": fi_series.index,
    "Importance": (fi_series.values * 100).round(2)
})
st.bar_chart(fi_df, x="Nutriment", y="Importance", height=240, alt="Importance de Gini des 7 nutriments")

# Threshold comparison
st.subheader("Coupures prioritaires de l'arbre vs seuils réglementaires")

with open("config/thresholds.yaml", encoding="utf-8") as f:
    cfg_thresh = yaml.safe_load(f)["versions"]["2023"]

official_tables = {
    "energy_kj": cfg_thresh["energy_kj"],
    "sat_fat": cfg_thresh["saturated_fat_g"],
    "sugars": cfg_thresh["sugars_g"],
    "salt": cfg_thresh["salt_g"],
    "fibre": cfg_thresh["fibre_g"],
    "proteins": cfg_thresh["protein_g"]
}

comp_rows = []
for f_idx, f_name in enumerate(FEATURES):
    if f_name not in official_tables:
        continue
    mask = clf.tree_.feature == f_idx
    splits = clf.tree_.threshold[mask]
    samples = clf.tree_.n_node_samples[mask]
    if len(splits) == 0:
        continue
    top_idx = np.argsort(samples)[::-1][:4]
    for idx in top_idx:
        split_val = splits[idx]
        n_samp = samples[idx]
        closest = min(official_tables[f_name], key=lambda x: abs(x - split_val))
        diff = abs(closest - split_val)
        comp_rows.append({
            "Nutriment": f_name,
            "Coupure arbre": round(float(split_val), 2),
            "Échantillons concernés": int(n_samp),
            "Seuil officiel le plus proche": float(closest),
            "Écart absolu": round(float(diff), 3)
        })

comp_df = pd.DataFrame(comp_rows)
st.dataframe(comp_df, hide_index=True)

# Confusion matrices on test
st.subheader("Matrice de confusion sur le jeu de test (2 192 produits)")
y_test_pred = clf.predict(test_df[list(FEATURES)])
cm = confusion_matrix(test_df["y_int"], y_test_pred)
cm_norm = confusion_matrix(test_df["y_int"], y_test_pred, normalize="true")

st.caption(
    f"Test ({len(test_df):,} produits) : accuracy "
    f"{100 * accuracy_score(test_df['y_int'], y_test_pred):.2f} % | "
    f"F1 macro {f1_score(test_df['y_int'], y_test_pred, average='macro'):.4f}"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["A","B","C","D","E"], yticklabels=["A","B","C","D","E"], ax=ax1)
ax1.set_title("Matrice brute")
ax1.set_xlabel("Note prédite par l'arbre")
ax1.set_ylabel("Note officielle OFF")

sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=["A","B","C","D","E"], yticklabels=["A","B","C","D","E"], ax=ax2)
ax2.set_title("Rappel par classe (normalisé)")
ax2.set_xlabel("Note prédite par l'arbre")
ax2.set_ylabel("Note officielle OFF")
plt.tight_layout()

st.pyplot(fig, alt="Matrices de confusion brute et normalisée sur le jeu de test")

