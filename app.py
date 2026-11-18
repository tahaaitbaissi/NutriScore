import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

from src.nutriscore import compute, grade as calc_grade, FEATURES

st.set_page_config(
    page_title="Nutri-Score — Rétro-ingénierie & Arbre de Décision",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colors for grades
GRADE_COLORS = {
    "A": "#038141",
    "B": "#85BB2F",
    "C": "#FECB02",
    "D": "#EE8100",
    "E": "#E63E11"
}

@st.cache_data
def load_datasets():
    # Load processed data
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    clean = pd.read_parquet("data/processed/clean.parquet")
    disagreements = pd.read_parquet("data/processed/calculator_disagreements.parquet")
    return train, val, test, clean, disagreements

@st.cache_resource
def load_trained_model(train_df):
    # Fit pruned decision tree
    clf = DecisionTreeClassifier(ccp_alpha=0.000129, random_state=42)
    features = list(FEATURES)
    clf.fit(train_df[features], train_df["y_int"])
    return clf

train_df, val_df, test_df, clean_df, dis_df = load_datasets()
clf = load_trained_model(train_df)

int_to_grade = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

# Sidebar
st.sidebar.title("🥗 Nutri-Score Mining")
st.sidebar.markdown("**Projet** : Retrouver les règles du Nutri-Score  \n**Auteurs** : Taha Ait Baissi & Yassine Amjad  \n**Enseignante** : Pr. Hiba El Balbali (EMSI)")

tab_sim, tab_exp, tab_rules, tab_multi = st.tabs([
    "🎛️ Simulateur & Comparateur",
    "🔍 Explorateur de Produits & Exceptions",
    "📊 Rétro-ingénierie & Seuils",
    "🧀 Test Multi-Catégories"
])

# Tab 1: Simulator
with tab_sim:
    st.subheader("Simulateur interactif : Calculateur Officiel vs Arbre de Décision")
    st.markdown("Ajustez les 7 nutriments pour observer en temps réel la divergence ou la convergence entre la règle officielle 2023 et l'arbre CART appris.")

    # Presets
    preset = st.selectbox(
        "Profil type prédéfini :",
        [
            "Personnalisé",
            "Flocons d'avoine complets (Note attendue A)",
            "Muesli fruits secs (Note attendue B)",
            "Granola noisettes & chocolat (Note attendue C)",
            "Pétales de maïs givrés au sucre (Note attendue D)",
            "Céréales fourrées ultra-sucrées (Note attendue E)",
            "Cas d'exception : Trink Porridge (Désaccord OFF)"
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

    if preset == "Flocons d'avoine complets (Note attendue A)":
        defaults = {"energy_kj": 1500.0, "sat_fat": 1.3, "sugars": 1.0, "salt": 0.02, "fibre": 10.0, "proteins": 12.0, "fvl_pct": 0.0}
    elif preset == "Muesli fruits secs (Note attendue B)":
        defaults = {"energy_kj": 1550.0, "sat_fat": 2.0, "sugars": 12.0, "salt": 0.15, "fibre": 7.5, "proteins": 9.0, "fvl_pct": 25.0}
    elif preset == "Granola noisettes & chocolat (Note attendue C)":
        defaults = {"energy_kj": 1850.0, "sat_fat": 4.5, "sugars": 18.0, "salt": 0.40, "fibre": 5.0, "proteins": 8.5, "fvl_pct": 0.0}
    elif preset == "Pétales de maïs givrés au sucre (Note attendue D)":
        defaults = {"energy_kj": 1650.0, "sat_fat": 0.6, "sugars": 37.0, "salt": 0.85, "fibre": 2.5, "proteins": 5.0, "fvl_pct": 0.0}
    elif preset == "Céréales fourrées ultra-sucrées (Note attendue E)":
        defaults = {"energy_kj": 1950.0, "sat_fat": 7.5, "sugars": 42.0, "salt": 0.95, "fibre": 2.0, "proteins": 4.5, "fvl_pct": 0.0}
    elif preset == "Cas d'exception : Trink Porridge (Désaccord OFF)":
        defaults = {"energy_kj": 1515.0, "sat_fat": 1.1, "sugars": 9.3, "salt": 0.02, "fibre": 11.0, "proteins": 11.0, "fvl_pct": 0.0}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Nutriments défavorables (Composante N)")
        val_energy = st.slider("Énergie (kJ / 100g)", 0.0, 3500.0, float(defaults["energy_kj"]), step=10.0)
        val_sat_fat = st.slider("Acides gras saturés (g / 100g)", 0.0, 30.0, float(defaults["sat_fat"]), step=0.1)
        val_sugars = st.slider("Sucres simples (g / 100g)", 0.0, 70.0, float(defaults["sugars"]), step=0.5)
        val_salt = st.slider("Sel (g / 100g)", 0.0, 5.0, float(defaults["salt"]), step=0.01)

    with col2:
        st.markdown("#### Nutriments favorables (Composante P)")
        val_fibre = st.slider("Fibres alimentaires (g / 100g)", 0.0, 30.0, float(defaults["fibre"]), step=0.2)
        val_proteins = st.slider("Protéines (g / 100g)", 0.0, 30.0, float(defaults["proteins"]), step=0.2)
        val_fvl = st.slider("Fruits, Légumes, Légumineuses (%)", 0.0, 100.0, float(defaults["fvl_pct"]), step=1.0)

    # Compute official score
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

    # Predict with tree
    input_df = pd.DataFrame([nutrients_dict])[list(FEATURES)]
    tree_pred_int = clf.predict(input_df)[0]
    tree_pred_grade = int_to_grade[tree_pred_int]
    tree_proba = clf.predict_proba(input_df)[0]

    st.markdown("---")
    res_c1, res_c2, res_c3 = st.columns([1, 1, 1])

    with res_c1:
        st.markdown("### 🏛️ Calculateur Officiel 2023")
        color_off = GRADE_COLORS[official_grade]
        st.markdown(f"<div style='background-color:{color_off}; color:white; font-size:36px; font-weight:bold; text-align:center; border-radius:10px; padding:10px;'>Note : {official_grade}</div>", unsafe_allow_html=True)
        st.write(f"**Score numérique** : `{official_score}` (N: {calc_res['points_N']} | P: {calc_res['points_P']})")
        st.caption(f"Points : Énergie ({calc_res['points']['energy_kj']}), SatFat ({calc_res['points']['sat_fat']}), Sucres ({calc_res['points']['sugars']}), Sel ({calc_res['points']['salt']}) - Fibres ({calc_res['points']['fibre']}), Protéines ({calc_res['points']['proteins']}), FVL ({calc_res['points']['fvl_pct']})")

    with res_c2:
        st.markdown("### 🌲 Arbre de Décision CART")
        color_tree = GRADE_COLORS[tree_pred_grade]
        st.markdown(f"<div style='background-color:{color_tree}; color:white; font-size:36px; font-weight:bold; text-align:center; border-radius:10px; padding:10px;'>Prédit : {tree_pred_grade}</div>", unsafe_allow_html=True)
        proba_str = " | ".join([f"{g}: {p*100:.0f}%" for g, p in zip(["A", "B", "C", "D", "E"], tree_proba)])
        st.write(f"**Distribution feuilles** : `{proba_str}`")

    with res_c3:
        st.markdown("### ⚖️ Diagnostic d'Accord")
        diff_abs = abs(ord(official_grade) - ord(tree_pred_grade))
        if diff_abs == 0:
            st.success("✅ **Accord Parfait** : L'arbre retrouve exactement la note officielle !")
        elif diff_abs == 1:
            st.warning(f"⚠️ **Erreur de Frontière (1 lettre)** : Officiel = {official_grade}, Arbre = {tree_pred_grade}. Fréquent aux limites de seuils.")
        else:
            st.error(f"❌ **Divergence Forte ({diff_abs} lettres)** : Incompatibilité entre l'hyperplan officiel et la coupe orthogonale.")

# Tab 2: Explorer
with tab_exp:
    st.subheader("Explorateur de Produits & Audit des Exceptions")
    st.markdown("Filtrez les **14 611 céréales propres** pour inspecter les anomalies de labels, les fiches suspectes et les désaccords.")

    # Filter controls
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        filter_grade = st.multiselect("Note officielle OFF :", ["A", "B", "C", "D", "E"], default=["A", "B", "C", "D", "E"])
    with f_c2:
        filter_tree_status = st.selectbox("Statut de l'arbre :", ["Tous", "Erreurs de l'arbre uniquement", "Bien classés"])
    with f_c3:
        filter_suspect = st.selectbox("Incohérence physique (R7) :", ["Tous", "suspect_consistency = True (417 produits)"])
    with f_c4:
        filter_calc = st.selectbox("Désaccord Calculateur/OFF :", ["Tous", "Désaccords uniquement (1 548 produits)"])

    # Compute predictions on clean
    clean_view = clean_df.copy()
    clean_view["pred_int"] = clf.predict(clean_view[list(FEATURES)])
    clean_view["pred_grade"] = clean_view["pred_int"].map(int_to_grade)
    clean_view["tree_error"] = clean_view["pred_grade"] != clean_view["grade"].astype(str)
    clean_view["calc_disagreement"] = clean_view["code"].isin(set(dis_df["code"]))

    # Apply filters
    filtered = clean_view[clean_view["grade"].astype(str).isin(filter_grade)]
    if filter_tree_status == "Erreurs de l'arbre uniquement":
        filtered = filtered[filtered["tree_error"]]
    elif filter_tree_status == "Bien classés":
        filtered = filtered[~filtered["tree_error"]]

    if filter_suspect == "suspect_consistency = True (417 produits)":
        filtered = filtered[filtered["suspect_consistency"]]

    if filter_calc == "Désaccords uniquement (1 548 produits)":
        filtered = filtered[filtered["calc_disagreement"]]

    st.write(f"**{len(filtered)} produits** correspondent aux critères de recherche.")

    cols_show = ["code", "product_name", "grade", "pred_grade", "off_score", "score_v2023", "sugars", "sat_fat", "fibre", "salt", "suspect_consistency", "countries"]
    st.dataframe(
        filtered[cols_show].head(200),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### 🔬 Zoom sur un produit")
    selected_code = st.selectbox("Sélectionnez un code-barres pour voir le détail :", filtered["code"].head(50))
    if selected_code:
        prod = clean_view[clean_view["code"] == selected_code].iloc[0]
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown(f"**Produit** : {prod['product_name']} (`{prod['code']}`)")
            st.write(f"- **Note OFF** : `{prod['grade']}` (Score OFF: `{prod['off_score']}`)")
            st.write(f"- **Prédit par l'arbre** : `{prod['pred_grade']}`")
            st.write(f"- **Calcul officiel 2023** : `{prod['grade_v2023']}` (Score: `{prod['score_v2023']}`)")
            st.write(f"- **Millésime estimé** : `{prod['nutriscore_version_guess']}`")
            st.write(f"- **Incohérence R7** : `{prod['suspect_consistency']}`")
        with p_c2:
            st.markdown("**Composition nutritionnelle (pour 100g)** :")
            st.write(f"- Énergie : `{prod['energy_kj']} kJ`")
            st.write(f"- Gras saturés : `{prod['sat_fat']} g`")
            st.write(f"- Sucres : `{prod['sugars']} g`")
            st.write(f"- Sel : `{prod['salt']} g`")
            st.write(f"- Fibres : `{prod['fibre']} g`")
            st.write(f"- Protéines : `{prod['proteins']} g`")

# Tab 3: Rules & Thresholds
with tab_rules:
    st.subheader("Rétro-ingénierie : Seuils découverts par l'arbre vs Décret officiel 2023")
    st.markdown("Démonstration mathématique : l'arbre CART retrouve avec une précision remarquable les seuils des paliers officiels de l'Arrêté du 14 mars 2025.")

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
                "Seuil trouvé par l'arbre": round(float(split_val), 2),
                "Échantillons concernés": int(n_samp),
                "Seuil officiel le plus proche": float(closest),
                "Écart absolu": round(float(diff), 3)
            })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Confusion matrix on test set
    st.markdown("#### Matrice de Confusion sur le jeu de test (2 192 produits)")
    y_test_pred = clf.predict(test_df[list(FEATURES)])
    cm = confusion_matrix(test_df["y_int"], y_test_pred)
    cm_norm = confusion_matrix(test_df["y_int"], y_test_pred, normalize="true")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["A","B","C","D","E"], yticklabels=["A","B","C","D","E"], ax=ax1)
    ax1.set_title("Matrice brute")
    ax1.set_xlabel("Prédit")
    ax1.set_ylabel("Réel")

    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=["A","B","C","D","E"], yticklabels=["A","B","C","D","E"], ax=ax2)
    ax2.set_title("Rappel par classe")
    ax2.set_xlabel("Prédit")
    ax2.set_ylabel("Réel")
    plt.tight_layout()
    st.pyplot(fig)

# Tab 4: Multi-Category
with tab_multi:
    st.subheader("Extension hors-domaine : Test sur d'autres catégories")
    st.markdown("Évaluez le comportement de l'arbre (entraîné sur les céréales) lorsqu'on lui soumet des biscuits ou des fromages.")

    cat_choice = st.radio("Sélectionnez la catégorie à tester :", ["Biscuits (Cas général, 58 956 produits)", "Fromages (Dérogation protéines, 87 080 produits)"])

    if st.button("Lancer l'évaluation sur 1 000 échantillons aléatoires"):
        with st.spinner("Chargement et calcul via DuckDB..."):
            file_path = "data/raw/category_biscuits.parquet" if "Biscuits" in cat_choice else "data/raw/category_cheeses.parquet"
            sample_df = pd.read_parquet(file_path).dropna(subset=["grade"]).sample(n=1000, random_state=42)

            # Map raw fields to features
            feat_map = {
                "energy_kj": sample_df["energy_kj_raw"].fillna(sample_df["energy_kj_field"]).fillna(0),
                "sat_fat": sample_df["sat_fat"].fillna(0),
                "sugars": sample_df["sugars"].fillna(0),
                "salt": sample_df["salt_g"].fillna(sample_df["sodium_g"] * 2.5).fillna(0),
                "fibre": sample_df["fiber"].fillna(sample_df["fibers"]).fillna(0),
                "proteins": sample_df["proteins"].fillna(0),
                "fvl_pct": sample_df["fvn_pct"].fillna(sample_df["fvl_pct"]).fillna(0).clip(0, 100)
            }
            X_sample = pd.DataFrame(feat_map)[list(FEATURES)]
            sample_grade = sample_df["grade"].astype(str).str.upper()
            valid_mask = sample_grade.isin(["A", "B", "C", "D", "E"])
            X_sample = X_sample[valid_mask]
            sample_grade = sample_grade[valid_mask]
            sample_y = sample_grade.map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4})

            preds = clf.predict(X_sample)
            acc = accuracy_score(sample_y, preds)
            f1 = f1_score(sample_y, preds, average="macro")

            st.write(f"**Accuracy sur la catégorie** : `{acc*100:.2f} %`")
            st.write(f"**F1 Macro** : `{f1:.4f}`")

            if "Fromages" in cat_choice:
                st.info("💡 **Constat Fromages** : L'arbre a tendance à sous-estimer la note de certains fromages. En effet, dans le calcul officiel des fromages, les protéines sont déduites même si N ≥ 11 (règle calcium). L'arbre n'ayant appris que sur les céréales solides applique le cutoff général !")
            else:
                st.success("💡 **Constat Biscuits** : Les biscuits relèvent du même algorithme général que les céréales. L'arbre conserve une bonne capacité de généralisation !")
