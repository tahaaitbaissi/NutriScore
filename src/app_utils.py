from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from sklearn.tree import DecisionTreeClassifier

from src.nutriscore import FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PARAMS = ROOT / "config" / "model_params.yaml"


@st.cache_data(ttl="1h")
def load_datasets():
    """Charge les jeux de la Partie 1 (chemins absolus, indépendants du cwd)."""
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    clean = pd.read_parquet(DATA_DIR / "clean.parquet")
    disagreements = pd.read_parquet(DATA_DIR / "calculator_disagreements.parquet")
    return train, val, test, clean, disagreements


@st.cache_data(ttl="1h")
def load_model_params():
    """Paramètres exacts exportés par le notebook 02 (pas d'alpha arrondi)."""
    with open(MODEL_PARAMS, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource
def load_trained_model():
    """Réentraîne l'arbre retenu par le notebook, avec le ccp_alpha exact."""
    params = load_model_params()
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    clf = DecisionTreeClassifier(
        ccp_alpha=float(params["ccp_alpha"]),
        random_state=int(params["random_state"]),
    )
    clf.fit(train[list(FEATURES)], train["y_int"])
    return clf


def add_split_labels(clean: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame,
                     test: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne `split` (train/val/test) à partir des codes de la Partie 1."""
    mapping = {}
    for split, frame in (("train", train), ("val", val), ("test", test)):
        mapping.update({code: split for code in frame["code"]})
    return clean.assign(split=clean["code"].map(mapping))


INT_TO_GRADE = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
