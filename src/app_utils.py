import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.nutriscore import FEATURES

@st.cache_data(ttl="1h")
def load_datasets():
    # Load processed datasets
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    clean = pd.read_parquet("data/processed/clean.parquet")
    disagreements = pd.read_parquet("data/processed/calculator_disagreements.parquet")
    return train, val, test, clean, disagreements

@st.cache_resource
def load_trained_model():
    # Train optimal decision tree
    train = pd.read_parquet("data/processed/train.parquet")
    clf = DecisionTreeClassifier(ccp_alpha=0.000129, random_state=42)
    clf.fit(train[list(FEATURES)], train["y_int"])
    return clf

INT_TO_GRADE = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
