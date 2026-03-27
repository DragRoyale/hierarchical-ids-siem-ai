import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_FILE = "data/raw/cybersecurity.csv"

GROUP_MAP = {
    "benign": "benign",
    "port-scan": "scanning",
    "sql-injection": "injection",
    "command-injection": "injection",
    "xss": "injection",
    "brute-force": "credential_attack",
    "credential-stuffing": "credential_attack",
    "c2": "malware_c2",
    "ddos": "ddos",
    "exploit-attempt": "malware_c2",
}

def add_features(df):
    df = df.copy()

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["hour"] = ts.dt.hour.fillna(0)
        df["dayofweek"] = ts.dt.dayofweek.fillna(0)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    if "url" in df.columns:
        url = df["url"].astype(str)
        df["url_len"] = url.str.len()
        df["url_num_params"] = url.str.count("&") + url.str.contains(r"\?").astype(int)
        df["url_has_ip"] = url.str.contains(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", regex=True).astype(int)
        df["url_has_sql_chars"] = url.str.contains(r"(?:%27|'|--|%23|#)", regex=True).astype(int)
        df["url_has_cmd_chars"] = url.str.contains(r"(?:;|\|\||&&|\$\(|`)", regex=True).astype(int)
        df["url_has_xss"] = url.str.contains(r"(?:<script|%3cscript|onerror=|onload=)", regex=True, case=False).astype(int)

    if "user_agent" in df.columns:
        ua = df["user_agent"].astype(str).str.lower()
        df["ua_len"] = ua.str.len()
        df["ua_is_curl"] = ua.str.contains("curl").astype(int)
        df["ua_is_wget"] = ua.str.contains("wget").astype(int)
        df["ua_is_python"] = ua.str.contains("python").astype(int)
        df["ua_is_browser"] = ua.str.contains("mozilla|chrome|safari|firefox|edge").astype(int)

    return df

def main():

    df = pd.read_csv(DATA_FILE)
    df = add_features(df)

    df["attack_type"] = df["attack_type"].astype(str).str.strip()
    df["attack_group"] = df["attack_type"].map(GROUP_MAP).fillna("other_attack")

    y = df["attack_group"].astype(str)

    drop_cols = ["attack_type", "attack_group", "url", "user_agent", "timestamp", "src_ip", "dst_ip"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for c in X.columns:
        if c != "protocol":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    cat_cols = []
    if "protocol" in X.columns and not pd.api.types.is_numeric_dtype(X["protocol"]):
        cat_cols.append("protocol")

    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores = []

    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_enc[tr], y_enc[te]

        model = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                objective="multi:softmax",
                num_class=len(le.classes_),
                n_estimators=700,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        macro = f1_score(y_te, pred, average="macro", zero_division=0)
        scores.append(float(macro))

        print(f"Fold {fold}: macro-F1={macro:.4f}")

    print(f"\nSingle-stage SIEM-like 5-fold macro-F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

if __name__ == "__main__":
    main()