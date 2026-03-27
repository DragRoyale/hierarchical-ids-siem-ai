import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from scipy.stats import ttest_rel

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

def build_preprocessor(X):

    cat_cols = []
    if "protocol" in X.columns and not pd.api.types.is_numeric_dtype(X["protocol"]):
        cat_cols.append("protocol")

    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

def main():

    df = pd.read_csv(DATA_FILE)
    df = add_features(df)

    df["attack_type"] = df["attack_type"].astype(str).str.strip()
    df["attack_group"] = df["attack_type"].map(GROUP_MAP).fillna("other_attack")

    y = df["attack_group"].astype(str)
    y_bin = (y != "benign").astype(int)

    drop_cols = ["attack_type", "attack_group", "url", "user_agent", "timestamp", "src_ip", "dst_ip"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for c in X.columns:
        if c != "protocol":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    pre = build_preprocessor(X)

    rskf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE
    )

    single_scores = []
    two_stage_scores = []

    for fold, (tr, te) in enumerate(rskf.split(X, y_bin), 1):

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        ybin_tr, ybin_te = y_bin.iloc[tr], y_bin.iloc[te]

        # ---------- Single-stage ----------
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        y_te_enc = le.transform(y_te)

        single_model = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                objective="multi:softmax",
                num_class=len(le.classes_),
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])

        single_model.fit(X_tr, y_tr_enc)
        pred_single = single_model.predict(X_te)

        macro_single = f1_score(y_te_enc, pred_single, average="macro", zero_division=0)
        single_scores.append(macro_single)

        # ---------- Two-stage ----------

        # Stage 1
        stage1 = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                objective="binary:logistic",
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])
        stage1.fit(X_tr, ybin_tr)
        attack_pred = stage1.predict(X_te)

        # Stage 2
        tr_attack_mask = (ybin_tr == 1).to_numpy()
        X_tr_attack = X_tr.iloc[tr_attack_mask]
        y_tr_attack = y_tr.iloc[tr_attack_mask]

        le_attack = LabelEncoder()
        y_tr_attack_enc = le_attack.fit_transform(y_tr_attack)

        stage2 = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                objective="multi:softmax",
                num_class=len(le_attack.classes_),
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
        stage2.fit(X_tr_attack, y_tr_attack_enc)

        final_pred = np.full(len(X_te), "benign", dtype=object)
        mask_te = (attack_pred == 1)

        if mask_te.sum() > 0:
            pred_enc = stage2.predict(X_te.iloc[mask_te])
            pred_str = le_attack.inverse_transform(pred_enc)
            final_pred[mask_te] = pred_str

        macro_two = f1_score(y_te, final_pred, average="macro", zero_division=0)
        two_stage_scores.append(macro_two)

        print(f"Run {fold}: single={macro_single:.4f} | two-stage={macro_two:.4f}")

    # -------- Statistics --------
    single_scores = np.array(single_scores)
    two_stage_scores = np.array(two_stage_scores)

    diff = two_stage_scores - single_scores
    t_stat, p_val = ttest_rel(two_stage_scores, single_scores)

    cohens_d = diff.mean() / diff.std()

    print("\n===== FINAL RESULTS (25 runs) =====")
    print("Single-stage mean:", single_scores.mean())
    print("Two-stage mean:", two_stage_scores.mean())
    print("Mean improvement:", diff.mean())
    print("t-statistic:", t_stat)
    print("p-value:", p_val)
    print("Cohen's d:", cohens_d)

    np.save("results/single_stage_25.npy", single_scores)
    np.save("results/two_stage_25.npy", two_stage_scores)

if __name__ == "__main__":
    main()