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

# --- label grouping: adapt if your attack_type names differ ---
GROUP_MAP = {
    "benign": "benign",

    # scanning
    "port-scan": "scanning",

    # injections
    "sql-injection": "injection",
    "command-injection": "injection",
    "xss": "injection",

    # credential attacks
    "brute-force": "credential_attack",
    "credential-stuffing": "credential_attack",

    # malware / c2
    "c2": "malware_c2",

    # ddos
    "ddos": "ddos",

    # (optional) if present in your data:
    "exploit-attempt": "malware_c2",
}

def is_private_ip(ip: str) -> bool:
    # quick check for RFC1918
    if not isinstance(ip, str):
        return False
    ip = ip.strip()
    return (
        ip.startswith("10.")
        or ip.startswith("192.168.")
        or re.match(r"^172\.(1[6-9]|2[0-9]|3[0-1])\.", ip) is not None
    )

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- time features ---
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["hour"] = ts.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- url features ---
    if "url" in df.columns:
        url = df["url"].astype(str).fillna("")
        df["url_len"] = url.str.len()
        df["url_num_params"] = url.str.count(r"&") + url.str.contains(r"\?").astype(int)
        df["url_has_ip"] = url.str.contains(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", regex=True).astype(int)
        df["url_has_sql_chars"] = url.str.contains(r"(?:%27|'|--|%23|#)", regex=True).astype(int)
        df["url_has_cmd_chars"] = url.str.contains(r"(?:;|\|\||&&|\$\(|`)", regex=True).astype(int)
        df["url_has_xss"] = url.str.contains(r"(?:<script|%3cscript|onerror=|onload=)", regex=True, case=False).astype(int)

    # --- user-agent features ---
    if "user_agent" in df.columns:
        ua = df["user_agent"].astype(str).fillna("")
        df["ua_len"] = ua.str.len()
        low = ua.str.lower()
        df["ua_is_curl"] = low.str.contains("curl").astype(int)
        df["ua_is_wget"] = low.str.contains("wget").astype(int)
        df["ua_is_python"] = low.str.contains("python").astype(int)
        df["ua_is_browser"] = low.str.contains("mozilla|chrome|safari|firefox|edge").astype(int)

    # --- internal traffic feature (if possible) ---
    if "is_internal_traffic" not in df.columns and "src_ip" in df.columns and "dst_ip" in df.columns:
        src_private = df["src_ip"].astype(str).apply(is_private_ip)
        dst_private = df["dst_ip"].astype(str).apply(is_private_ip)
        df["is_internal_traffic"] = (src_private & dst_private).astype(int)

    return df

def group_attack_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "attack_type" not in df.columns:
        raise ValueError("Expected column 'attack_type' not found in cybersecurity.csv")

    df["attack_type"] = df["attack_type"].astype(str).str.strip()
    df["attack_group"] = df["attack_type"].map(GROUP_MAP)

    # Anything unmapped becomes "other_attack" (still an attack)
    df["attack_group"] = df["attack_group"].fillna("other_attack")
    return df

def build_preprocessor(X: pd.DataFrame):
    # treat protocol as categorical if present and non-numeric
    cat_cols = []
    if "protocol" in X.columns and not pd.api.types.is_numeric_dtype(X["protocol"]):
        cat_cols.append("protocol")

    # everything else numeric-ish
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop"
    )
    return pre

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = add_features(df)
    df = group_attack_type(df)

    # y: grouped multi-class
    y = df["attack_group"].astype(str).str.strip()

    # binary label for stage 1
    y_bin = (y != "benign").astype(int)

    # features: drop label columns + obvious text columns (url/user_agent) after engineered features
    drop_cols = ["attack_type", "attack_group"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Drop raw text cols (we use engineered features instead)
    for raw_text in ["url", "user_agent", "timestamp"]:
        if raw_text in X.columns:
            X = X.drop(columns=[raw_text])

    # (optional) drop raw IPs if present (too high-cardinality)
    for ipcol in ["src_ip", "dst_ip"]:
        if ipcol in X.columns:
            X = X.drop(columns=[ipcol])

    # Ensure numeric coercion where possible (except protocol)
    for c in X.columns:
        if c != "protocol" and not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    pre = build_preprocessor(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores = []
    fold_preds = []
    fold_trues = []

    for fold, (tr, te) in enumerate(skf.split(X, y_bin), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        ybin_tr, ybin_te = y_bin.iloc[tr], y_bin.iloc[te]

        # -------- Stage 1: binary (benign vs attack) --------
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
        attack_pred = stage1.predict(X_te)  # 0/1

        # -------- Stage 2: multi-class on attacks only --------
        # Train on true attacks only
        tr_attack_mask = (ybin_tr == 1).to_numpy()
        X_tr_attack = X_tr.iloc[tr_attack_mask]
        y_tr_attack = y_tr.iloc[tr_attack_mask].astype(str).str.strip()

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
        te_attack_mask = (attack_pred == 1)

        if te_attack_mask.sum() > 0:
            pred_enc = stage2.predict(X_te.iloc[te_attack_mask]).astype(int)
            pred_str = le_attack.inverse_transform(pred_enc)
            final_pred[te_attack_mask] = pred_str

        y_true = y_te.astype(str).str.strip().to_numpy()

        macro = f1_score(y_true, final_pred, average="macro", zero_division=0)
        scores.append(float(macro))

        fold_trues.append(y_true)
        fold_preds.append(final_pred)

        print(f"Fold {fold}: macro-F1={macro:.4f}")

    print(f"\nTwo-stage SIEM-like 5-fold macro-F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

if __name__ == "__main__":
    main()