import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42

def group_attack_types(y: pd.Series) -> pd.Series:
    y = y.astype(str)
    mapping = {
        "benign": "benign",
        "port-scan": "scanning",
        "brute-force": "credential_attack",
        "credential-stuffing": "credential_attack",
        "sql-injection": "injection",
        "command-injection": "injection",
        "xss": "injection",
        "ddos": "ddos",
        "c2": "malware_c2",
        "exploit-attempt": "malware_c2",
    }
    return y.map(lambda v: mapping.get(v, "other"))

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    url = df.get("url", pd.Series([""] * len(df))).astype(str).fillna("")
    df["url_len"] = url.str.len()
    df["url_num_params"] = url.str.count(r"[?&]")
    df["url_has_ip"] = url.str.contains(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", regex=True).astype(int)
    df["url_has_sql_chars"] = url.str.contains(r"(?:%27|'|--|%23|#)", regex=True).astype(int)
    df["url_has_cmd_chars"] = url.str.contains(r"(?:;|\|\||&&|\$\(.+\)|`)", regex=True).astype(int)
    df["url_has_xss"] = url.str.contains(r"(?:<script|%3cscript|onerror=|onload=)", regex=True, case=False).astype(int)

    ua = df.get("user_agent", pd.Series([""] * len(df))).astype(str).fillna("").str.lower()
    df["ua_len"] = ua.str.len()
    df["ua_is_curl"] = ua.str.contains("curl").astype(int)
    df["ua_is_wget"] = ua.str.contains("wget").astype(int)
    df["ua_is_python"] = ua.str.contains("python|requests|urllib").astype(int)
    df["ua_is_browser"] = ua.str.contains("mozilla|chrome|safari|firefox|edge").astype(int)

    if "is_internal_traffic" in df.columns:
        df["is_internal_traffic"] = df["is_internal_traffic"].astype(str).str.lower().map({"true": 1, "false": 0})

    for col in ["src_port", "dst_port"]:
        if col in df.columns:
            p = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_is_web"] = p.isin([80, 443, 8080, 8443]).astype(int)
            df[f"{col}_is_ssh"] = p.isin([22]).astype(int)
            df[f"{col}_is_db"]  = p.isin([3306, 5432, 1433]).astype(int)

    if "bytes_sent" in df.columns and "bytes_received" in df.columns:
        bs = pd.to_numeric(df["bytes_sent"], errors="coerce").fillna(0)
        br = pd.to_numeric(df["bytes_received"], errors="coerce").fillna(0)
        df["bytes_total"] = bs + br
        df["bytes_ratio_sent_recv"] = bs / (br + 1.0)
        df["log_bytes_total"] = np.log1p(df["bytes_total"])

    return df

def main():
    df = pd.read_csv("data/raw/cybersecurity.csv")
    df = add_feature_engineering(df)

    y = group_attack_types(df["attack_type"])
    drop_cols = ["attack_type", "label", "timestamp", "src_ip", "dst_ip", "url", "user_agent"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    cat_cols = ["protocol"] if "protocol" in X.columns else []
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    print("Grouped classes:\n", y.value_counts())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        w = compute_sample_weight(class_weight="balanced", y=y_tr)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            random_seed=RANDOM_STATE,
            iterations=2000,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            verbose=False
        )

        model.fit(X_tr, y_tr, cat_features=cat_idx, sample_weight=w)
        pred = model.predict(X_te).reshape(-1)

        macro = f1_score(y_te, pred, average="macro")
        scores.append(macro)
        print(f"Fold {fold}: macro-F1={macro:.4f}")

    print(f"\nCatBoost 5-fold macro-F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

if __name__ == "__main__":
    main()