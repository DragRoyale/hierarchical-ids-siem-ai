import os, glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42

def load_data(raw_dir="data/raw"):
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError("No CSV files found in data/raw/. Put dataset CSVs there.")
    print("Loading files:", paths)
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def add_feature_engineering(df):
    df = df.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    if "url" in df.columns:
        url = df["url"].astype(str).fillna("")
        df["url_len"] = url.str.len()
        df["url_num_params"] = url.str.count(r"[?&]")
        df["url_has_ip"] = url.str.contains(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", regex=True).astype(int)
        df["url_has_sql_chars"] = url.str.contains(r"(?:%27|'|--|%23|#)", regex=True).astype(int)
        df["url_has_cmd_chars"] = url.str.contains(r"(?:;|\|\||&&|\$\(.+\)|`)", regex=True).astype(int)
        df["url_has_xss"] = url.str.contains(r"(?:<script|%3cscript|onerror=|onload=)", regex=True, case=False).astype(int)

    if "user_agent" in df.columns:
        ua = df["user_agent"].astype(str).fillna("").str.lower()
        df["ua_len"] = ua.str.len()
        df["ua_is_curl"] = ua.str.contains("curl").astype(int)
        df["ua_is_wget"] = ua.str.contains("wget").astype(int)
        df["ua_is_python"] = ua.str.contains("python|requests|urllib").astype(int)
        df["ua_is_browser"] = ua.str.contains("mozilla|chrome|safari|firefox|edge").astype(int)

    if "is_internal_traffic" in df.columns:
        df["is_internal_traffic"] = df["is_internal_traffic"].astype(str).str.lower().map({"true": 1, "false": 0})

    return df

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

def build_X_y(df):
    if "attack_type" not in df.columns:
        raise ValueError("Missing attack_type column.")
    y = group_attack_types(df["attack_type"])

    df = add_feature_engineering(df)

    drop_cols = ["attack_type", "label", "timestamp", "src_ip", "dst_ip", "url", "user_agent"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return X, y, pre

def main():
    df = load_data("data/raw")
    X, y, pre = build_X_y(df)

    print("Grouped classes:", y.value_counts())

    models = {
        "hgb": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "rf_balanced": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in models.items():
        fold_scores = []
        print("\n==============================")
        print("MODEL:", name)

        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]

            pipe = Pipeline([("pre", pre), ("clf", clf)])

            # sample weights help imbalance (especially for HGB)
            w = compute_sample_weight(class_weight="balanced", y=y_tr)
            pipe.fit(X_tr, y_tr, clf__sample_weight=w) if name == "hgb" else pipe.fit(X_tr, y_tr)

            pred = pipe.predict(X_te)
            score = f1_score(y_te, pred, average="macro")
            fold_scores.append(score)
            print(f"Fold {fold}: macro-F1={score:.4f}")

        print(f"Mean macro-F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    print("\nDone.")

if __name__ == "__main__":
    main()