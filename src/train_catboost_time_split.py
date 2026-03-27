import os
import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

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

    # Timestamp features
    if "timestamp" not in df.columns:
        raise ValueError("Your CSV must contain a 'timestamp' column for time-based split.")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # URL features
    url = df.get("url", pd.Series([""] * len(df))).astype(str).fillna("")
    df["url_len"] = url.str.len()
    df["url_num_params"] = url.str.count(r"[?&]")
    df["url_has_ip"] = url.str.contains(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", regex=True).astype(int)
    df["url_has_sql_chars"] = url.str.contains(r"(?:%27|'|--|%23|#)", regex=True).astype(int)
    df["url_has_cmd_chars"] = url.str.contains(r"(?:;|\|\||&&|\$\(.+\)|`)", regex=True).astype(int)
    df["url_has_xss"] = url.str.contains(r"(?:<script|%3cscript|onerror=|onload=)", regex=True, case=False).astype(int)

    # User-agent features
    ua = df.get("user_agent", pd.Series([""] * len(df))).astype(str).fillna("").str.lower()
    df["ua_len"] = ua.str.len()
    df["ua_is_curl"] = ua.str.contains("curl").astype(int)
    df["ua_is_wget"] = ua.str.contains("wget").astype(int)
    df["ua_is_python"] = ua.str.contains("python|requests|urllib").astype(int)
    df["ua_is_browser"] = ua.str.contains("mozilla|chrome|safari|firefox|edge").astype(int)

    # Internal traffic flag
    if "is_internal_traffic" in df.columns:
        df["is_internal_traffic"] = df["is_internal_traffic"].astype(str).str.lower().map({"true": 1, "false": 0})

    # Port features (very useful)
    for col in ["src_port", "dst_port"]:
        if col in df.columns:
            p = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_is_web"] = p.isin([80, 443, 8080, 8443]).astype(int)
            df[f"{col}_is_ssh"] = p.isin([22]).astype(int)
            df[f"{col}_is_db"]  = p.isin([3306, 5432, 1433]).astype(int)

    # Bytes features
    if "bytes_sent" in df.columns and "bytes_received" in df.columns:
        bs = pd.to_numeric(df["bytes_sent"], errors="coerce").fillna(0)
        br = pd.to_numeric(df["bytes_received"], errors="coerce").fillna(0)
        df["bytes_total"] = bs + br
        df["bytes_ratio_sent_recv"] = bs / (br + 1.0)
        df["log_bytes_total"] = np.log1p(df["bytes_total"])

    return df

def load_csv(path="data/raw/cybersecurity.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def time_split(df: pd.DataFrame, train_frac=0.70):
    df = df.copy()
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp_parsed")
    cut = int(len(df) * train_frac)
    train_df = df.iloc[:cut].copy()
    test_df  = df.iloc[cut:].copy()
    return train_df, test_df

def main():
    df = load_csv("data/raw/cybersecurity.csv")

    # Target
    if "attack_type" not in df.columns:
        raise ValueError("Missing attack_type column in CSV.")
    y = group_attack_types(df["attack_type"])

    print("Grouped class counts:\n", y.value_counts())

    df = add_feature_engineering(df)

    # Keep protocol as categorical for CatBoost (do NOT one-hot)
    # Drop raw high-cardinality identifiers to avoid leakage
    drop_cols = ["attack_type", "label", "timestamp", "timestamp_parsed", "src_ip", "dst_ip", "url", "user_agent"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = group_attack_types(df["attack_type"])

    # Identify categorical feature indices for CatBoost
    cat_cols = []
    if "protocol" in X.columns:
        cat_cols.append("protocol")
    cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

    # Time-based split
    df_for_split = df.copy()
    df_for_split["attack_type_grouped"] = y.values
    train_df, test_df = time_split(df_for_split, train_frac=0.70)

    X_train = train_df[X.columns]
    y_train = train_df["attack_type_grouped"]
    X_test  = test_df[X.columns]
    y_test  = test_df["attack_type_grouped"]

    # Compute class weights (balanced)
    classes = sorted(y_train.unique())
    class_counts = y_train.value_counts().to_dict()
    total = len(y_train)
    class_weights = [total / (len(classes) * class_counts[c]) for c in classes]
    class_to_weight = dict(zip(classes, class_weights))

    # CatBoost model
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=RANDOM_STATE,
        iterations=2500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        auto_class_weights=None,  # we pass our own
        verbose=200
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_feature_indices,
        sample_weight=y_train.map(class_to_weight).values,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    preds = model.predict(X_test).reshape(-1)
    print("\nTIME-SPLIT RESULTS (train early → test later)")
    print(classification_report(y_test, preds, digits=4, zero_division=0))
    print("Macro-F1:", f1_score(y_test, preds, average="macro"))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    os.makedirs("models", exist_ok=True)
    model.save_model("models/catboost_time_split.cbm")
    joblib.dump(list(X.columns), "models/feature_columns.joblib")
    print("\nSaved model to models/catboost_time_split.cbm")

if __name__ == "__main__":
    main()