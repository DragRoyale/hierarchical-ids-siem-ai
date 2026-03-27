import os, glob, re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

RANDOM_STATE = 42

def load_data(raw_dir="data/raw"):
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError("No CSV files found in data/raw/. Put dataset CSVs there.")
    print("Loading files:", paths)
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Timestamp → time features (safe parse)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    else:
        df["hour"] = np.nan
        df["dayofweek"] = np.nan
        df["is_weekend"] = np.nan

    # URL features
    if "url" in df.columns:
        url = df["url"].astype(str).fillna("")
        df["url_len"] = url.str.len()
        df["url_num_params"] = url.str.count(r"[?&]")
        df["url_has_ip"] = url.str.contains(r"\b\d{1,3}(\.\d{1,3}){3}\b", regex=True).astype(int)
        df["url_has_sql_chars"] = url.str.contains(r"(%27|'|--|%23|#)", regex=True).astype(int)
        df["url_has_cmd_chars"] = url.str.contains(r"(;|\|\||&&|\$\(.*\)|`)", regex=True).astype(int)
        df["url_has_xss"] = url.str.contains(r"(<script|%3cscript|onerror=|onload=)", regex=True, case=False).astype(int)
    else:
        df["url_len"] = np.nan
        df["url_num_params"] = np.nan
        df["url_has_ip"] = np.nan
        df["url_has_sql_chars"] = np.nan
        df["url_has_cmd_chars"] = np.nan
        df["url_has_xss"] = np.nan

    # User-agent features
    if "user_agent" in df.columns:
        ua = df["user_agent"].astype(str).fillna("").str.lower()
        df["ua_len"] = ua.str.len()
        df["ua_is_curl"] = ua.str.contains("curl").astype(int)
        df["ua_is_wget"] = ua.str.contains("wget").astype(int)
        df["ua_is_python"] = ua.str.contains("python|requests|urllib").astype(int)
        df["ua_is_browser"] = ua.str.contains("mozilla|chrome|safari|firefox|edge").astype(int)
    else:
        df["ua_len"] = np.nan
        df["ua_is_curl"] = np.nan
        df["ua_is_wget"] = np.nan
        df["ua_is_python"] = np.nan
        df["ua_is_browser"] = np.nan

    # Convert boolean to int if needed
    if "is_internal" in df.columns:
        df["is_internal"] = df["is_internal"].astype(str).str.lower().map({"true": 1, "false": 0})
    return df

def main():
    df = load_data("data/raw")

    # Targets
    if "attack_type" not in df.columns:
        raise ValueError("No 'attack_type' column found. Check your CSV headers.")
    y = df["attack_type"].astype(str)

    print("Unique attack types:", y.unique())
    if "label" in df.columns:
        print("Label distribution:\n", df["label"].value_counts())

    df = add_feature_engineering(df)

    # Drop columns that cause leakage / too high cardinality (raw IPs, raw strings)
    drop_cols = ["attack_type", "label", "timestamp", "src_ip", "dst_ip", "url", "user_agent"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Identify column types
    # Treat anything non-numeric as categorical (covers pandas "string" dtype)
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]

    print("Feature columns:", list(X.columns))
    print("Categorical:", categorical)
    print("Numeric:", numeric)
    print("X shape:", X.shape)

    # Split (stratify for multi-class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessor
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric),
            ("cat", cat_tf, categorical),
        ],
        remainder="drop"
    )

    models = {
        "hgb": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
    }

    os.makedirs("models", exist_ok=True)

    for name, clf in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        print("\n==============================")
        print("MODEL:", name)
        print(classification_report(y_test, preds, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))

        joblib.dump(pipe, f"models/{name}_multiclass.joblib")

    print("\nSaved pipelines to models/")

if __name__ == "__main__":
    main()