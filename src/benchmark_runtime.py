import os
import time
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.base import clone

from xgboost import XGBClassifier


RANDOM_STATE = 42

# ---------- Paths ----------
RAW_DIR = "data/raw"
UNSW_TRAIN = os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv")
UNSW_TEST  = os.path.join(RAW_DIR, "UNSW_NB15_testing-set.csv")
SIEM_FILE  = os.path.join(RAW_DIR, "cybersecurity.csv")

OUT_CSV = os.path.join("results", "runtime_benchmark.csv")


# ---------- Helpers ----------
def ensure_results_dir():
    os.makedirs("results", exist_ok=True)

def now():
    return time.perf_counter()

def timed_fit(model, X_train, y_train):
    t0 = now()
    model.fit(X_train, y_train)
    t1 = now()
    return t1 - t0

def timed_predict(model, X, repeats=5, warmup=1):
    # Warmup
    for _ in range(warmup):
        _ = model.predict(X)

    times = []
    for _ in range(repeats):
        t0 = now()
        _ = model.predict(X)
        t1 = now()
        times.append(t1 - t0)
    return float(np.mean(times)), float(np.std(times))

def build_preprocessor(X, cat_candidates):
    cat_cols = [c for c in cat_candidates if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])
    return pre, cat_cols, num_cols

def add_siem_features(df):
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


# ---------- Models ----------
def make_single_stage(pre, num_class):
    return Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            objective="multi:softmax",
            num_class=num_class,
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

def make_stage1(pre):
    return Pipeline([
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

def make_stage2(pre, num_class):
    return Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            objective="multi:softmax",
            num_class=num_class,
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


# ---------- Benchmarks ----------
def bench_unsw():
    train_df = pd.read_csv(UNSW_TRAIN)
    test_df  = pd.read_csv(UNSW_TEST)

    y_train = train_df["attack_cat"].astype(str)
    y_test  = test_df["attack_cat"].astype(str)

    X_train = train_df.drop(columns=["attack_cat", "label"], errors="ignore")
    X_test  = test_df.drop(columns=["attack_cat", "label"], errors="ignore")

    pre, cat_cols, num_cols = build_preprocessor(X_train, ["proto", "service", "state"])

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # Single-stage
    single = make_single_stage(clone(pre), num_class=len(le.classes_))
    t_fit_single = timed_fit(single, X_train, y_train_enc)
    t_pred_mean_single, t_pred_std_single = timed_predict(single, X_test, repeats=5, warmup=1)
    rows = len(X_test)
    thr_single = rows / t_pred_mean_single

    # Two-stage
    normal_label = "Normal"
    y_bin_train = (y_train != normal_label).astype(int)

    stage1 = make_stage1(clone(pre))
    t_fit_s1 = timed_fit(stage1, X_train, y_bin_train)

    # Train stage2 on attacks only
    attack_mask = (y_bin_train == 1).to_numpy()
    X_train_attack = X_train.iloc[attack_mask]
    y_train_attack = y_train.iloc[attack_mask]

    le2 = LabelEncoder()
    y2 = le2.fit_transform(y_train_attack)

    stage2 = make_stage2(clone(pre), num_class=len(le2.classes_))
    t_fit_s2 = timed_fit(stage2, X_train_attack, y2)

    # Predict timing: stage1 + stage2 on subset
    # Warmup
    _ = stage1.predict(X_test)
    _ = stage2.predict(X_test.iloc[:min(1000, len(X_test))])

    times = []
    for _rep in range(5):
        t0 = now()
        pred1 = stage1.predict(X_test)
        mask = (pred1 == 1)
        if mask.sum() > 0:
            _ = stage2.predict(X_test.iloc[mask])
        t1 = now()
        times.append(t1 - t0)

    t_pred_mean_two = float(np.mean(times))
    t_pred_std_two  = float(np.std(times))
    thr_two = rows / t_pred_mean_two

    # Optional: quality quick check (macro-F1 on official split, single-stage only)
    # (keep it light; runtime is main goal)
    macro_single = f1_score(y_test_enc, single.predict(X_test), average="macro", zero_division=0)

    return [
        dict(dataset="UNSW_official", model="single_stage",
             train_rows=len(X_train), test_rows=len(X_test),
             fit_time_s=t_fit_single,
             pred_time_mean_s=t_pred_mean_single, pred_time_std_s=t_pred_std_single,
             rows_per_sec=thr_single,
             notes=f"macroF1_single={macro_single:.4f}"
        ),
        dict(dataset="UNSW_official", model="two_stage",
             train_rows=len(X_train), test_rows=len(X_test),
             fit_time_s=(t_fit_s1 + t_fit_s2),
             pred_time_mean_s=t_pred_mean_two, pred_time_std_s=t_pred_std_two,
             rows_per_sec=thr_two,
             notes=f"fit_stage1={t_fit_s1:.2f}s; fit_stage2={t_fit_s2:.2f}s"
        ),
    ]

def bench_siem_like():
    df = pd.read_csv(SIEM_FILE)
    df = add_siem_features(df)

    # Group labels (same mapping we used)
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

    df["attack_type"] = df["attack_type"].astype(str).str.strip()
    y = df["attack_type"].map(GROUP_MAP).fillna("other_attack").astype(str)

    drop_cols = ["attack_type", "url", "user_agent", "timestamp", "src_ip", "dst_ip"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Make numeric where possible except protocol
    for c in X.columns:
        if c != "protocol":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    pre, cat_cols, num_cols = build_preprocessor(X_train, ["protocol"])

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # Single-stage
    single = make_single_stage(clone(pre), num_class=len(le.classes_))
    t_fit_single = timed_fit(single, X_train, y_train_enc)
    t_pred_mean_single, t_pred_std_single = timed_predict(single, X_test, repeats=7, warmup=1)
    rows = len(X_test)
    thr_single = rows / t_pred_mean_single

    # Two-stage
    normal_label = "benign"
    y_bin_train = (y_train != normal_label).astype(int)

    stage1 = make_stage1(clone(pre))
    t_fit_s1 = timed_fit(stage1, X_train, y_bin_train)

    attack_mask = (y_bin_train == 1).to_numpy()
    X_train_attack = X_train.iloc[attack_mask]
    y_train_attack = y_train.iloc[attack_mask]

    le2 = LabelEncoder()
    y2 = le2.fit_transform(y_train_attack)

    stage2 = make_stage2(clone(pre), num_class=len(le2.classes_))
    t_fit_s2 = timed_fit(stage2, X_train_attack, y2)

    # Predict timing: stage1 + stage2 on subset
    _ = stage1.predict(X_test)
    _ = stage2.predict(X_test.iloc[:min(1000, len(X_test))])

    times = []
    for _rep in range(7):
        t0 = now()
        pred1 = stage1.predict(X_test)
        mask = (pred1 == 1)
        if mask.sum() > 0:
            _ = stage2.predict(X_test.iloc[mask])
        t1 = now()
        times.append(t1 - t0)

    t_pred_mean_two = float(np.mean(times))
    t_pred_std_two  = float(np.std(times))
    thr_two = rows / t_pred_mean_two

    macro_single = f1_score(y_test_enc, single.predict(X_test), average="macro", zero_division=0)

    return [
        dict(dataset="SIEM_like_split", model="single_stage",
             train_rows=len(X_train), test_rows=len(X_test),
             fit_time_s=t_fit_single,
             pred_time_mean_s=t_pred_mean_single, pred_time_std_s=t_pred_std_single,
             rows_per_sec=thr_single,
             notes=f"macroF1_single={macro_single:.4f}"
        ),
        dict(dataset="SIEM_like_split", model="two_stage",
             train_rows=len(X_train), test_rows=len(X_test),
             fit_time_s=(t_fit_s1 + t_fit_s2),
             pred_time_mean_s=t_pred_mean_two, pred_time_std_s=t_pred_std_two,
             rows_per_sec=thr_two,
             notes=f"fit_stage1={t_fit_s1:.2f}s; fit_stage2={t_fit_s2:.2f}s"
        ),
    ]


def main():
    ensure_results_dir()
    rows = []
    rows += bench_unsw()
    rows += bench_siem_like()

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print("\n=== Runtime Benchmark Saved ===")
    print(OUT_CSV)
    print(out)

if __name__ == "__main__":
    main()