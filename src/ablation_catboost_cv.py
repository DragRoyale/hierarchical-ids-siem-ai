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

    # --- time ---
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- URL ---
    url = df.get("url", pd.Series([""] * len(df))).astype(str).fillna("")
    df["url_len"] = url.str.len()
    df["url_num_params"] = url.str.count(r"[?&]")
    df["url_has_ip"] = url.str.contains(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", regex=True).astype(int)
    df["url_has_sql_chars"] = url.str.contains(r"(?:%27|'|--|%23|#)", regex=True).astype(int)
    df["url_has_cmd_chars"] = url.str.contains(r"(?:;|\|\||&&|\$\(.+\)|`)", regex=True).astype(int)
    df["url_has_xss"] = url.str.contains(r"(?:<script|%3cscript|onerror=|onload=)", regex=True, case=False).astype(int)

    # --- UA ---
    ua = df.get("user_agent", pd.Series([""] * len(df))).astype(str).fillna("").str.lower()
    df["ua_len"] = ua.str.len()
    df["ua_is_curl"] = ua.str.contains("curl").astype(int)
    df["ua_is_wget"] = ua.str.contains("wget").astype(int)
    df["ua_is_python"] = ua.str.contains("python|requests|urllib").astype(int)
    df["ua_is_browser"] = ua.str.contains("mozilla|chrome|safari|firefox|edge").astype(int)

    # --- internal flag ---
    if "is_internal_traffic" in df.columns:
        df["is_internal_traffic"] = (
            df["is_internal_traffic"].astype(str).str.lower().map({"true": 1, "false": 0})
        )

    # --- ports ---
    for col in ["src_port", "dst_port"]:
        if col in df.columns:
            p = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_is_web"] = p.isin([80, 443, 8080, 8443]).astype(int)
            df[f"{col}_is_ssh"] = p.isin([22]).astype(int)
            df[f"{col}_is_db"]  = p.isin([3306, 5432, 1433]).astype(int)

    # --- bytes ---
    if "bytes_sent" in df.columns and "bytes_received" in df.columns:
        bs = pd.to_numeric(df["bytes_sent"], errors="coerce").fillna(0)
        br = pd.to_numeric(df["bytes_received"], errors="coerce").fillna(0)
        df["bytes_total"] = bs + br
        df["bytes_ratio_sent_recv"] = bs / (br + 1.0)
        df["log_bytes_total"] = np.log1p(df["bytes_total"])

    return df

def select_features(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode in {"BASE","BASE+TIME","BASE+URL","BASE+UA","ALL"}
    BASE = protocol + ports + bytes + internal flag
    """
    base_cols = []

    # protocol is categorical (if exists)
    if "protocol" in df.columns:
        base_cols.append("protocol")

    # core numeric
    for c in [
        "src_port", "dst_port",
        "bytes_sent", "bytes_received",
        "is_internal_traffic",
        "bytes_total", "bytes_ratio_sent_recv", "log_bytes_total",
        "src_port_is_web", "src_port_is_ssh", "src_port_is_db",
        "dst_port_is_web", "dst_port_is_ssh", "dst_port_is_db",
    ]:
        if c in df.columns:
            base_cols.append(c)

    time_cols = [c for c in ["hour", "dayofweek", "is_weekend"] if c in df.columns]
    url_cols  = [c for c in ["url_len","url_num_params","url_has_ip","url_has_sql_chars","url_has_cmd_chars","url_has_xss"] if c in df.columns]
    ua_cols   = [c for c in ["ua_len","ua_is_curl","ua_is_wget","ua_is_python","ua_is_browser"] if c in df.columns]

    if mode == "BASE":
        cols = base_cols
    elif mode == "BASE+TIME":
        cols = base_cols + time_cols
    elif mode == "BASE+URL":
        cols = base_cols + url_cols
    elif mode == "BASE+UA":
        cols = base_cols + ua_cols
    elif mode == "ALL":
        cols = base_cols + time_cols + url_cols + ua_cols
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Drop duplicates and keep order
    seen = set()
    cols_unique = []
    for c in cols:
        if c not in seen:
            cols_unique.append(c)
            seen.add(c)

    return df[cols_unique].copy()

def cat_indices(X: pd.DataFrame):
    # CatBoost needs categorical feature indices (for protocol only here)
    return [X.columns.get_loc("protocol")] if "protocol" in X.columns else []

def run_cv(X: pd.DataFrame, y: pd.Series, cat_idx_list):
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
            verbose=False,
        )

        model.fit(X_tr, y_tr, cat_features=cat_idx_list, sample_weight=w)
        pred = model.predict(X_te).reshape(-1)
        scores.append(f1_score(y_te, pred, average="macro"))

    return float(np.mean(scores)), float(np.std(scores)), scores

def main():
    df = pd.read_csv("data/raw/cybersecurity.csv")

    y = group_attack_types(df["attack_type"])
    df = add_feature_engineering(df)

    # Remove leakage/high-cardinality raw identifiers (keep engineered features instead)
    drop_cols = ["attack_type", "label", "timestamp", "src_ip", "dst_ip", "url", "user_agent"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    modes = ["BASE", "BASE+TIME", "BASE+URL", "BASE+UA", "ALL"]
    results = []

    print("Grouped classes:\n", y.value_counts(), "\n")

    for mode in modes:
        X = select_features(df, mode)
        idx = cat_indices(X)
        mean_f1, std_f1, fold_scores = run_cv(X, y, idx)
        results.append((mode, mean_f1, std_f1, fold_scores))
        print(f"{mode:10s}  macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}  folds={['%.4f'%s for s in fold_scores]}")

    # Pretty summary table
    print("\n=== Ablation Summary (macro-F1) ===")
    for mode, mean_f1, std_f1, _ in results:
        print(f"{mode:10s}  {mean_f1:.4f} ± {std_f1:.4f}")

if __name__ == "__main__":
    main()