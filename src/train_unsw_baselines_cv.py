import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
RAW_DIR = "data/raw"
TRAIN_FILE = os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv")
TEST_FILE  = os.path.join(RAW_DIR, "UNSW_NB15_testing-set.csv")

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def load_unsw_combined():
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        raise FileNotFoundError("Missing UNSW CSVs in data/raw/")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    df = pd.concat([train_df, test_df], ignore_index=True)
    if "attack_cat" not in df.columns:
        raise ValueError("Expected column 'attack_cat' not found.")
    return df

def prepare_X_y(df: pd.DataFrame, drop_categoricals: bool):
    df = df.copy()
    y = df["attack_cat"].astype(str)

    # Drop target columns + optional id column
    X = df.drop(columns=["attack_cat", "label"], errors="ignore")
    for c in ["id", "Id", "ID"]:
        if c in X.columns:
            X = X.drop(columns=[c], errors="ignore")

    # UNSW common categoricals
    cat_cols = [c for c in ["proto", "service", "state"] if c in X.columns]
    if drop_categoricals and cat_cols:
        X = X.drop(columns=cat_cols, errors="ignore")
        cat_cols = []

    # Detect remaining categoricals robustly
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    # Numeric: median impute
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Categorical: most-frequent + onehot
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ],
        remainder="drop"
    )
    return pre

def run_cv(drop_categoricals: bool):
    df = load_unsw_combined()
    X, y, num_cols, cat_cols = prepare_X_y(df, drop_categoricals=drop_categoricals)
    # Encode labels for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Combined shape:", X.shape)
    print("Categorical columns used:", cat_cols[:10], ("...(more)" if len(cat_cols) > 10 else ""))
    print("Class count (top 10):\n", y.value_counts().head(10))

    pre = build_preprocessor(num_cols, cat_cols)

    models = {
        "hgb": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_depth=None,
            learning_rate=0.1,
            max_iter=300
        ),
        "xgb": XGBClassifier(
            objective="multi:softmax",
            num_class=y.nunique(),
            n_estimators=800,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_scores = {name: [] for name in models.keys()}

    for fold, (tr, te) in enumerate(skf.split(X, y_encoded), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr_str, y_te_str = y.iloc[tr], y.iloc[te]
        y_tr = y_encoded[tr]
        y_te = y_encoded[te]

        w = compute_sample_weight(class_weight="balanced", y=y_tr_str)

        fold_scores = {}

        # ---- HGB ----
        try:
            pipe_hgb = Pipeline([("pre", pre), ("clf", models["hgb"])])
            pipe_hgb.fit(X_tr, y_tr_str, clf__sample_weight=w)
            pred_hgb = pipe_hgb.predict(X_te)
            fold_scores["hgb"] = float(f1_score(y_te_str, pred_hgb, average="macro"))
        except Exception as e:
            print(f"[ERROR] Fold {fold} HGB: {type(e).__name__}: {e}")
            fold_scores["hgb"] = float("nan")

        # ---- XGB ----
        try:
            pipe_xgb = Pipeline([("pre", pre), ("clf", models["xgb"])])
            pipe_xgb.fit(X_tr, y_tr, clf__sample_weight=w)
            pred_enc = pipe_xgb.predict(X_te).astype(int)
            pred_xgb = le.inverse_transform(pred_enc)
            fold_scores["xgb"] = float(f1_score(y_te_str, pred_xgb, average="macro"))
        except Exception as e:
            print(f"[ERROR] Fold {fold} XGB: {type(e).__name__}: {e}")
            fold_scores["xgb"] = float("nan")

        # Append exactly once per fold
        for k in models.keys():
            all_scores[k].append(fold_scores[k])

        print(f"Fold {fold}: hgb={fold_scores['hgb']:.4f}  xgb={fold_scores['xgb']:.4f}")

        parts = []
        for k in models.keys():
            if len(all_scores[k]) == fold:  # score exists for this fold
                parts.append(f"{k}={all_scores[k][-1]:.4f}")
            else:
                parts.append(f"{k}=FAIL")
        print(f"Fold {fold}: " + "  ".join(parts))

    # Print summary
    print("\n=== UNSW 5-fold CV macro-F1 ===")
    for name, scores in all_scores.items():
        print(f"{name:>4s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}   folds={['%.4f'%s for s in scores]}")

    # Save scores for significance tests
    suffix = "dropCats" if drop_categoricals else "withCats"
    out_path = os.path.join(OUT_DIR, f"unsw_baselines_cv_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)
    print("\nSaved fold scores to:", out_path)

if __name__ == "__main__":
    # Main benchmark run (use categoricals)
    run_cv(drop_categoricals=False)

    # Optional ablation (drop proto/service/state)
    # run_cv(drop_categoricals=True)