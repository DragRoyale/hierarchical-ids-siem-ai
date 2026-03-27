import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42

RAW_DIR = "data/raw"
TRAIN_FILE = os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv")
TEST_FILE  = os.path.join(RAW_DIR, "UNSW_NB15_testing-set.csv")

def load_unsw():
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Missing: {TRAIN_FILE}")
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Missing: {TEST_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    # UNSW usually has: 'attack_cat' (multi-class) + 'label' (0/1)
    if "attack_cat" not in train_df.columns or "attack_cat" not in test_df.columns:
        raise ValueError("Expected column 'attack_cat' not found in UNSW CSVs.")
    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Expected column 'label' not found in UNSW CSVs.")

    return train_df, test_df

def prepare_X_y(df: pd.DataFrame):
    df = df.copy()

    # Target: attack category (multi-class). Normal traffic is typically "Normal".
    y = df["attack_cat"].astype(str)

    # Drop targets + obvious identifiers
    drop_cols = ["attack_cat", "label"]
    for c in ["id", "Id", "ID"]:
        if c in df.columns:
            drop_cols.append(c)

    X = df.drop(columns=drop_cols, errors="ignore")

    # CatBoost can handle categoricals directly; UNSW often uses these:
    # proto, service, state
    cat_cols_candidates = ["proto", "service", "state"]
    cat_cols = [c for c in cat_cols_candidates if c in X.columns]

    # Convert any non-numeric to string for CatBoost safety
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).fillna("NA")

    # Fill numeric NaNs (CatBoost can handle some missing, but keep clean)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if num_cols:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
        X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))

    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    return X, y, cat_cols, cat_idx

def train_official_split():
    train_df, test_df = load_unsw()

    X_train, y_train, cat_cols, cat_idx = prepare_X_y(train_df)
    X_test,  y_test,  _,       _        = prepare_X_y(test_df)

    print("UNSW official split:")
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Categorical cols:", cat_cols)
    print("Train class counts (top 12):\n", y_train.value_counts().head(12))

    # Balanced sample weights
    w = compute_sample_weight(class_weight="balanced", y=y_train)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=RANDOM_STATE,
        iterations=2500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        verbose=200
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_idx,
        sample_weight=w,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    preds = model.predict(X_test).reshape(-1)
    macro = f1_score(y_test, preds, average="macro")

    print("\n=== OFFICIAL TEST RESULTS ===")
    print("Macro-F1:", macro)
    print(classification_report(y_test, preds, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    os.makedirs("models", exist_ok=True)
    model.save_model("models/unsw_catboost_official.cbm")
    pd.Series(X_train.columns).to_csv("models/unsw_feature_columns.csv", index=False)
    print("\nSaved: models/unsw_catboost_official.cbm")

def cv_on_combined(n_splits=5):
    train_df, test_df = load_unsw()
    df = pd.concat([train_df, test_df], ignore_index=True)

    X, y, cat_cols, cat_idx = prepare_X_y(df)

    print("\nUNSW combined CV:")
    print("Combined shape:", X.shape)
    print("Categorical cols:", cat_cols)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
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
        m = f1_score(y_te, pred, average="macro")
        scores.append(m)
        print(f"Fold {fold}: macro-F1={m:.4f}")

    print(f"\nUNSW {n_splits}-fold macro-F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

if __name__ == "__main__":
    # 1) Official train/test split (most important for benchmark reporting)
    # train_official_split()

    # 2) Optional but strong for paper robustness:
    cv_on_combined(n_splits=5)