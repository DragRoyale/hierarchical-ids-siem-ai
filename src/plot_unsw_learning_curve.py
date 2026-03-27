import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

RAW_DIR = "data/raw"
TRAIN_FILE = os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv")
TEST_FILE  = os.path.join(RAW_DIR, "UNSW_NB15_testing-set.csv")

RANDOM_STATE = 42

def load_unsw_official():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    return train_df, test_df

def preprocess_for_catboost(df: pd.DataFrame, cat_cols):
    # Separate X/y
    y = df["attack_cat"].astype(str).str.strip()
    X = df.drop(columns=["attack_cat", "label"], errors="ignore").copy()

    # Impute categorical with "missing"
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("missing")

    # Impute numeric with median
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    return X, y

def main():
    train_df, test_df = load_unsw_official()

    cat_cols = [c for c in ["proto", "service", "state"] if c in train_df.columns]

    X_train, y_train = preprocess_for_catboost(train_df, cat_cols)
    X_test,  y_test  = preprocess_for_catboost(test_df,  cat_cols)

    # CatBoost wants indices of categorical columns
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        random_seed=RANDOM_STATE,
        verbose=200,
        od_type="Iter",
        od_wait=200
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_idx,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # Final macro-F1 on official test
    y_pred = model.predict(X_test)
    y_pred = np.array([str(v[0]) for v in y_pred])  # CatBoost returns shape (n,1)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print("\nOFFICIAL TEST macro-F1:", macro)
    print("Best iteration:", model.get_best_iteration())

    # Extract eval history
    evals = model.get_evals_result()
    learn = evals["learn"]["TotalF1:average=Macro"]
    valid = evals["validation"]["TotalF1:average=Macro"]

    os.makedirs("results", exist_ok=True)
    with open("results/unsw_catboost_evals.json", "w", encoding="utf-8") as f:
        json.dump({"learn": learn, "validation": valid}, f)

    # Plot (iteration curve)
    plt.figure(figsize=(7, 4.5))
    plt.plot(learn, label="Train (Macro-F1)")
    plt.plot(valid, label="Validation/Test (Macro-F1)")
    plt.axvline(model.get_best_iteration(), linestyle="--", label="Best iteration")
    plt.xlabel("Boosting iteration")
    plt.ylabel("Macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/unsw_catboost_learning_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()