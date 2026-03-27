import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

RANDOM_STATE = 42
RAW_DIR = "data/raw"
TRAIN_FILE = os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv")
TEST_FILE  = os.path.join(RAW_DIR, "UNSW_NB15_testing-set.csv")

def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    df = pd.concat([train_df, test_df], ignore_index=True)
    return df

def prepare_X_y(df):
    y = df["attack_cat"].astype(str).str.strip()
    X = df.drop(columns=["attack_cat", "label"], errors="ignore")

    cat_cols = [c for c in ["proto", "service", "state"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

    return X, y, pre

def main():
    df = load_data()
    X, y, pre = prepare_X_y(df)

    normal_label = "Normal"
    if normal_label not in set(y.unique()):
        # fallback: find a label that equals "normal" ignoring case/spaces
        for v in y.unique():
            if str(v).strip().lower() == "normal":
                normal_label = str(v).strip()
                break
    print("Normal label used:", normal_label)

    # Binary labels
    y_binary = (y != normal_label).astype(int)

    # Encode multi-class labels
    le = LabelEncoder()
    y_multi = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores = []

    for fold, (tr, te) in enumerate(skf.split(X, y_binary), 1):

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_bin_tr, y_bin_te = y_binary.iloc[tr], y_binary.iloc[te]
        y_multi_tr, y_multi_te = y_multi[tr], y_multi[te]

        # ---------- Stage 1 (Binary) ----------
        bin_model = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                objective="binary:logistic",
                n_estimators=600,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])

        bin_model.fit(X_tr, y_bin_tr)

        # Predict attack or not
        attack_pred = bin_model.predict(X_te)

        # ---------- Stage 2 (Multi-class on attacks) ----------

        attack_mask_tr = y_bin_tr == 1
        attack_mask_te = attack_pred == 1

        X_tr_attack = X_tr[attack_mask_tr]
        y_tr_attack_str = y.iloc[tr][attack_mask_tr]

        # Re-encode attack labels only
        le_attack = LabelEncoder()
        y_tr_attack = le_attack.fit_transform(y_tr_attack_str)

        multi_model = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                objective="multi:softmax",
                num_class=len(le_attack.classes_),
                n_estimators=800,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])

        multi_model.fit(X_tr_attack, y_tr_attack)

        final_pred = np.full(len(X_te), normal_label, dtype=object)

        if attack_mask_te.sum() > 0:
            attack_preds_enc = multi_model.predict(X_te[attack_mask_te]).astype(int)
            attack_preds_str = le_attack.inverse_transform(attack_preds_enc)
            attack_preds_str = np.array([str(x).strip() for x in attack_preds_str], dtype=object)
            final_pred[attack_mask_te] = attack_preds_str

        y_true_str = y.iloc[te].astype(str).str.strip().to_numpy()

        if fold == 1:
            print("y_true unique (sample):", pd.Series(y_true_str).value_counts().head(5))
            print("y_pred unique (sample):", pd.Series(final_pred).value_counts().head(5))

        macro = f1_score(y_true_str, final_pred, average="macro", zero_division=0)
        scores.append(macro)

        print(f"Fold {fold}: macro-F1={macro:.4f}")

    print("\nTwo-stage mean macro-F1:", np.mean(scores), "±", np.std(scores))

if __name__ == "__main__":
    main()