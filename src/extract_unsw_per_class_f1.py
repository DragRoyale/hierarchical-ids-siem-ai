import os
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from sklearn.metrics import f1_score




RANDOM_STATE = 42
RAW_DIR = "data/raw"
TRAIN_FILE = os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv")
TEST_FILE  = os.path.join(RAW_DIR, "UNSW_NB15_testing-set.csv")

def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    return train_df, test_df

def build_preprocessor(X):

    cat_cols = [c for c in ["proto", "service", "state"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

    return pre

def main():

    train_df, test_df = load_data()

    y_train = train_df["attack_cat"].astype(str)
    y_test  = test_df["attack_cat"].astype(str)

    drop_cols = ["attack_cat", "label"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    pre = build_preprocessor(X_train)

    # ---------------- SINGLE-STAGE ----------------
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    single_model = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            objective="multi:softmax",
            num_class=len(le.classes_),
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

    single_model.fit(X_train, y_train_enc)
    single_pred = single_model.predict(X_test)

    report_single = classification_report(
        y_test_enc,
        single_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    # ---------------- TWO-STAGE ----------------
    normal_label = "Normal"
    y_bin_train = (y_train != normal_label).astype(int)
    y_bin_test  = (y_test != normal_label).astype(int)

    stage1 = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            objective="binary:logistic",
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

    stage1.fit(X_train, y_bin_train)
    proba = stage1.predict_proba(X_test)[:,1]
    attack_pred = (proba > 0.3).astype(int)

    # Stage 2 train only on attacks
    attack_mask_train = y_bin_train == 1
    X_train_attack = X_train[attack_mask_train]
    y_train_attack = y_train[attack_mask_train]

    le_attack = LabelEncoder()
    y_train_attack_enc = le_attack.fit_transform(y_train_attack)

    stage2 = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            objective="multi:softmax",
            num_class=len(le_attack.classes_),
            n_estimators=1200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

    stage2.fit(X_train_attack, y_train_attack_enc)

    final_pred = np.full(len(X_test), normal_label, dtype=object)
    mask_test = attack_pred == 1

    if mask_test.sum() > 0:
        pred_enc = stage2.predict(X_test[mask_test])
        pred_str = le_attack.inverse_transform(pred_enc)
        final_pred[mask_test] = pred_str

    report_two = classification_report(
        y_test,
        final_pred,
        output_dict=True,
        zero_division=0
    )

    # ---------------- SAVE RESULTS ----------------
    df_out = pd.DataFrame({
        "Class": le.classes_,
        "Single_F1": [report_single[c]["f1-score"] for c in le.classes_],
        "TwoStage_F1": [report_two[c]["f1-score"] for c in le.classes_],
    })

    df_out["Improvement"] = df_out["TwoStage_F1"] - df_out["Single_F1"]

    df_out.to_csv("results/unsw_per_class_f1.csv", index=False)

    macro_single = f1_score(y_test_enc, single_pred, average="macro")
    macro_two = f1_score(y_test, final_pred, average="macro")

    print("\nOfficial split macro-F1:")
    print("Single-stage:", macro_single)
    print("Two-stage:", macro_two)

    print(df_out.sort_values("Improvement", ascending=False))

if __name__ == "__main__":
    main()