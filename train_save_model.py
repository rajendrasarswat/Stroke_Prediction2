"""Train XGBoost with notebook hyperparameters and save Models/XGBoostTunedModel.pkl."""
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Models"
MODEL_PATH = MODEL_DIR / "XGBoostTunedModel.pkl"


def main() -> None:
    df = pd.read_csv(ROOT / "train.csv", na_values=["N/A", "NaN", ""])
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df.drop(["id"], axis=1, inplace=True)
    df["bmi"] = df["bmi"].fillna(df.groupby("age")["bmi"].transform("mean"))
    df["bmi"].fillna(df["bmi"].mean(), inplace=True)

    label_gender = LabelEncoder()
    label_married = LabelEncoder()
    label_work = LabelEncoder()
    label_residence = LabelEncoder()
    label_smoking = LabelEncoder()

    df["gender"] = label_gender.fit_transform(df["gender"])
    df["ever_married"] = label_married.fit_transform(df["ever_married"])
    df["work_type"] = label_work.fit_transform(df["work_type"])
    df["Residence_type"] = label_residence.fit_transform(df["Residence_type"])
    df["smoking_status"] = label_smoking.fit_transform(df["smoking_status"])

    smote = SMOTE(sampling_strategy="minority", random_state=42)
    X, y = smote.fit_resample(df.loc[:, df.columns != "stroke"], df["stroke"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42
    )

    xgb_kw = dict(
        objective="reg:logistic",
        random_state=42,
        colsample_bytree=0.5,
        gamma=0.2,
        learning_rate=0.25,
        max_depth=10,
        min_child_weight=1,
        eval_metric="logloss",
    )
    try:
        model = XGBClassifier(use_label_encoder=False, **xgb_kw)
    except TypeError:
        model = XGBClassifier(**xgb_kw)

    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
