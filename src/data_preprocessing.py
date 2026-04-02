"""
DataAgent: Data preprocessing pipeline for credit risk prediction.

Loads the German Credit Dataset (UCI), engineers features,
and creates train/future splits to simulate data drift.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Column definitions (German Credit Dataset) ────────────────────────────────
COLUMN_NAMES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker", "class",
]

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/statlog/german/german.data"
)

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_german_credit() -> pd.DataFrame:
    """
    Load German Credit Dataset from local cache or UCI repository.
    Falls back to synthetic generation when neither is available.
    """
    local_path = os.path.join(RAW_DIR, "german_credit.csv")

    if os.path.exists(local_path):
        print("[DataAgent] Loading from local cache...")
        return pd.read_csv(local_path)

    try:
        print("[DataAgent] Downloading German Credit Dataset from UCI...")
        df = pd.read_csv(UCI_URL, sep=" ", header=None, names=COLUMN_NAMES)
        df.to_csv(local_path, index=False)
        print(f"[DataAgent] Saved to {local_path}")
        return df
    except Exception as exc:
        print(f"[DataAgent] Download failed ({exc}), generating synthetic data...")
        return _generate_synthetic_data()


def _generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic credit dataset with realistic distributions.
    Includes gender proxy and age for fairness analysis.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "checking_status": rng.choice(
                ["A11", "A12", "A13", "A14"], n, p=[0.27, 0.27, 0.06, 0.40]
            ),
            "duration": rng.integers(4, 72, n),
            "credit_history": rng.choice(
                ["A30", "A31", "A32", "A33", "A34"], n, p=[0.04, 0.05, 0.53, 0.09, 0.29]
            ),
            "purpose": rng.choice(
                ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49"], n
            ),
            "credit_amount": rng.integers(250, 18424, n),
            "savings_status": rng.choice(
                ["A61", "A62", "A63", "A64", "A65"], n, p=[0.60, 0.10, 0.06, 0.06, 0.18]
            ),
            "employment": rng.choice(
                ["A71", "A72", "A73", "A74", "A75"], n, p=[0.04, 0.17, 0.34, 0.24, 0.21]
            ),
            "installment_commitment": rng.integers(1, 5, n),
            "personal_status": rng.choice(
                ["A91", "A92", "A93", "A94"], n, p=[0.05, 0.31, 0.55, 0.09]
            ),
            "other_parties": rng.choice(["A101", "A102", "A103"], n, p=[0.91, 0.04, 0.05]),
            "residence_since": rng.integers(1, 5, n),
            "property_magnitude": rng.choice(
                ["A121", "A122", "A123", "A124"], n, p=[0.28, 0.22, 0.33, 0.17]
            ),
            "age": rng.integers(19, 75, n),
            "other_payment_plans": rng.choice(
                ["A141", "A142", "A143"], n, p=[0.14, 0.05, 0.81]
            ),
            "housing": rng.choice(["A151", "A152", "A153"], n, p=[0.18, 0.71, 0.11]),
            "existing_credits": rng.integers(1, 5, n),
            "job": rng.choice(
                ["A171", "A172", "A173", "A174"], n, p=[0.02, 0.20, 0.60, 0.18]
            ),
            "num_dependents": rng.integers(1, 3, n),
            "own_telephone": rng.choice(["A191", "A192"], n, p=[0.60, 0.40]),
            "foreign_worker": rng.choice(["A201", "A202"], n, p=[0.96, 0.04]),
            "class": rng.choice([1, 2], n, p=[0.70, 0.30]),
        }
    )
    df.to_csv(os.path.join(RAW_DIR, "german_credit.csv"), index=False)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data into model-ready features:
    - Binary target (1 = default)
    - Gender proxy from personal_status (for fairness analysis)
    - Age group bins (for fairness analysis)
    - Label-encoded categoricals
    """
    df = df.copy()

    # Target: 1 = default (class=2 in original), 0 = good credit
    df["default"] = (df["class"] == 2).astype(int)
    df.drop(columns=["class"], inplace=True)

    # Fairness attributes
    # personal_status A92 = female/married, rest = male (proxy)
    df["gender"] = df["personal_status"].apply(
        lambda x: "female" if x == "A92" else "male"
    )
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 100],
        labels=["18-25", "26-35", "36-50", "51+"],
        right=True,
    )

    # Encode categoricals
    cat_cols = [
        "checking_status", "credit_history", "purpose", "savings_status",
        "employment", "personal_status", "other_parties", "property_magnitude",
        "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker",
    ]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df["age_group_encoded"] = LabelEncoder().fit_transform(df["age_group"].astype(str))
    df["gender_encoded"] = LabelEncoder().fit_transform(df["gender"])

    return df


# ── Split & scale ─────────────────────────────────────────────────────────────

def split_and_scale(df: pd.DataFrame):
    """
    Temporal train / test / future split (70 / 15 / 15 %).
    'future' simulates the drifted production distribution.

    Returns:
        X_train, X_test, X_future,
        y_train, y_test, y_future,
        prot_train, prot_test, prot_future,
        feature_cols
    """
    protected_cols = ["gender", "age_group", "gender_encoded", "age_group_encoded"]
    target_col = "default"
    
    # AI Governance Unaware Model: Exclude demographic proxies from training features
    excluded_cols = protected_cols + [target_col, "age", "personal_status"]
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    X, y, protected = df[feature_cols], df[target_col], df[protected_cols]

    n = len(df)
    i_train, i_test = int(n * 0.70), int(n * 0.85)

    X_tr, X_te, X_fu = X.iloc[:i_train], X.iloc[i_train:i_test], X.iloc[i_test:]
    y_tr, y_te, y_fu = y.iloc[:i_train], y.iloc[i_train:i_test], y.iloc[i_test:]
    p_tr, p_te, p_fu = (
        protected.iloc[:i_train],
        protected.iloc[i_train:i_test],
        protected.iloc[i_test:],
    )

    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_cols)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=feature_cols)
    X_fu_s = pd.DataFrame(scaler.transform(X_fu), columns=feature_cols)
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))

    return (
        X_tr_s, X_te_s, X_fu_s,
        y_tr.reset_index(drop=True),
        y_te.reset_index(drop=True),
        y_fu.reset_index(drop=True),
        p_tr.reset_index(drop=True),
        p_te.reset_index(drop=True),
        p_fu.reset_index(drop=True),
        feature_cols,
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 60)
    print("[DataAgent] Starting data preprocessing pipeline...")
    print("=" * 60)

    df_raw = load_german_credit()
    print(f"[DataAgent] Raw data shape: {df_raw.shape}")

    df = engineer_features(df_raw)
    print(f"[DataAgent] Engineered data shape: {df.shape}")
    print(f"[DataAgent] Default rate: {df['default'].mean():.2%}")
    print(f"[DataAgent] Gender:\n{df['gender'].value_counts().to_string()}")
    print(f"[DataAgent] Age group:\n{df['age_group'].value_counts().to_string()}")

    (
        X_tr, X_te, X_fu,
        y_tr, y_te, y_fu,
        p_tr, p_te, p_fu,
        feature_cols,
    ) = split_and_scale(df)

    # Persist datasets
    for name, X, y, prot in [
        ("train", X_tr, y_tr, p_tr),
        ("test", X_te, y_te, p_te),
        ("future", X_fu, y_fu, p_fu),
    ]:
        pd.concat([X, y.rename("default")], axis=1).to_csv(
            os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False
        )
        pd.concat([X, prot, y.rename("default")], axis=1).to_csv(
            os.path.join(PROCESSED_DIR, f"{name}_full.csv"), index=False
        )

    print(f"\n[DataAgent] ✅ Saved to {PROCESSED_DIR}")
    print(f"  train:   {len(X_tr)} samples")
    print(f"  test:    {len(X_te)} samples")
    print(f"  future:  {len(X_fu)} samples (drift simulation)")
    print(f"  features: {len(feature_cols)}")
    print("=" * 60)


if __name__ == "__main__":
    run()
