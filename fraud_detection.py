#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Fraud Detection
=========================

Goal
----
Detect fraudulent transactions in financial data (PaySim mobile-money dataset).

Guidelines implemented
----------------------
1. Anomaly detection / classification : Isolation Forest and Autoencoder
2. Class imbalance handling            : SMOTE (SMOTE-NC, applied to training data only)
3. Performance metrics                 : F1-score and AUC-ROC (plus accuracy, precision,
                                         recall, PR-AUC, MAE, MSE, RMSE and R-squared)

Pipeline
--------
 1. Load data
 2. Data preprocessing + feature engineering
 3. Exploratory data analysis (EDA)
 4. Train / validation / test split (stratified)
 5. Feature scaling
 6. Model 1 - Isolation Forest        (unsupervised anomaly detection)
 7. Model 2 - Autoencoder             (anomaly detection via reconstruction error,
                                       trained on normal transactions only)
 8. Model 3 - SMOTE + Random Forest   (supervised classification on SMOTE-balanced
                                       training data; this is where SMOTE is used)
 9. Evaluation + comparison + plots

Usage
-----
    python fraud_detection.py                       # full dataset
    python fraud_detection.py --sample-frac 0.2     # quick run on a 20% stratified sample
    python fraud_detection.py --data-path path/to/file.csv --epochs 20

All figures and result tables are written to the ``outputs/`` folder.
"""

import os

# Silence TensorFlow info logs (must be set before TensorFlow is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved to disk, never shown
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

TARGET = "isFraud"

# Columns read from the CSV. The account-name identifier columns (nameOrig, nameDest) are
# unique IDs with no generalisable signal, so they are not loaded at all (saves memory).
USE_COLS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",          # float64 keeps balance-error features exact
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

# Fraud in this dataset only occurs in these two transaction types.
FRAUD_PRONE_TYPES = ["TRANSFER", "CASH_OUT"]

# Model feature groups (continuous first, binary last - the order matters for SMOTE-NC).
MONEY_COLS = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
]
CONTINUOUS_COLS = MONEY_COLS + ["hour_of_day"]
BINARY_COLS = ["type_TRANSFER", "orig_emptied", "dest_zero"]
FEATURE_COLS = CONTINUOUS_COLS + BINARY_COLS

# Resampling settings (applied to the training split only).
UNDERSAMPLE_RATIO = 0.10  # fraud : legit after random under-sampling of the majority class
SMOTE_RATIO = 0.50        # fraud : legit after SMOTE over-sampling of the minority class

# Isolation Forest settings.
IFOREST_ESTIMATORS = 200
IFOREST_MAX_SAMPLES = 256

# Random Forest settings (classifier trained on the SMOTE-balanced data).
RF_ESTIMATORS = 200


def banner(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> {path.name}")


def signed_log1p(values):
    """log1p that keeps the sign (balance-error features can be negative)."""
    return np.sign(values) * np.log1p(np.abs(values))


# --------------------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------------------
def find_dataset(data_path):
    if data_path is not None:
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path
    candidates = sorted(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV file found in {DATA_DIR}. Place the dataset there or pass --data-path."
        )
    return candidates[0]


def load_data(path, sample_frac):
    banner("1. LOADING DATA")
    print(f"  file: {path.name}")
    df = pd.read_csv(path, usecols=USE_COLS, dtype=DTYPES)
    print(f"  loaded {len(df):,} rows x {df.shape[1]} columns")

    if sample_frac < 1.0:
        # Stratified sample: every class is sampled at the same rate so the fraud rate is kept.
        df = (
            df.groupby(TARGET, group_keys=False)
            .sample(frac=sample_frac, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        print(f"  stratified {sample_frac:.0%} sample -> {len(df):,} rows")
    return df


# --------------------------------------------------------------------------------------
# 2. Data preprocessing + feature engineering
# --------------------------------------------------------------------------------------
def preprocess(df):
    banner("2. DATA PREPROCESSING & FEATURE ENGINEERING")

    # --- Data quality checks ---------------------------------------------------------
    n_missing = int(df.isna().sum().sum())
    print(f"  missing values            : {n_missing}")
    if n_missing:
        df = df.dropna().reset_index(drop=True)
        print(f"  rows after dropping NaNs  : {len(df):,}")

    n_dupes = int(df.duplicated().sum())
    print(f"  exact duplicate rows      : {n_dupes} (kept: the unique account IDs were dropped, "
          "so identical-looking rows can be genuinely different transactions)")

    neg = int((df[["amount", "oldbalanceOrg", "newbalanceOrig",
                   "oldbalanceDest", "newbalanceDest"]] < 0).sum().sum())
    print(f"  negative monetary values  : {neg}")

    # --- Snapshot of raw statistics used later by the EDA ------------------------------
    raw_summary = {
        "type_vs_fraud": pd.crosstab(df["type"], df[TARGET]),
        "flagged_vs_fraud": pd.crosstab(df["isFlaggedFraud"], df[TARGET]),
        "step_fraud": df.groupby(df["step"] // 24)[TARGET].agg(["sum", "count"]),
        "n_rows_raw": len(df),
    }
    print("\n  Fraud by transaction type (raw data):")
    print(raw_summary["type_vs_fraud"].to_string())

    # --- Keep only fraud-prone transaction types ---------------------------------------
    df = df[df["type"].isin(FRAUD_PRONE_TYPES)].copy()
    df["type"] = df["type"].astype(str)
    print(f"\n  kept only {FRAUD_PRONE_TYPES} -> {len(df):,} rows "
          f"({len(df) / raw_summary['n_rows_raw']:.1%} of the raw data)")

    # --- Feature engineering ---------------------------------------------------------
    # Balance-consistency errors: for a normal transaction the balances should reconcile.
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["hour_of_day"] = (df["step"] % 24).astype("int16")   # 1 step = 1 hour
    df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype("int8")
    df["orig_emptied"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype("int8")
    df["dest_zero"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype("int8")

    # --- Drop columns that must not be used as model inputs -----------------------------
    # isFlaggedFraud is the simulator's own rule-based flag (almost never set) and is a
    # label-derived field, so it is excluded to avoid leakage. 'step' is replaced by
    # hour_of_day, and 'type' by the binary type_TRANSFER column.
    df = df[FEATURE_COLS + [TARGET]].reset_index(drop=True)

    n_fraud = int(df[TARGET].sum())
    print(f"\n  engineered features       : {FEATURE_COLS}")
    print(f"  final dataset             : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  fraud transactions        : {n_fraud:,} ({n_fraud / len(df):.3%})")
    return df, raw_summary


# --------------------------------------------------------------------------------------
# 3. Exploratory data analysis
# --------------------------------------------------------------------------------------
def run_eda(df, raw_summary, fig_dir):
    banner("3. EXPLORATORY DATA ANALYSIS")
    rng = np.random.RandomState(RANDOM_STATE)

    print("\n  Dataset overview (after preprocessing):")
    print(df.head().to_string())
    print("\n  Descriptive statistics:")
    print(df.describe().T.to_string())

    counts = df[TARGET].value_counts().sort_index()
    print("\n  Class distribution:")
    print(f"    legitimate (0): {counts.get(0, 0):,}")
    print(f"    fraud      (1): {counts.get(1, 0):,}")
    print(f"    imbalance ratio: 1 fraud per {counts.get(0, 0) / max(counts.get(1, 1), 1):,.0f} legitimate")

    print("\n  Rule-based flag (isFlaggedFraud) vs. real fraud (raw data):")
    print(raw_summary["flagged_vs_fraud"].to_string())

    print("\n  Mean of each feature by class:")
    print(df.groupby(TARGET)[FEATURE_COLS].mean().T.to_string())

    # Figure 1: class distribution -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(x=counts.index.map({0: "Legitimate", 1: "Fraud"}), y=counts.values,
                palette=["#4c72b0", "#c44e52"], ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Number of transactions (log scale)")
    ax.set_title("Class distribution (after filtering)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
    save_fig(fig, fig_dir / "eda_01_class_distribution.png")

    # Figure 2: fraud by transaction type (raw data) --------------------------------------
    ct = raw_summary["type_vs_fraud"].rename(columns={0: "Legitimate", 1: "Fraud"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct.plot(kind="bar", ax=ax, color=["#4c72b0", "#c44e52"], logy=True)
    ax.set_ylabel("Number of transactions (log scale)")
    ax.set_xlabel("Transaction type")
    ax.set_title("Transactions by type and class (raw data)")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, fig_dir / "eda_02_fraud_by_type.png")

    # Figure 3: amount distribution by class ----------------------------------------------
    legit = df.loc[df[TARGET] == 0, "amount"]
    legit = legit.sample(min(len(legit), 200_000), random_state=RANDOM_STATE)
    fraud = df.loc[df[TARGET] == 1, "amount"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(np.log10(legit + 1), bins=60, stat="density", alpha=0.55, color="#4c72b0",
                 label="Legitimate", ax=ax)
    sns.histplot(np.log10(fraud + 1), bins=60, stat="density", alpha=0.55, color="#c44e52",
                 label="Fraud", ax=ax)
    ax.set_xlabel("log10(amount + 1)")
    ax.set_title("Transaction amount distribution by class")
    ax.legend()
    save_fig(fig, fig_dir / "eda_03_amount_distribution.png")

    # Figure 4: balance-error features by class --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, ["errorBalanceOrig", "errorBalanceDest"]):
        l = signed_log1p(df.loc[df[TARGET] == 0, col].sample(
            min(len(legit), 200_000), random_state=RANDOM_STATE))
        f = signed_log1p(df.loc[df[TARGET] == 1, col])
        sns.histplot(l, bins=60, stat="density", alpha=0.55, color="#4c72b0",
                     label="Legitimate", ax=ax)
        sns.histplot(f, bins=60, stat="density", alpha=0.55, color="#c44e52",
                     label="Fraud", ax=ax)
        ax.set_xlabel(f"signed log1p({col})")
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Balance-consistency errors by class")
    save_fig(fig, fig_dir / "eda_04_balance_errors.png")

    # Figure 5: fraud over time --------------------------------------------------------------
    sf = raw_summary["step_fraud"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(sf.index, sf["sum"], color="#c44e52")
    ax.set_xlabel("Day of simulation (step // 24)")
    ax.set_ylabel("Fraud transactions")
    ax.set_title("Fraud transactions per simulated day (raw data)")
    save_fig(fig, fig_dir / "eda_05_fraud_over_time.png")

    # Figure 6: fraud rate by hour of day ------------------------------------------------------
    hourly = df.groupby("hour_of_day")[TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(hourly.index, hourly.values, color="#dd8452")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate by hour of day")
    save_fig(fig, fig_dir / "eda_06_fraud_rate_by_hour.png")

    # Figure 7: correlation heatmap ------------------------------------------------------------
    n_corr = min(len(df), 500_000)
    sample = df.iloc[rng.choice(len(df), n_corr, replace=False)]
    corr = sample[FEATURE_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature correlation matrix")
    save_fig(fig, fig_dir / "eda_07_correlation_heatmap.png")

    top = corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    print("\n  Correlation of each feature with isFraud:")
    print(top.round(4).to_string())


# --------------------------------------------------------------------------------------
# 4. Splitting + 5. Scaling
# --------------------------------------------------------------------------------------
def split_data(df):
    banner("4. TRAIN / VALIDATION / TEST SPLIT (stratified)")
    X = df[FEATURE_COLS]
    y = df[TARGET].to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=RANDOM_STATE)

    for name, yy in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        print(f"  {name:<11}: {len(yy):>9,} rows | frauds: {int(yy.sum()):>5,} ({yy.mean():.3%})")
    print("  (train 70% / validation 10% / test 20%)")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    banner("5. FEATURE SCALING")
    print("  signed log1p on monetary columns, then StandardScaler on continuous columns")
    print("  (scaler fitted on the training split only - no leakage)")

    def log_transform(frame):
        out = frame.copy()
        out[MONEY_COLS] = signed_log1p(out[MONEY_COLS].to_numpy())
        return out

    X_train_l, X_val_l, X_test_l = map(log_transform, (X_train, X_val, X_test))

    scaler = StandardScaler()
    scaler.fit(X_train_l[CONTINUOUS_COLS])

    def transform(frame):
        out = frame.copy()
        out[CONTINUOUS_COLS] = scaler.transform(frame[CONTINUOUS_COLS])
        return out[FEATURE_COLS].to_numpy(dtype=np.float32)

    return transform(X_train_l), transform(X_val_l), transform(X_test_l)


# --------------------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------------------
def best_f1_threshold(y_true, scores):
    """Decision threshold that maximises F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def evaluate(name, y_true, y_pred, scores):
    """Return a dict with all requested evaluation statistics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, scores),
        "PR-AUC": average_precision_score(y_true, scores),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def report_model(name, y_val, val_scores, y_test, test_scores, results, store):
    """Tune the threshold on validation data, then evaluate once on the test set."""
    threshold, val_f1 = best_f1_threshold(y_val, val_scores)
    y_pred = (test_scores >= threshold).astype(int)
    metrics = evaluate(name, y_test, y_pred, test_scores)
    results.append(metrics)
    store[name] = {"scores": test_scores, "pred": y_pred}

    print(f"\n  {name}")
    print(f"    threshold tuned on validation set (max F1): {threshold:.6f}  (val F1 = {val_f1:.4f})")
    print("    test-set results:")
    for key, value in metrics.items():
        if key != "Model":
            print(f"      {key:<10}: {value:.6f}")
    print("    confusion matrix [[TN, FP], [FN, TP]]:")
    print("     ", confusion_matrix(y_test, y_pred).tolist())
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"],
                                digits=4, zero_division=0))


# --------------------------------------------------------------------------------------
# 6. Model 1 - Isolation Forest
# --------------------------------------------------------------------------------------
def run_isolation_forest(X_train, y_train, X_val, y_val, X_test, y_test, results, store):
    banner("6. MODEL 1 - ISOLATION FOREST (anomaly detection)")
    contamination = float(np.clip(y_train.mean(), 1e-4, 0.5))
    print(f"  n_estimators={IFOREST_ESTIMATORS}, max_samples={IFOREST_MAX_SAMPLES}, "
          f"contamination={contamination:.5f}")
    start = time.time()
    model = IsolationForest(
        n_estimators=IFOREST_ESTIMATORS,
        max_samples=IFOREST_MAX_SAMPLES,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Unsupervised: fitted on the training features only, labels are NOT used for fitting.
    model.fit(X_train)
    print(f"  trained in {time.time() - start:.1f}s")

    # score_samples: lower = more anomalous -> negate so that higher = more likely fraud.
    val_scores = -model.score_samples(X_val)
    test_scores = -model.score_samples(X_test)
    report_model("Isolation Forest", y_val, val_scores, y_test, test_scores, results, store)


# --------------------------------------------------------------------------------------
# 7. Model 2 - Autoencoder
# --------------------------------------------------------------------------------------
def build_autoencoder(input_dim):
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(4, activation="relu"),    # bottleneck
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(input_dim, activation="linear"),
        ],
        name="autoencoder",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def reconstruction_error(model, X, batch_size=8192):
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def run_autoencoder(X_train, y_train, X_val, y_val, X_test, y_test, epochs, results, store):
    banner("7. MODEL 2 - AUTOENCODER (anomaly detection via reconstruction error)")
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # The autoencoder learns what "normal" looks like: it is trained on legitimate
    # transactions only. Fraud is expected to reconstruct poorly (high error).
    X_normal = X_train[y_train == 0]
    X_fit, X_stop = train_test_split(X_normal, test_size=0.10, random_state=RANDOM_STATE)
    print(f"  training on {len(X_fit):,} legitimate transactions "
          f"({len(X_stop):,} held out for early stopping)")

    model = build_autoencoder(X_train.shape[1])
    model.summary(print_fn=lambda line: print("   ", line))

    start = time.time()
    history = model.fit(
        X_fit, X_fit,
        validation_data=(X_stop, X_stop),
        epochs=epochs,
        batch_size=2048,
        shuffle=True,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                          restore_best_weights=True),
        ],
    )
    print(f"  trained in {time.time() - start:.1f}s ({len(history.history['loss'])} epochs)")

    val_scores = reconstruction_error(model, X_val)
    test_scores = reconstruction_error(model, X_test)
    report_model("Autoencoder", y_val, val_scores, y_test, test_scores, results, store)
    return history


# --------------------------------------------------------------------------------------
# 8. Model 3 - SMOTE + Random Forest
# --------------------------------------------------------------------------------------
def run_smote_classifier(X_train, y_train, X_val, y_val, X_test, y_test, results, store):
    banner("8. MODEL 3 - SMOTE + RANDOM FOREST (supervised classification)")
    print(f"  class counts before resampling : legit={int((y_train == 0).sum()):,}, "
          f"fraud={int((y_train == 1).sum()):,}")

    # Step 1: random under-sampling of the huge majority class keeps training fast.
    under = RandomUnderSampler(sampling_strategy=UNDERSAMPLE_RATIO, random_state=RANDOM_STATE)
    X_u, y_u = under.fit_resample(X_train, y_train)
    # Step 2: SMOTE creates synthetic minority (fraud) samples by interpolating between real
    # frauds. SMOTE-NC is used because three features are binary (categorical) columns;
    # they are placed last in FEATURE_COLS, so their indices are known.
    cat_idx = list(range(len(CONTINUOUS_COLS), len(FEATURE_COLS)))
    smote = SMOTENC(categorical_features=cat_idx, sampling_strategy=SMOTE_RATIO,
                    k_neighbors=5, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_u, y_u)
    print(f"  after under-sampling           : legit={int((y_u == 0).sum()):,}, "
          f"fraud={int((y_u == 1).sum()):,}")
    print(f"  after SMOTE-NC                 : legit={int((y_res == 0).sum()):,}, "
          f"fraud={int((y_res == 1).sum()):,}")
    print("  (resampling is applied to the TRAINING split only; validation/test stay untouched)")

    start = time.time()
    clf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_res, y_res)
    print(f"  trained in {time.time() - start:.1f}s")

    val_scores = clf.predict_proba(X_val)[:, 1]
    test_scores = clf.predict_proba(X_test)[:, 1]
    name = "SMOTE + Random Forest"
    report_model(name, y_val, val_scores, y_test, test_scores, results, store)
    return clf


# --------------------------------------------------------------------------------------
# 9. Evaluation plots + comparison
# --------------------------------------------------------------------------------------
def plot_evaluation(y_test, store, history, rf_model, results_df, fig_dir):
    banner("9. EVALUATION PLOTS")
    palette = {"Isolation Forest": "#4c72b0", "Autoencoder": "#55a868",
               "SMOTE + Random Forest": "#c44e52"}

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in store.items():
        fpr, tpr, _ = roc_curve(y_test, res["scores"])
        auc = roc_auc_score(y_test, res["scores"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=palette[name], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    save_fig(fig, fig_dir / "eval_01_roc_curves.png")

    # Precision-recall curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in store.items():
        prec, rec, _ = precision_recall_curve(y_test, res["scores"])
        ap = average_precision_score(y_test, res["scores"])
        ax.plot(rec, prec, label=f"{name} (AP = {ap:.4f})", color=palette[name], lw=2)
    ax.axhline(y_test.mean(), color="k", ls="--", lw=1, label="Random guess")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend(loc="upper right")
    save_fig(fig, fig_dir / "eval_02_precision_recall_curves.png")

    # Confusion matrices
    fig, axes = plt.subplots(1, len(store), figsize=(5.2 * len(store), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, store.items()):
        cm = confusion_matrix(y_test, res["pred"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices (test set)")
    save_fig(fig, fig_dir / "eval_03_confusion_matrices.png")

    # Metric comparison
    metrics_to_plot = ["Precision", "Recall", "F1-score", "AUC-ROC", "PR-AUC"]
    plot_df = results_df.set_index("Model")[metrics_to_plot]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.T.plot(kind="bar", ax=ax, color=[palette[m] for m in plot_df.index], width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (test set)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(plot_df.index))
    save_fig(fig, fig_dir / "eval_04_model_comparison.png")

    # Autoencoder training curve
    if history is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(history.history["loss"], label="Training loss", lw=2)
        ax.plot(history.history["val_loss"], label="Validation loss", lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE reconstruction loss")
        ax.set_title("Autoencoder training history")
        ax.legend()
        save_fig(fig, fig_dir / "eval_05_autoencoder_loss.png")

    # Autoencoder reconstruction-error distribution
    ae_scores = store["Autoencoder"]["scores"]
    legit_err = ae_scores[y_test == 0]
    legit_err = np.random.RandomState(RANDOM_STATE).choice(
        legit_err, size=min(len(legit_err), 100_000), replace=False)
    fraud_err = ae_scores[y_test == 1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(np.log10(legit_err + 1e-9), bins=60, stat="density", alpha=0.55,
                 color="#4c72b0", label="Legitimate", ax=ax)
    sns.histplot(np.log10(fraud_err + 1e-9), bins=60, stat="density", alpha=0.55,
                 color="#c44e52", label="Fraud", ax=ax)
    ax.set_xlabel("log10(reconstruction error)")
    ax.set_title("Autoencoder reconstruction error by class (test set)")
    ax.legend()
    save_fig(fig, fig_dir / "eval_06_autoencoder_error_distribution.png")

    # Random Forest feature importance
    imp = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", color="#c44e52", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    save_fig(fig, fig_dir / "eval_07_feature_importance.png")


def print_final_summary(results_df):
    banner("FINAL MODEL COMPARISON (test set)")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results_df.round(6).to_string(index=False))
    best_f1 = results_df.loc[results_df["F1-score"].idxmax(), "Model"]
    best_auc = results_df.loc[results_df["AUC-ROC"].idxmax(), "Model"]
    print(f"\n  Best F1-score : {best_f1}")
    print(f"  Best AUC-ROC  : {best_auc}")
    print("\n  Note: the data is extremely imbalanced, so accuracy (and MAE / MSE / R2 computed "
          "on 0/1\n  predictions) look excellent even for weak models. Rely on F1-score, "
          "Precision/Recall,\n  AUC-ROC and PR-AUC to judge fraud-detection quality.")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Financial fraud detection pipeline")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to the CSV file (default: first CSV inside ./data)")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of the dataset to use, stratified by class (0 < f <= 1)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum autoencoder training epochs (early stopping applies)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Folder for figures and result tables")
    args = parser.parse_args()
    if not 0.0 < args.sample_frac <= 1.0:
        parser.error("--sample-frac must be in the interval (0, 1]")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def main():
    args = parse_args()
    total_start = time.time()
    np.random.seed(RANDOM_STATE)

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load -> 2. Preprocess -> 3. EDA
    raw = load_data(find_dataset(args.data_path), args.sample_frac)
    data, raw_summary = preprocess(raw)
    del raw
    run_eda(data, raw_summary, fig_dir)

    # 4. Split -> 5. Scale
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(data)
    del data
    X_train, X_val, X_test = scale_features(X_train_df, X_val_df, X_test_df)
    del X_train_df, X_val_df, X_test_df

    # 6-8. Models
    results, store = [], {}
    run_isolation_forest(X_train, y_train, X_val, y_val, X_test, y_test, results, store)
    history = run_autoencoder(X_train, y_train, X_val, y_val, X_test, y_test,
                              args.epochs, results, store)
    rf_model = run_smote_classifier(X_train, y_train, X_val, y_val, X_test, y_test,
                                    results, store)

    # 9. Evaluation
    results_df = pd.DataFrame(results)
    results_path = output_dir / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  saved results table -> {results_path}")
    plot_evaluation(y_test, store, history, rf_model, results_df, fig_dir)
    print_final_summary(results_df)

    print(f"\nDone. Total runtime: {(time.time() - total_start) / 60:.1f} minutes")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
