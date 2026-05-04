"""Training script — produces model.pkl.

Run:
    python train.py

Improvements vs baseline.py:
- 62 features (vs 20): adds per-frame velocity sequence (vx/vy for each of the
  15 intervals) so trees can learn temporal patterns like deceleration or
  direction flip; also adds 12 new summary features (speed magnitude, net
  displacement, bbox size trend, etc.).
- Ensemble of 6 XGBoost models: 1 full-data model + 5 stochastic models
  (subsample=0.8, colsample_bytree=0.8, different seeds). Averaging reduces
  variance and improves dev BCE by ~0.002 over a single model.
- Trajectory: CV_4 kept unchanged — all regression variants tested worse.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from predict import EnsembleClassifier, _engineered_features

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def row_to_request(row: pd.Series) -> dict:
    return {k: row[k] for k in REQUEST_FIELDS}


def featurize(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    sample = _engineered_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    for i in range(n):
        X[i] = _engineered_features(row_to_request(df.iloc[i]))
    return X


def train_xgb(X_tr, y_tr, X_dev, y_dev, subsample=1.0, colsample=1.0, seed=42):
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=subsample,
        colsample_bytree=colsample,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], verbose=False)
    return clf


def main() -> None:
    print("Loading train + dev...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")
    pos_tr = train.will_cross_2s.mean()
    print(f"  positive rates — train: {pos_tr:.3%}  dev: {dev.will_cross_2s.mean():.3%}")

    print("\nFeaturizing (62 features)...")
    t0 = time.time()
    X_train = featurize(train)
    X_dev = featurize(dev)
    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  {time.time() - t0:.1f}s  shape: {X_train.shape}")

    print("\nTraining ensemble (1 full + 5 stochastic XGBoost)...")
    clfs = []

    # Model 1: full data (no subsampling — different bias from stochastic models)
    t0 = time.time()
    clf_full = train_xgb(X_train, y_train, X_dev, y_dev)
    p_full = clf_full.predict_proba(X_dev)[:, 1]
    ll_full = log_loss(y_dev, np.clip(p_full, 1e-6, 1 - 1e-6))
    print(f"  full model  BCE={ll_full:.4f}  ({time.time()-t0:.1f}s)")
    clfs.append(clf_full)

    # Models 2-6: stochastic with different seeds
    probs_stoch = []
    for seed in [42, 123, 456, 789, 1234]:
        t0 = time.time()
        clf = train_xgb(X_train, y_train, X_dev, y_dev, subsample=0.8, colsample=0.8, seed=seed)
        p = clf.predict_proba(X_dev)[:, 1]
        ll = log_loss(y_dev, np.clip(p, 1e-6, 1 - 1e-6))
        print(f"  seed={seed}      BCE={ll:.4f}  ({time.time()-t0:.1f}s)")
        clfs.append(clf)
        probs_stoch.append(p)

    # Ensemble: blend full (30%) + average stochastic (70%)
    p_blend = 0.3 * p_full + 0.7 * np.stack(probs_stoch).mean(axis=0)
    ll_ens = log_loss(y_dev, np.clip(p_blend, 1e-6, 1 - 1e-6))
    print(f"\n  Ensemble BCE:    {ll_ens:.4f}  intent_term={ll_ens/0.2488:.3f}")
    prior_ll = log_loss(y_dev, np.full(len(y_dev), pos_tr))
    print(f"  Class-prior BCE: {prior_ll:.4f}")

    # Build ensemble classifier that applies the 30/70 blend at predict time
    # Store all 6 models with weights [0.3, 0.14, 0.14, 0.14, 0.14, 0.14]
    weights = [0.3] + [0.7 / 5] * 5
    ensemble = EnsembleClassifier(clfs)
    ensemble.weights = weights  # EnsembleClassifier will use these if present

    print(f"\nSaving model → {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"intent": ensemble}, f)
    print("Done.")


if __name__ == "__main__":
    main()
    sys.exit(0)
