"""Submission entry point.

predict(request) → dict  — do not change the signature.

Improvements vs baseline:
- 62-feature engineering: keeps the 20 baseline features, adds per-frame
  velocity sequence (30 values) and 12 genuinely new summary features.
  Giving the model the full velocity history lets trees learn temporal
  patterns (e.g. "decelerating in the last 3 frames") that summary stats
  miss.
- Trajectory: CV_4 (baseline) is empirically optimal for CPU-only inference;
  all regression/polynomial variants tested worse on dev.
- LightGBM intent classifier (see train.py).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "model.pkl"
HORIZONS_FRAMES = [8, 15, 23, 30]   # at 15 Hz → 0.5, 1.0, 1.5, 2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

_cached_model = None


class EnsembleClassifier:
    """Weighted average of multiple classifier predict_proba outputs.

    Stored in model.pkl so predict() works without changes.
    Weights default to uniform if not set.
    """

    def __init__(self, classifiers: list):
        self.classifiers = classifiers
        self.weights: list | None = None  # set by train.py after construction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.stack([c.predict_proba(X)[:, 1] for c in self.classifiers])
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
            avg = (probs * w[:, None]).sum(axis=0) / w.sum()
        else:
            avg = probs.mean(axis=0)
        return np.column_stack([1.0 - avg, avg])


def _load_model():
    global _cached_model
    if _cached_model is None:
        with open(MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
    return _cached_model


def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _engineered_features(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    vx = np.diff(cx)  # (15,) per-frame x-velocity
    vy = np.diff(cy)  # (15,) per-frame y-velocity

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    # ── block A: original 20 baseline features ──────────────────────────
    feats_baseline = [
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        vx[-4:].mean() / fw,
        vy[-4:].mean() / fh,
        vx.std() / fw,
        vy.std() / fh,
        (h / (w + 1e-6)).mean(),
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(),
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
    ]

    # ── block B: full per-frame velocity sequence (30 features) ─────────
    # Trees can learn temporal patterns (deceleration, direction flip) from
    # the raw sequence that summary statistics miss.
    feats_seq = list(vx / fw) + list(vy / fh)  # 15 + 15 = 30

    # ── block C: 12 genuinely new summary features ───────────────────────
    speed4 = float(np.hypot(vx[-4:].mean(), vy[-4:].mean()))  # speed magnitude
    speed_r2 = float(np.hypot(vx[-2:].mean(), vy[-2:].mean()))
    speed_old = float(np.hypot(vx[-6:-2].mean(), vy[-6:-2].mean()))

    feats_new = [
        # speed magnitude (not captured by vx/vy separately)
        speed4 / max(fw, fh),
        # speed change: negative = decelerating (key crossing signal)
        (speed_r2 - speed_old) / max(fw, fh),
        # net displacement over full 1.07 s window
        (cx[-1] - cx[0]) / fw,
        (cy[-1] - cy[0]) / fh,
        float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])) / max(fw, fh),
        # bbox size trend (pedestrian approaching camera = getting bigger)
        (h[-1] - h[0]) / fh,
        (w[-1] - w[0]) / fw,
        # position relative to horizontal center (approaching road center)
        abs(cx[-1] - fw * 0.5) / (fw * 0.5),
        # moving toward horizontal center? (positive = yes)
        float(np.sign(fw * 0.5 - cx[-1]) * vx[-4:].mean()) / fw,
        # ego deceleration (negative = ego slowing; may yield for crossing ped)
        float((ego_s[-1] - ego_s.mean()) * float(req["ego_available"])),
        # velocity consistency: how erratic is the walk?
        float(1.0 - np.hypot(vx, vy).std() / (np.hypot(vx, vy).mean() + 1e-6)),
        # location flag (only ~6% of data has "street" but it's a real signal)
        1.0 if req.get("location") == "street" else 0.0,
    ]

    all_feats = feats_baseline + feats_seq + feats_new  # 20 + 30 + 12 = 62
    arr = np.asarray(all_feats, dtype=np.float32)
    return arr


def _constant_velocity_trajectory(req: dict) -> dict[str, list[float]]:
    """CV_4: mean velocity over last 4 intervals. Empirically beats all
    regression variants on this dataset (shorter windows add noise)."""
    hist = _as_2d(req["bbox_history"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w_last = hist[-1, 2] - hist[-1, 0]
    h_last = hist[-1, 3] - hist[-1, 1]
    vx = float(np.diff(cx[-5:]).mean())
    vy = float(np.diff(cy[-5:]).mean())
    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])

    out: dict[str, list[float]] = {}
    for h, key in zip(HORIZONS_FRAMES, HORIZON_KEYS):
        nx, ny = cur_cx + vx * h, cur_cy + vy * h
        out[key] = [nx - w_last / 2, ny - h_last / 2, nx + w_last / 2, ny + h_last / 2]
    return out


def predict(request: dict) -> dict:
    intent_model = _load_model()["intent"]
    feats = _engineered_features(request).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    intent_prob = float(intent_model.predict_proba(feats)[0, 1])
    if not np.isfinite(intent_prob):
        intent_prob = 0.5

    out = _constant_velocity_trajectory(request)
    for k in HORIZON_KEYS:
        out[k] = [float(v) if np.isfinite(v) else 0.0 for v in out[k]]
    out["intent"] = intent_prob
    return out
