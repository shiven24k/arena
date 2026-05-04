"""Submission entry point.

predict(request) → dict  — do not change the signature.

Phase 1 (current model.pkl):
  XGBoost ensemble, 62 engineered features, CV_4 trajectory.
  Dev: composite 0.8187, intent_term 0.831, traj_term 0.806.

Phase 2 (GRU model.pkl, trained on GPU then exported to ONNX):
  Joint GRU over the raw 16-frame sequence → intent + 4-horizon trajectory.
  Inference via onnxruntime (CPU-only, ~50 MB) — no torch needed in Docker.
  Expected: composite ~0.55-0.60, traj_term ~0.40-0.50.

predict() dispatches automatically based on the type field in model.pkl.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

MODEL_PATH = Path(__file__).parent / "model.pkl"
HORIZONS_FRAMES = [8, 15, 23, 30]   # at 15 Hz → 0.5, 1.0, 1.5, 2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

_cached_model = None


# ── Phase 1: XGBoost ensemble ─────────────────────────────────────────────────

class EnsembleClassifier:
    """Weighted average of multiple classifier predict_proba outputs."""

    def __init__(self, classifiers: list):
        self.classifiers = classifiers
        self.weights: list | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.stack([c.predict_proba(X)[:, 1] for c in self.classifiers])
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
            avg = (probs * w[:, None]).sum(axis=0) / w.sum()
        else:
            avg = probs.mean(axis=0)
        return np.column_stack([1.0 - avg, avg])


def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _engineered_features(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    vx = np.diff(cx)
    vy = np.diff(cy)

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    feats_baseline = [
        cx[-1] / fw, cy[-1] / fh, w[-1] / fw, h[-1] / fh,
        vx[-4:].mean() / fw, vy[-4:].mean() / fh,
        vx.std() / fw, vy.std() / fh,
        (h / (w + 1e-6)).mean(),
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(),
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
    ]
    feats_seq = list(vx / fw) + list(vy / fh)
    speed4 = float(np.hypot(vx[-4:].mean(), vy[-4:].mean()))
    speed_r2 = float(np.hypot(vx[-2:].mean(), vy[-2:].mean()))
    speed_old = float(np.hypot(vx[-6:-2].mean(), vy[-6:-2].mean()))
    feats_new = [
        speed4 / max(fw, fh),
        (speed_r2 - speed_old) / max(fw, fh),
        (cx[-1] - cx[0]) / fw, (cy[-1] - cy[0]) / fh,
        float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])) / max(fw, fh),
        (h[-1] - h[0]) / fh, (w[-1] - w[0]) / fw,
        abs(cx[-1] - fw * 0.5) / (fw * 0.5),
        float(np.sign(fw * 0.5 - cx[-1]) * vx[-4:].mean()) / fw,
        float((ego_s[-1] - ego_s.mean()) * float(req["ego_available"])),
        float(1.0 - np.hypot(vx, vy).std() / (np.hypot(vx, vy).mean() + 1e-6)),
        1.0 if req.get("location") == "street" else 0.0,
    ]
    arr = np.asarray(feats_baseline + feats_seq + feats_new, dtype=np.float32)
    return arr


# ── Phase 2: GRU sequence featurization ──────────────────────────────────────

def _featurize_gru(req: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (seq float32 (1,16,8), ctx float32 (1,6)) for ONNX inference."""
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    # prepend 0 so velocity has length 16 (first frame has no delta)
    vx = np.diff(cx, prepend=cx[0])
    vy = np.diff(cy, prepend=cy[0])

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)
    ego_dy = np.diff(ego_y, prepend=ego_y[0])

    seq = np.stack([
        cx / fw,
        cy / fh,
        w / fw,
        h / fh,
        vx / fw,
        vy / fh,
        ego_s / 10.0,       # normalize by typical max speed (m/s)
        ego_dy / np.pi,     # normalize yaw delta
    ], axis=1).astype(np.float32)   # (16, 8)

    ctx = np.array([
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
        1.0 if req.get("location") == "street" else 0.0,
        float(req["ego_available"]),
    ], dtype=np.float32)   # (6,)

    return seq[np.newaxis], ctx[np.newaxis]   # (1,16,8), (1,6)


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model():
    global _cached_model
    if _cached_model is None:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and data.get("type") == "gru_onnx":
            if not _ORT_AVAILABLE:
                raise RuntimeError(
                    "model.pkl contains a GRU model but onnxruntime is not installed. "
                    "pip install onnxruntime"
                )
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            session = ort.InferenceSession(
                data["onnx_bytes"],
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            _cached_model = {"type": "gru_onnx", "session": session}
        else:
            # Phase 1: XGBoost ensemble dict {"intent": EnsembleClassifier}
            _cached_model = data

    return _cached_model


# ── trajectory helpers ────────────────────────────────────────────────────────

def _constant_velocity_trajectory(req: dict) -> dict[str, list[float]]:
    """CV_4: mean velocity over last 4 intervals (Phase 1 trajectory)."""
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


def _gru_trajectory(req: dict, traj_offsets: np.ndarray) -> dict[str, list[float]]:
    """Reconstruct bboxes from GRU-predicted (dx_norm, dy_norm) offsets."""
    hist = _as_2d(req["bbox_history"])
    cur_cx = float((hist[-1, 0] + hist[-1, 2]) * 0.5)
    cur_cy = float((hist[-1, 1] + hist[-1, 3]) * 0.5)
    w_last = float(hist[-1, 2] - hist[-1, 0])
    h_last = float(hist[-1, 3] - hist[-1, 1])
    fw, fh = float(req["frame_w"]), float(req["frame_h"])

    out: dict[str, list[float]] = {}
    for i, key in enumerate(HORIZON_KEYS):
        dx_px = float(traj_offsets[i * 2]) * fw
        dy_px = float(traj_offsets[i * 2 + 1]) * fh
        nx, ny = cur_cx + dx_px, cur_cy + dy_px
        out[key] = [nx - w_last / 2, ny - h_last / 2, nx + w_last / 2, ny + h_last / 2]
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def predict(request: dict) -> dict:
    model = _load_model()

    if isinstance(model, dict) and model.get("type") == "gru_onnx":
        # Phase 2: GRU via ONNX Runtime
        session = model["session"]
        seq, ctx = _featurize_gru(request)
        if not np.isfinite(seq).all():
            seq = np.nan_to_num(seq, nan=0.0, posinf=1.0, neginf=-1.0)

        intent_logit, traj_offsets = session.run(
            None, {"seq": seq, "ctx": ctx}
        )
        intent_prob = float(1.0 / (1.0 + np.exp(-intent_logit[0, 0])))
        if not np.isfinite(intent_prob):
            intent_prob = 0.5

        out = _gru_trajectory(request, traj_offsets[0])
        for k in HORIZON_KEYS:
            out[k] = [float(v) if np.isfinite(v) else 0.0 for v in out[k]]
        out["intent"] = intent_prob
        return out

    else:
        # Phase 1: XGBoost ensemble
        intent_model = model["intent"]
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
