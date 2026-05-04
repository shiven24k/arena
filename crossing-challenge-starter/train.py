"""Training script — produces model.pkl.

Phase 2: Joint GRU that predicts intent + trajectory from the raw 16-frame
bbox/ego sequence. Trains with PyTorch (GPU recommended), exports to ONNX so
inference runs via onnxruntime with no torch dependency in Docker.

Quick start on Google Colab / Kaggle (free GPU)
------------------------------------------------
  # 1. Upload the repo (or clone it)
  # 2. !pip install torch onnx onnxruntime pandas pyarrow scikit-learn
  # 3. !python train.py
  # 4. Download model.pkl and commit it back to the repo.

Architecture
------------
  Input  : seq (B,16,8) per-frame features  +  ctx (B,6) static context
  Encoder: GRU(input=8, hidden=128, layers=2, dropout=0.2)
  Trunk  : Linear(134,64) → ReLU → Dropout(0.1)
  Heads  : intent  Linear(64,1)  — BCE loss
           traj    Linear(64,8)  — 4 horizons × (dx_norm, dy_norm), Huber loss

Trajectory targets are normalized bbox-center offsets:
  dx_norm = (future_cx − cur_cx) / frame_w
  dy_norm = (future_cy − cur_cy) / frame_h
predict.py reverses this to pixel coords at inference time.

Phase 1 baseline (for reference):
  python -c "
  import pickle, pandas as pd, numpy as np
  from sklearn.metrics import log_loss
  from predict import _engineered_features
  dev = pd.read_parquet('data/dev.parquet')
  with open('model.pkl','rb') as f: m = pickle.load(f)
  # ... XGBoost ensemble path
  "
"""

from __future__ import annotations

import io
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# torch / onnx are GPU-training dependencies — not needed in Docker.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import onnx  # noqa: F401 — just to verify it's installed
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: onnx not installed — will save PyTorch model instead. "
          "pip install onnx for ONNX export.", file=sys.stderr)

from predict import _featurize_gru, _as_2d

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

# ── hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN = 128
GRU_LAYERS = 2
DROPOUT = 0.2
BATCH = 256
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
# Trajectory loss weight. Normalized offsets ≈ 0.02, so raw Huber ≈ 2e-4
# vs BCE ≈ 0.16 → scale up to balance gradients.
TRAJ_WEIGHT = 50.0
HUBER_DELTA = 0.05   # ~5% of frame dimension → ~96px at 1920px wide

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


# ── model ─────────────────────────────────────────────────────────────────────
class PedCrossingGRU(nn.Module):
    """Joint intent + trajectory predictor over pedestrian tracklet sequence."""
    SEQ_FEAT = 8
    CTX_FEAT = 6

    def __init__(self, hidden: int = HIDDEN, layers: int = GRU_LAYERS,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.gru = nn.GRU(
            self.SEQ_FEAT, hidden, layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden + self.CTX_FEAT, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.intent_head = nn.Linear(64, 1)
        self.traj_head = nn.Linear(64, 8)   # 4 × (dx_norm, dy_norm)

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor):
        # seq: (B, 16, 8)  ctx: (B, 6)
        _, h_n = self.gru(seq)                         # h_n: (layers, B, hidden)
        x = self.trunk(torch.cat([h_n[-1], ctx], dim=1))
        return self.intent_head(x).squeeze(1), self.traj_head(x)
        # → intent_logit (B,), traj_offsets (B, 8)


# ── dataset ───────────────────────────────────────────────────────────────────
def _row_traj_target(row: pd.Series) -> np.ndarray:
    """Normalized center offsets (8,) for 4 future horizons."""
    hist = np.asarray(row["bbox_history"], dtype=np.float64)
    fw, fh = float(row["frame_w"]), float(row["frame_h"])
    cur_cx = (hist[-1, 0] + hist[-1, 2]) * 0.5
    cur_cy = (hist[-1, 1] + hist[-1, 3]) * 0.5

    out = []
    for h in HORIZONS:
        fb = np.asarray(row[h], dtype=np.float64)
        fcx = (fb[0] + fb[2]) * 0.5
        fcy = (fb[1] + fb[3]) * 0.5
        out.extend([(fcx - cur_cx) / fw, (fcy - cur_cy) / fh])
    return np.array(out, dtype=np.float32)


class PedDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        n = len(df)
        print(f"  Featurizing {n:,} rows … ", end="", flush=True)
        t0 = time.time()

        self.seq = np.zeros((n, 16, PedCrossingGRU.SEQ_FEAT), dtype=np.float32)
        self.ctx = np.zeros((n, PedCrossingGRU.CTX_FEAT), dtype=np.float32)
        self.intent = np.zeros(n, dtype=np.float32)
        self.traj = np.zeros((n, 8), dtype=np.float32)

        # Store fw/fh for pixel-space ADE evaluation
        self.fw = df["frame_w"].to_numpy(dtype=np.float64)
        self.fh = df["frame_h"].to_numpy(dtype=np.float64)

        req_cols = [c for c in REQUEST_FIELDS if c in df.columns]
        for i, (_, row) in enumerate(df[req_cols + HORIZONS].iterrows()):
            req = {k: row[k] for k in req_cols}
            seq_i, ctx_i = _featurize_gru(req)
            self.seq[i] = seq_i[0]
            self.ctx[i] = ctx_i[0]
            self.intent[i] = float(row["will_cross_2s"]) if "will_cross_2s" in row else 0.0
            self.traj[i] = _row_traj_target(row)

        print(f"done ({time.time()-t0:.1f}s)")

    def __len__(self) -> int:
        return len(self.intent)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.seq[i]),
            torch.from_numpy(self.ctx[i]),
            torch.tensor(self.intent[i]),
            torch.from_numpy(self.traj[i]),
        )


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(model: PedCrossingGRU, loader: DataLoader,
             fw: np.ndarray, fh: np.ndarray) -> dict:
    model.eval()
    all_intent_p, all_intent_gt, all_traj_p, all_traj_gt = [], [], [], []

    with torch.no_grad():
        for seq, ctx, intent, traj in loader:
            seq, ctx = seq.to(DEVICE), ctx.to(DEVICE)
            logit, traj_p = model(seq, ctx)
            all_intent_p.append(torch.sigmoid(logit).cpu().numpy())
            all_intent_gt.append(intent.numpy())
            all_traj_p.append(traj_p.cpu().numpy())
            all_traj_gt.append(traj.numpy())

    intent_prob = np.clip(np.concatenate(all_intent_p), 1e-6, 1 - 1e-6)
    intent_gt = np.concatenate(all_intent_gt)
    traj_p = np.concatenate(all_traj_p)     # (N, 8)
    traj_gt = np.concatenate(all_traj_gt)   # (N, 8)

    bce = log_loss(intent_gt, intent_prob)

    ades = []
    for hi in range(4):
        dx_px = (traj_p[:, hi * 2] - traj_gt[:, hi * 2]) * fw
        dy_px = (traj_p[:, hi * 2 + 1] - traj_gt[:, hi * 2 + 1]) * fh
        ades.append(float(np.hypot(dx_px, dy_px).mean()))
    mean_ade = float(np.mean(ades))

    BCE_FLOOR, ADE_FLOOR = 0.2488, 49.80
    composite = 0.5 * (bce / BCE_FLOOR) + 0.5 * (mean_ade / ADE_FLOOR)
    return {
        "composite": composite,
        "intent_term": bce / BCE_FLOOR,
        "traj_term": mean_ade / ADE_FLOOR,
        "bce": bce,
        "ade_px": mean_ade,
        "ade_per_horizon": ades,
    }


# ── ONNX export ───────────────────────────────────────────────────────────────
def export_onnx(model: PedCrossingGRU) -> bytes:
    model.eval().cpu()
    dummy_seq = torch.randn(1, 16, PedCrossingGRU.SEQ_FEAT)
    dummy_ctx = torch.randn(1, PedCrossingGRU.CTX_FEAT)
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        (dummy_seq, dummy_ctx),
        buf,
        input_names=["seq", "ctx"],
        output_names=["intent_logit", "traj_offsets"],
        dynamic_axes={
            "seq": {0: "batch"},
            "ctx": {0: "batch"},
            "intent_logit": {0: "batch"},
            "traj_offsets": {0: "batch"},
        },
        opset_version=17,
    )
    return buf.getvalue()


# ── training loop ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device: {DEVICE}")

    print("\nLoading data …")
    train_df = pd.read_parquet(DATA / "train.parquet")
    dev_df = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train {len(train_df):,}   dev {len(dev_df):,}")
    print(f"  positive rate — train {train_df.will_cross_2s.mean():.3%}  "
          f"dev {dev_df.will_cross_2s.mean():.3%}")

    print("\nBuilding datasets …")
    train_ds = PedDataset(train_df)
    dev_ds = PedDataset(dev_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))
    dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False,
                            num_workers=2, pin_memory=(DEVICE == "cuda"))

    model = PedCrossingGRU().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, verbose=True,
    )

    print(f"Training {EPOCHS} epochs  "
          f"(TRAJ_WEIGHT={TRAJ_WEIGHT}, HUBER_DELTA={HUBER_DELTA})\n")

    best_composite = float("inf")
    best_state: dict | None = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        sum_loss = sum_bce = sum_traj = 0.0

        for seq, ctx, intent, traj in train_loader:
            seq, ctx = seq.to(DEVICE), ctx.to(DEVICE)
            intent = intent.to(DEVICE)
            traj = traj.to(DEVICE)

            logit, traj_pred = model(seq, ctx)
            bce_loss = F.binary_cross_entropy_with_logits(logit, intent)
            traj_loss = F.huber_loss(traj_pred, traj, delta=HUBER_DELTA)
            loss = bce_loss + TRAJ_WEIGHT * traj_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sum_loss += float(loss)
            sum_bce += float(bce_loss)
            sum_traj += float(traj_loss)

        nb = len(train_loader)
        m = evaluate(model, dev_loader, dev_ds.fw, dev_ds.fh)
        scheduler.step(m["composite"])

        tag = ""
        if m["composite"] < best_composite:
            best_composite = m["composite"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " ← best"

        ade_str = "  ".join(f"{a:.1f}" for a in m["ade_per_horizon"])
        print(
            f"Ep {epoch:3d}/{EPOCHS}  "
            f"L {sum_loss/nb:.4f} (bce {sum_bce/nb:.4f} traj {sum_traj/nb:.5f})  "
            f"| dev {m['composite']:.4f}  "
            f"intent {m['intent_term']:.3f}  traj {m['traj_term']:.3f}  "
            f"ADE [{ade_str}] px"
            f"{tag}  ({time.time()-t0:.1f}s)"
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval().cpu()

    print(f"\nBest dev composite: {best_composite:.4f}")

    if ONNX_AVAILABLE:
        print("Exporting to ONNX …")
        onnx_bytes = export_onnx(model)
        payload: dict = {"type": "gru_onnx", "onnx_bytes": onnx_bytes}
        print(f"  ONNX size: {len(onnx_bytes)/1024:.1f} KB")
    else:
        print("Saving PyTorch model (onnxruntime inference will not work; "
              "install onnx and re-run to export ONNX).")
        payload = {"type": "gru_torch", "model": model}

    print(f"Saving model.pkl → {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print("Done. Run  python grade.py  to verify.")


if __name__ == "__main__":
    main()
    sys.exit(0)
