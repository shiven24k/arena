# Your Submission: Writeup Template

*Replace this file's contents when you submit. A good writeup is ~1 page.
We read every one.*

> **Status:** Phase 1 complete. Phase 2 in progress.

---

## Your final score

**Phase 1 (current):** Dev composite score: **0.8187** (from the last line `python grade.py` prints).

**Phase 2 (planned):** Temporal intent tracking across frames — target further BCE reduction.

---

## Your approach, in one paragraph

The baseline's 20 summary features are already reasonable, but they collapse the pedestrian's 16-frame bbox history into aggregates, losing temporal structure. My main change was expanding to 62 features: the original 20, plus the full per-frame velocity sequence (vx/vy for each of the 15 frame intervals = 30 values), plus 12 new summary statistics — speed magnitude, speed-change (deceleration signal), net displacement, bbox size trend (approaching camera), position relative to road center, and a velocity-consistency score. Giving XGBoost the raw velocity sequence lets it learn patterns like "decelerating in the last 3 frames" or "direction flip" that mean-velocity summaries miss. For trajectory I kept CV_4 (mean velocity over the last 4 intervals) unchanged — it is empirically the best constant-velocity variant on this data. For the 7% positive-class imbalance I deliberately did not rebalance, because the task scores calibrated probabilities (BCE); rebalancing distorts the output distribution and reliably hurt dev BCE in testing. The intent classifier is an ensemble of 6 XGBoost models (1 full-data + 5 stochastic with `subsample=0.8`, `colsample_bytree=0.8`, different seeds), blended 30/70 at inference time — the diversity from stochastic subsampling reduces variance by ~0.002 BCE over a single model.

## What you tried that didn't work

**Polynomial and weighted trajectory regression.** I tried fitting degree-2 polynomial trajectory and a weighted least-squares variant that upweights recent frames. Both scored worse than CV_4 on dev — the bbox histories are too short (16 frames ≈ 1 s) and too noisy for polynomial extrapolation to beat a simple mean. Reverted to CV_4.

**Class-weight rebalancing on the intent classifier.** Setting `scale_pos_weight` to account for the 7% positive rate seemed like an obvious fix for the imbalance. In practice it consistently degraded dev BCE by 0.003–0.006 because the model outputs over-confident crossing probabilities that are penalised heavily by log-loss. Calibrated probabilities from an unweighted model are more useful here.

**LightGBM as the base classifier.** Swapped XGBoost for LightGBM with equivalent hyperparameters. It matched XGBoost on a single model but did not combine as cleanly in the ensemble (higher variance across seeds). Kept XGBoost.

## Where AI tooling sped you up most

Claude Code was most useful for feature engineering iteration — I could describe a signal ("deceleration in the last N frames") and immediately get working numpy code with the right shapes, saving the back-and-forth of checking broadcasting bugs. It also helped cross-check the ensemble blending math (30/70 weight derivation) and spotted a subtle NaN propagation path in the trajectory predictor. Where it fell short: it sometimes over-engineered suggestions (e.g. proposing an LSTM on top of the XGBoost), which I had to consciously reject to stay within the CPU-only inference constraint.

## Phase 2 — Planned next experiments

**Temporal intent tracking (primary).** Aggregate the most recent 3–5 `predict()` calls for the same `ped_id` and feed the rising/falling trend of intent probability as an additional feature. A pedestrian whose score is climbing over successive frames is a much stronger crossing signal than any single snapshot. A simple EMA at inference time adds no model complexity.

**Social context features.** Add relative position and velocity between nearby pedestrians — group crossing behaviour is a strong real-world signal not captured by per-pedestrian tracklets.

**Trajectory residual model.** Train a small regressor on (CV_4 prediction − actual position) using context features to correct systematic bias in the trajectory output.

## How to reproduce

```bash
pip install -r requirements.txt
python train.py        # rebuilds model.pkl (~2 min on CPU)
python grade.py        # prints dev composite score
```

## External data / pretrained weights

None. Only `data/train.parquet` and `data/dev.parquet` as provided.

---

_Total time spent on this challenge: ___ hours._
