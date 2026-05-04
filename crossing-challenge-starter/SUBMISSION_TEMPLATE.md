# Your Submission: Writeup

---

## Your final score

Dev composite score: **0.8187**

(Baseline was around ~0.83, so small but consistent improvement)

---

## Your approach, in one paragraph

The baseline compresses the 16-frame bbox history into summary features, which loses a lot of useful motion information. Instead of relying only on aggregates, I expanded the feature set from 20 → 62 features. I included the full velocity sequence (vx/vy across all frame intervals) along with some additional motion-based signals like deceleration, displacement, bbox size trend (to capture approach), and velocity consistency. This helps the model pick up short-term patterns like slowing down or direction change that are important for crossing intent. For intent prediction, I used an ensemble of 6 XGBoost models (one trained on full data and five with stochastic subsampling), combined with a 30/70 weighting to reduce variance. For trajectory, I kept the CV_4 method (mean velocity of last 4 steps), since it consistently worked better than more complex alternatives.

---

## What you tried that didn't work

**Polynomial / weighted trajectory models**
I tried quadratic fitting and giving more weight to recent frames. Both made things worse — the data is too short and noisy, and simple constant velocity worked better.

**Class rebalancing**
Since positives are only ~8%, I tried `scale_pos_weight`. It actually hurt BCE because predictions became overconfident. Since evaluation uses log-loss, calibration matters more than recall here.

**GRU-based sequence model**
I built a GRU model to directly learn temporal behavior. During training, it looked very strong (dev score went down to ~0.33), mainly because trajectory improved a lot. But the intent head collapsed and started predicting mostly zeros due to class imbalance. Also, reproducing the exact feature pipeline at inference (after ONNX export) turned out to be tricky and led to mismatches in evaluation. Given the time constraint, I decided not to submit this version and stick with a stable approach.

---

## Where AI tooling sped you up most

AI tools helped a lot with feature engineering. I could describe an idea like “detect deceleration trend” and quickly get working NumPy code without spending time debugging shapes. It also helped validate ensemble logic and catch small issues like NaN propagation. One downside was that it sometimes suggested overly complex solutions (like jumping straight to deep learning), which I had to ignore to stay within constraints.

---

## Next experiments

If I had more time, I would:

* Fix the GRU model using class-weighted loss or focal loss to handle imbalance
* Add a small residual model on top of CV_4 for trajectory correction
* Use temporal smoothing (EMA) of intent predictions across frames
* Add interaction features between nearby pedestrians

---

## How to reproduce

```bash
pip install -r requirements.txt
python train.py
python grade.py
```

---

## External data / pretrained weights

None. Only the provided dataset was used.

---

*Total time spent on this challenge: ~12 hours.*
