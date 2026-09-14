# 🛩️ CMAPSS Predictive Maintenance — Jet Engine RUL Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=flat-square&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green?style=flat-square)
![Tests](https://img.shields.io/badge/tests-42%20passing-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-lightblue?style=flat-square)

---

End-to-end predictive maintenance system for estimating the **Remaining Useful Life (RUL)** of jet engines on the NASA C-MAPSS benchmark. It combines a gradient-boosted tree baseline with an LSTM, and studies the two constraints that actually bite in industry: **labelled run-to-failure data is scarce**, and **a point prediction without an uncertainty band is not actionable**.

> **Why this matters:** unplanned engine removals are among the most expensive events in aviation. A usable RUL model has to say not just *when*, but *how sure* — and has to be trained on the handful of instrumented engines an operator really has, not on a hundred.

---

## 📊 Results — FD001

Measured on the **official test split** (`test_FD001` + `RUL_FD001`), one prediction per engine, n = 100. Training and early stopping used disjoint splits of `train_FD001`; the test files were touched once, for this table.

| Model | RMSE ↓ | MAE ↓ | R² ↑ | NASA score ↓ |
|---|---|---|---|---|
| XGBoost — per-cycle features | 18.19 | 12.56 | 0.79 | 1861.6 |
| **XGBoost — rolling features** | **12.38** | **9.31** | **0.90** | **237.0** |
| LSTM (30-cycle windows) | 14.84 | 11.33 | 0.86 | 440.1 |

Reproduce with `python scripts/run_experiments.py --stage baselines`.

**The tree wins once it is given the same temporal information as the LSTM.** Adding rolling mean/std and drift cuts XGBoost's RMSE by 32% (18.19 → 12.38) and its NASA score by 87%, moving it past the sequence model on both. The earlier claim that the LSTM beat XGBoost by 24% compared it against the per-cycle variant, which structurally cannot represent a trend — that gap measured the input representation, not the model class.

The asymmetric NASA score separates the two models much more sharply than RMSE does (237 vs 440, a 1.9x gap against 1.2x on RMSE), because it charges for *late* predictions — the ones where an engine fails before its scheduled maintenance. On this benchmark that is the number a maintenance planner would optimise.

FD001 is the easiest C-MAPSS subset: one operating condition, one fault mode. A well-featurised tree being competitive here does not imply the same on the multi-condition subsets, which is what Phase 4 probes.

### Still to be measured

Phases 3 and 4 — label scarcity, augmentation and transfer learning — have been rebuilt but not yet re-run. **No numbers are quoted for them** until they are. The previously published figures came from a setup where:

- the reported score was computed on the same split that drove early stopping, and the official `test_FD001` / `RUL_FD001` files were loaded but never used;
- the "limited data" scenario kept the first 30% of each engine's cycles, which after clipping at RUL 125 put **93.7% of retained rows exactly on the cap** and left 74% of engines with no label other than 125 — the model was fitted on a near-constant target and scored across the full range;
- the augmentation experiment had no un-augmented control arm, added noise at 1% of one standard deviation, and built sliding windows across the seam between concatenated copies;
- the transfer-learning run normalised single-condition FD001 with a scaler fitted on six-condition FD002, and compared a fully fine-tuned network against a frozen one as though they were the same run.

Details in [`docs/METHODOLOGY_FIXES.md`](docs/METHODOLOGY_FIXES.md). All four are fixed in code and covered by tests; what remains is the compute.

---

## 🗂️ Project Structure

```
cmapss-predictive-maintenance/
│
├── src/                            # the implementation
│   ├── config.py                   # schema, sensor list, RUL cap, seeds, paths
│   ├── data.py                     # loading, RUL labelling, engine-level splits
│   ├── preprocessing.py            # scaling (incl. per-regime), sequence windows
│   ├── models.py                   # LSTM/XGBoost builders, MC dropout, persistence
│   └── evaluate.py                 # RMSE, MAE, R², NASA score, label diagnostics
│
├── notebooks/                      # experiment logs; they import from src/
│   ├── 01_eda.ipynb                # sensor screening, RUL cap justification
│   ├── 02_baseline_models.ipynb    # XGBoost + LSTM on the official test split
│   ├── 03_limited_data_experiments.ipynb   # scarcity + augmentation ablation
│   └── 04_transfer_learning.ipynb  # FD002 → FD001, plus MC-dropout uncertainty
│
├── api/                            # FastAPI inference service
│   ├── schemas.py                  # request/response validation
│   ├── predictor.py                # model + scaler, window building
│   └── main.py                     # /health, /predict
│
├── tests/                          # 42 tests, no GPU and no dataset required
├── docs/METHODOLOGY_FIXES.md       # what was wrong, and why it mattered
├── data/{raw,processed}/           # NASA .txt files go in raw/ (gitignored)
├── models/  results/               # artifacts and experiment outputs (gitignored)
└── requirements.txt
```

---

## ⚙️ Methodology

### Evaluation protocol

Three disjoint splits, not two:

| split | source | used for |
|---|---|---|
| train | 80% of `train_FD001` engines | fitting |
| validation | 20% of `train_FD001` engines | early stopping, model selection |
| test | `test_FD001` + `RUL_FD001` | reported metrics only |

Scoring on the split that selected the model is optimistic by construction. Scoring on anything but the official test files makes the numbers incomparable to the published literature. The benchmark asks for **one prediction per test engine**, from the window ending at its last recorded cycle.

### Data pipeline

- **RUL construction** — piecewise-linear, capped at **125 cycles**. While an engine is healthy its sensors carry no information about how much life remains; notebook 01 shows sensor–RUL correlation collapsing above the cap. An uncapped label asks the model to predict something it cannot infer.
- **Splitting** — by `engine_id`, with shuffled ids under a fixed seed. Splitting by row would leak: two windows from the same engine a few cycles apart are nearly the same sample.
- **Normalisation** — `StandardScaler` fitted on the training engines only, persisted with the model. For the multi-condition subsets (FD002/FD004), `ConditionScaler` clusters the three operating settings into six regimes and standardises **within** each: the regime shifts the sensors far more than degradation does, so a single global scaler mostly encodes "which regime is this".
- **Windowing** — 30-cycle sliding windows that never cross an engine boundary, labelled with the RUL at the window's **last** cycle. An engine of length *n* yields *n − 30 + 1* windows; dropping the `+ 1` discards the failure window, the single most informative sample each engine has.

### Feature selection

11 sensors, screened on coefficient of variation (scale-free, so sensors with different units compare). The discarded ten are constant or near-constant fleet-wide — testing `nunique() == 1` alone misses the sensors that take two or three distinct values:

`sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_14`, `sensor_15`, `sensor_20`, `sensor_21`

### Models

- **XGBoost** — in two variants. Per-cycle (no history) and windowed (rolling mean/std over 5 and 20 cycles, plus drift from the engine's first reading). Only the second is a fair comparison: a tree fed one cycle at a time structurally cannot represent a trend, so beating it measures the input representation, not the model class.
- **LSTM** — `LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)`, input shape derived from the data rather than hard-coded.

### Metrics

RMSE, MAE and R², plus the benchmark's **asymmetric NASA score**. Predicting *more* remaining life than there is means the engine fails before its scheduled maintenance, which is far worse than an early warning — the score's time constants are 10 against 13. Two models with identical RMSE can be very differently useful, so RMSE alone cannot support a maintenance decision.

---

## 🔬 Experiments

**Phase 1 — EDA.** Sensor screening, RUL distribution, and the empirical case for the cap.

**Phase 2 — Baselines.** XGBoost (both variants) and LSTM on the official test split, with the model and scaler persisted for serving.

**Phase 3 — Label scarcity.** Scarcity is simulated by **dropping whole engines** and keeping the survivors' full run-to-failure histories, so the target distribution still matches deployment and the only thing that varies is data volume. The scaler is refitted on whatever data each scenario actually has — reusing full-fleet statistics leaks information a genuinely data-poor team would not possess. Gaussian jitter is applied to the sequence tensor, swept over realistic noise levels, and measured against an un-augmented control. The flawed prefix-truncation design is kept in the notebook as a worked example rather than deleted.

**Phase 4 — Transfer learning and uncertainty.** FD002 (260 engines, six operating regimes) → FD001 (100 engines, one), each fleet standardised within its own regimes so that "+2σ on sensor_4" means the same thing in both. Four arms — from scratch, zero-shot, frozen LSTM, full fine-tune — then a sweep of target-data volume against a from-scratch control, which is the actual business case: how many instrumented engines transfer saves. Closes with MC-dropout prediction intervals and an empirical coverage check.

**Phase 5 — Serving.** `POST /predict` takes a window of readings and returns a point estimate with a 95% interval, flags windows that were left-padded because fewer than 30 cycles were supplied, and clamps to the RUL cap.

---

## 🚀 Installation & Usage

```bash
git clone https://github.com/MlazicM/cmapss-predictive-maintenance.git
cd cmapss-predictive-maintenance

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements-dev.txt
```

**Run the tests** — they use synthetic data in the CMAPSS wire format, so they need neither the download nor TensorFlow:

```bash
pytest
```

**Run the experiments** — place the NASA `.txt` files in `data/raw/`, then run the notebooks in order:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Serve predictions** — after `02_baseline_models.ipynb` has persisted a model:

```bash
uvicorn api.main:app --reload
```

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "engine_id": 42,
  "readings": [{"cycle": 1, "sensors": {"sensor_2": 642.1, "sensor_3": 1589.7, "...": 0.0}}]
}'
```

```json
{
  "engine_id": 42,
  "predicted_rul": 47.3,
  "lower_95": 31.8,
  "upper_95": 62.8,
  "std": 7.9,
  "cycles_supplied": 30,
  "padded": false,
  "rul_cap": 125
}
```

*(Illustrative response shape — not a measured prediction.)*

---

## 📦 Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** — [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

| Subset | Train engines | Operating conditions | Fault modes |
|---|---|---|---|
| FD001 | 100 | 1 | 1 |
| FD002 | 260 | 6 | 1 |
| FD003 | 100 | 1 | 2 |
| FD004 | 249 | 6 | 2 |

Training engines run to failure. Test engines are truncated before failure, with the remaining life given in the matching `RUL_*.txt` file.

---

## 📈 Roadmap

- [x] EDA and sensor screening
- [x] Reusable `src/` pipeline with test coverage
- [x] Evaluation on the official test split, with the NASA asymmetric score
- [x] Sound scarcity and augmentation designs
- [x] Regime-aware transfer learning (FD002 → FD001)
- [x] MC-dropout uncertainty bounds
- [x] FastAPI inference service
- [x] Measure and publish the FD001 baselines
- [ ] **Re-run the scarcity and transfer phases and publish those results**
- [ ] FD003/FD004 (multi-fault, multi-condition)
- [ ] Calibrated intervals (quantile regression or deep ensembles)
- [ ] Dockerised inference service

---

## 👤 Author

**Miloš** — ML/AI Engineer
Building production-grade AI systems with real-world impact.

---

*Dataset: NASA Prognostics Center of Excellence Data Repository*
