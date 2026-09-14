# 🛩️ CMAPSS Predictive Maintenance — Jet Engine RUL Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=flat-square&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green?style=flat-square)
![Tests](https://img.shields.io/badge/tests-93%20passing-brightgreen?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-inference%20image-2496ED?style=flat-square&logo=docker)
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

FD001 is the easiest C-MAPSS subset: one operating condition, one fault mode. A well-featurised tree being competitive here does not imply the same on the harder subsets — [all four are measured below](#all-four-subsets), and it holds on every one.

### Label scarcity and augmentation — FD001

Scarcity is simulated by dropping whole engines and keeping the survivors' full run-to-failure histories. Every arm is scored on the same official test split.

| Training data | Engines | Sequences | RMSE ↓ | R² ↑ | NASA ↓ |
|---|---|---|---|---|---|
| 100% engines (reference) | 80 | 14,459 | 14.84 | 0.863 | 440 |
| 50% engines | 40 | 6,890 | 16.00 | 0.841 | 483 |
| 30% engines | 24 | 4,124 | 15.97 | 0.841 | 448 |
| 15% engines | 12 | 2,044 | 20.76 | 0.732 | 843 |
| *30% prefix (flawed design)* | *80* | *2,678* | *38.98* | *0.054* | *8,878* |
| 30% engines + noise σ=0.05 | 24 | 16,496 | 16.49 | 0.831 | 437 |
| 30% engines + noise σ=0.10 | 24 | 16,496 | 16.57 | 0.829 | 478 |
| 30% engines + noise σ=0.25 | 24 | 16,496 | 16.40 | 0.832 | 410 |

Reproduce with `python scripts/run_experiments.py --stage scarcity`.

**About 24 run-to-failure engines are enough on FD001.** Going from 80 engines to 24 costs roughly 1.1 RMSE, and the 50% and 30% arms are indistinguishable from one another. Between 24 and 12 engines the model falls off a cliff: RMSE rises 30% and the NASA score nearly doubles. For an operator deciding how many engines to instrument, that break point is the number worth knowing.

**The flawed design fails for a reason that has nothing to do with data volume.** The prefix arm trained on *all 80* engines and on *more* sequences than the 15% arm (2,678 against 2,044), yet scored 88% worse, at R² = 0.054 — a model that explains essentially nothing, because 93.7% of its labels were the same clipped constant. Its NASA score is 20x the reference. This also confirms where the previously published "limited data" figure of 37.62 RMSE came from: re-running that design measures 38.98.

**Gaussian noise augmentation does not help.** Quadrupling the training set to 16,496 sequences left RMSE slightly *worse* than the un-augmented control at every noise level tested (16.40–16.57 against 15.97), with NASA scores scattered on both sides of it. Jitter perturbs existing degradation trajectories rather than creating new ones, and the model was not short of within-trajectory variation. This is a supportable negative result, which the earlier version could not claim: it had no control arm and added noise at 1% of one standard deviation.

### Transfer learning FD002 → FD001

Pre-trained on FD002 (260 engines, six operating regimes), fine-tuned on 30% of FD001's training engines (24 engines). Each fleet is standardised within its own operating regimes. All arms scored on the same official FD001 test split.

| Arm | RMSE ↓ | MAE ↓ | R² ↑ | NASA ↓ |
|---|---|---|---|---|
| From scratch, 24 engines | 16.71 | 12.51 | 0.826 | 515 |
| Zero-shot — FD002 only, never saw FD001 | 13.33 | 9.34 | 0.889 | 355 |
| **Fine-tuned, LSTM frozen** | **13.07** | **9.20** | **0.894** | **302** |
| Fine-tuned, full network | 13.37 | 9.46 | 0.889 | 327 |

**How much target data transfer saves:**

| FD001 training engines | Pre-trained RMSE ↓ | From scratch RMSE ↓ |
|---|---|---|
| 8 | 13.01 | 23.29 |
| 16 | 13.08 | 19.93 |
| 24 | 13.07 | 16.71 |
| 40 | 13.07 | 16.78 |
| 80 (all) | **12.94** | 14.84 |

Reproduce with `python scripts/run_experiments.py --stage transfer`.

**Transfer is the largest single win in this project.** At 24 target engines it cuts RMSE by 22% and the NASA score by 41% against training from scratch. Where notebook 03 showed that synthetic variation buys nothing, borrowing degradation patterns from another fleet buys a great deal.

**The pre-trained curve is flat.** From 8 engines to 80, RMSE moves only between 12.94 and 13.08 — the amount of FD001 data barely matters once the FD002 representation is in place. Eight target engines with pre-training (13.01) beat eighty without it (14.84). For an operator, that is the difference between instrumenting a fleet and instrumenting a handful of aircraft.

**Almost all of the gain is zero-shot.** The FD002 model scores 13.33 on FD001 without ever seeing a single target engine; fine-tuning adds only 0.26 RMSE on top. The win comes from the source model, not from adaptation — worth stating plainly, because "fine-tuning worked" would be the wrong lesson.

**Freezing beats full fine-tuning** (13.07 vs 13.37), as expected when target data is scarce: fewer trainable parameters, less room to overfit 24 engines.

Two things make this work that the earlier attempt lacked: each fleet is standardised *within its own operating regimes*, so a standardised sensor value means the same thing in both, and frozen versus full fine-tuning is run as a controlled comparison rather than two different setups reported as one. The previously published figure for this experiment was 24.70 RMSE.

**Caveats.** FD002 is both larger (260 engines) and broader (six regimes, likely including conditions resembling FD001's single one), so part of this result is "more data from a superset domain" rather than transfer across a genuine gap — [the controlled FD001 → FD003 experiment below](#cross-transfer) tests exactly that, and finds the distinction decisive. Single seed per arm. And the best LSTM here (12.94) still does not beat the windowed XGBoost baseline (12.38 RMSE, NASA 237) — transfer closes most of the gap to the tree without overturning it.

### All four subsets

<a id="all-four-subsets"></a>

The same two models, re-run on every C-MAPSS subset. Each is normalised by a scaler derived from **its own** regime count rather than one chosen by hand, and each is scored on its own official test files.

NASA scores are shown **per engine** here. The published score is a sum, so FD002's 259 test engines would otherwise look 2.6x worse than FD001's 100 for free.

| Subset | Regimes | Faults | Train engines | XGBoost RMSE ↓ | LSTM RMSE ↓ | XGBoost NASA/engine ↓ | LSTM NASA/engine ↓ |
|---|---|---|---|---|---|---|---|
| FD001 | 1 | 1 | 80 | **12.38** | 17.21 | **2.4** | 7.6 |
| FD002 | 6 | 1 | 208 | **13.45** | 14.70 | **4.0** | 4.6 |
| FD003 | 1 | 2 | 80 | **13.07** | 18.65 | **3.5** | 19.6 |
| FD004 | 6 | 2 | 199 | **15.00** | 16.57 | 5.5 | **5.4** |

Reproduce with `python scripts/run_experiments.py --stage subsets`.

**The tree wins on RMSE on all four subsets, not just the easy one.** The gap is widest exactly where the LSTM was supposed to have an advantage — FD003, two fault modes, where XGBoost scores 13.07 against 18.65 and takes a per-engine NASA score five times lower. The one place the LSTM draws level is FD004, the hardest subset, and only on NASA score (5.4 vs 5.5) while still losing on RMSE.

**A second fault mode costs the LSTM far more than six operating regimes do.** Going from FD001 to FD002 adds five operating conditions and *improves* the LSTM to 14.70 — the extra 128 engines more than pay for the extra complexity, once per-regime normalisation removes the regime from the features. Going from FD001 to FD003 adds a second fault mode at the same fleet size and costs it 1.4 RMSE and 2.6x on NASA score. Operating condition is a nuisance variable that normalisation can remove; a second failure mode is a genuinely different function to learn, and 80 engines split across two of them is a thinner training set than 80 across one.

**Degradation across the benchmark is graceful for the tree.** FD001 → FD004 costs XGBoost 21% RMSE (12.38 → 15.00) while doubling both the regimes and the fault modes. Per-engine NASA score is the harsher read: 2.4 → 5.5, a 2.3x rise, and a reminder that the cost of a *late* prediction grows faster than RMSE suggests.

**Caveat — single seed, and the noise is larger than several gaps in this table.** The FD001 LSTM scores 17.21 here and 14.84 in the baselines table above, on identical data, splits and scaling: with one operating regime `ConditionScaler` reduces to plain standardisation, so the two runs differ only in what the random seed did to the weight initialisation.

That is not a guess. Five seeds of the same architecture on the same FD001 split, [measured below](#seed-variance), span **13.44 to 15.34 RMSE** — 1.91 cycles of pure run-to-run noise. Read every LSTM-to-LSTM comparison across these rows with that band in mind: FD001-versus-FD002 (17.21 against 14.70) is barely outside it, and FD004's NASA-score win over XGBoost sits well inside it. The XGBoost column is unaffected, being deterministic under a fixed seed, and the XGBoost-versus-LSTM gaps on FD001 and FD003 are far larger than the noise.


### Transfer across a controlled gap: FD001 → FD003

<a id="cross-transfer"></a>

The FD002 → FD001 result above is confounded, and the caveat there says so: FD002 is both **larger** (260 engines against 100) and **broader** (six operating regimes, very likely covering FD001's single one), so part of that win is simply more data from a superset domain rather than transfer across a real gap.

FD001 → FD003 removes the confound. Same single operating condition, same fleet size, one extra fault mode — so whatever transfers has to be degradation structure, not coverage. Pre-trained on FD001, LSTM frozen, fine-tuned on a varying share of FD003's training engines, against a from-scratch control on the identical data.

| FD003 training engines | Fine-tuned RMSE ↓ | From scratch RMSE ↓ | Fine-tuned NASA ↓ | From scratch NASA ↓ |
|---|---|---|---|---|
| 0 (zero-shot) | 25.59 | — | 5,860 | — |
| 8 | 20.87 | 21.99 | 2,551 | 2,184 |
| 24 | 20.25 | 18.63 | 2,217 | 1,141 |
| 80 (all) | 19.91 | **12.78** | 2,431 | **317** |

Reproduce with `python scripts/run_experiments.py --stage cross-transfer`.

**Transfer across a fault-mode gap does not work, and past eight engines it actively hurts.** With the full target fleet, training from scratch scores 12.78 against the pre-trained model's 19.91 — 56% worse RMSE and a NASA score 7.7x higher. The gap grows monotonically with target data: −1.1 RMSE at 8 engines, +1.6 at 24, +7.1 at 80. That last number is several times the 1.91 seed-noise floor, so the direction is not in doubt.

**Zero-shot is where it shows most plainly.** An FD001 model applied to FD003 without adaptation scores 25.59 — twice the RMSE of the same architecture trained on FD003 directly, and a NASA score of 5,860. The FD002 → FD001 zero-shot number was 13.33. Same benchmark, same architecture, opposite outcome.

**This reframes the transfer result above rather than contradicting it.** Both are true: borrowing from a *larger, broader* fleet helped a great deal, and borrowing across a genuine domain shift did not. The honest reading of the 22% FD002 → FD001 win is therefore closer to "FD002 contains FD001's operating condition and 160 more engines" than to "degradation patterns transfer between fleets". The caveat there suspected this; this experiment measures it.

**The frozen LSTM is the mechanism.** Freezing the recurrent layer is what made fine-tuning win when target data was scarce — fewer trainable parameters, less room to overfit 24 engines. Here it is exactly the problem: the frozen encoder carries FD001's single failure mode and cannot represent the second one, so extra FD003 data flows into a two-layer head that has nowhere to put it. The fine-tuned curve is nearly flat from 8 engines to 80 (20.87 → 19.91) while the scratch curve drops by 42%. A frozen representation caps what any amount of target data can buy.

**Caveat.** One direction, one seed per arm, and only the frozen variant was swept — full fine-tuning at 80 engines would likely recover most of the gap, since it can overwrite the source representation. What this rules out is the strong claim, that a pre-trained encoder is a free head start; it is not, when the failure physics differ.

### Predictive uncertainty

MC dropout, 100 forward passes, on the 100 test engines:

| | |
|---|---|
| 95% interval empirical coverage | **62%** |
| Mean interval width | 19.0 cycles |
| Mean absolute error | 9.2 cycles |
| Mean absolute error on the widest half of intervals | 11.6 cycles |
| Mean absolute error on the narrowest half | 6.1 cycles |

**The intervals are badly overconfident but genuinely informative.** A nominal 95% band covers 62% of engines, because MC dropout captures model uncertainty and not observation noise — so the bounds must not be handed to a planner as calibrated probabilities. But the width tracks the error: engines with wider intervals are wrong by 11.6 cycles on average against 6.1 for narrow ones. The ranking is usable for triage — which engines deserve a human look — even though the stated confidence level is not.

### Calibrated intervals

Three ways of producing a 95% band, each measured raw and under **split conformal**, on the same 100 test engines. The splits are four-way: the model fits on 64 engines, early stopping watches 20, conformal calibration uses 16 that the model never saw and that never selected it, and the official test files are still touched once.

| Method | Coverage (nominal 95%) | Mean width ↓ | Interval score ↓ | Point RMSE ↓ |
|---|---|---|---|---|
| MC dropout | 63% | 22.7 | 162.8 | 15.18 |
| MC dropout + conformal (adaptive) | 94% | 60.3 | 84.2 | 15.18 |
| MC dropout + conformal (constant) | 94% | 61.8 | 82.8 | 15.18 |
| Deep ensemble, 5 members | **42%** | 15.1 | 214.0 | **13.84** |
| Deep ensemble + conformal (adaptive) | 94% | 68.7 | 90.2 | **13.84** |
| Deep ensemble + conformal (constant) | 95% | 59.0 | 74.1 | **13.84** |
| **Quantile regression (pinball loss)** | **94%** | **54.0** | **67.1** | 14.94 |
| Quantile regression + CQR | 96% | 59.0 | 67.2 | 14.94 |

Reproduce with `python scripts/run_experiments.py --stage calibration`.

Coverage is reported beside width and the **interval score** on purpose. Coverage alone is gameable — a band of `[0, 125]` covers every engine and tells a planner nothing — so the comparison is decided on the interval score, a proper scoring rule that charges for width *and* for how far a missed observation fell outside.

**Confirmed, on a second model: the MC-dropout band is a 63% band wearing a 95% label.** The 62% above came from the transfer-tuned network; this 63% comes from a plain LSTM trained on a different split. Two independent models landing in the same place says the shortfall is a property of the method, not of one unlucky fit. The multiplier a calibrated interval actually needs is not 1.96 but **5.2**. The error is in the dangerous direction: anyone scheduling maintenance to the upper bound is surprised by a third of the fleet.

**The deep ensemble is the most overconfident method tested, not the least.** Five independently seeded networks agree with each other far more than any of them agrees with reality: 42% coverage, against MC dropout's 63%. Averaging them gives the best *point* estimate in the table (RMSE 13.84 against 15.18 for a single network) and simultaneously the worst *uncertainty* estimate. Better point accuracy and better calibration are independent properties, and it is worth saying so plainly — "ensemble the model" is the standard advice for both.

**The simplest method wins.** A network with a 2.5/50/97.5 head trained on the pinball loss lands at 94% coverage untouched, with the best interval score of any arm (67.1) and a band 12% narrower than the best conformalised alternative. It needs one forward pass instead of a hundred. Conformalising it (CQR) moves the offset by only +2.5 cycles — confirmation that it was already close to calibrated rather than a correction.

**Constant-width conformal beats adaptive-width conformal here.** The adaptive score divides by the model's own spread, which preserves its ranking of which engines are uncertain; the constant one throws that ranking away. On this data the ranking is not worth what it costs: a spread that collapses towards zero on some calibration points produces enormous normalised scores, and the multiplier needed to cover 95% of them inflates every interval. The ensemble shows it most clearly — 8.9× std adaptive against a flat 29.5 cycles, for a worse score (90.2 against 74.1).

**What ships.** The service serves the quantile model with its CQR offset, because that is what the table above chose. `/predict` names the method in every response and `/health` reports `interval_calibrated`, so a caller can always tell a guaranteed band from a suggestive one.

### Seed variance

<a id="seed-variance"></a>

The five ensemble members are the same architecture on the same data, differing only in seed. Scoring each one individually costs nothing extra and measures something every LSTM number in this README depends on:

| Seed | RMSE ↓ | MAE ↓ | R² ↑ | NASA ↓ |
|---|---|---|---|---|
| 42 | 15.15 | 11.34 | 0.857 | 468 |
| 43 | 14.49 | 10.51 | 0.869 | 445 |
| 44 | 15.34 | 11.39 | 0.853 | 593 |
| 45 | 14.38 | 10.96 | 0.871 | 391 |
| 46 | **13.44** | **10.14** | **0.888** | **296** |
| *Average of all five* | *13.84* | *10.18* | *0.881* | *364* |

**Changing only the seed moves RMSE by 1.91 cycles and the NASA score by 2x.** The worst seed scores 15.34 and the best 13.44, on identical data. The NASA spread is harsher still — 296 to 593 — because the asymmetric score is dominated by a handful of late predictions, and whether any given engine lands on the late side of the line is close to a coin flip between seeds.

This is the single most important caveat in this README, and it applies retroactively to every LSTM row above. A one-point RMSE difference between two single-seed LSTM arms is not a result. It is also why the 32% XGBoost improvement and the 22% transfer-learning win are worth believing where a 5% difference would not be: both are several times the noise floor.

**Averaging the five seeds beats four of them** (13.84 against a 13.44–15.34 spread) — the usual ensembling result, and the reason the ensemble has the best point estimate in the table above despite having the worst calibrated interval.

**Caveats.** Single seed per arm except the ensemble, which is five by construction. The point RMSE here (15.18 for one network) is worse than the 14.84 in the baselines table because the model fits on 64 engines rather than 80 — 16 were held back for calibration, which is what a conformal guarantee costs. And the calibration windows come from run-to-failure engines truncated at every cycle, which is a valid draw from the deployment distribution but not necessarily the same marginal over truncation points that NASA chose for the test set.



---

## 🗂️ Project Structure

```
cmapss-predictive-maintenance/
│
├── src/                            # the implementation
│   ├── config.py                   # schema, sensor list, RUL cap, seeds, subsets, paths
│   ├── data.py                     # loading, RUL labelling, engine-level splits
│   ├── preprocessing.py            # scaling (incl. per-regime), sequence windows
│   ├── models.py                   # LSTM/XGBoost/quantile builders, MC dropout, persistence
│   ├── calibration.py              # split-conformal interval calibration
│   └── evaluate.py                 # RMSE, MAE, R², NASA score, interval score
│
├── notebooks/                      # experiment logs; they import from src/
│   ├── 01_eda.ipynb                # sensor screening, RUL cap justification
│   ├── 02_baseline_models.ipynb    # XGBoost + LSTM on the official test split
│   ├── 03_limited_data_experiments.ipynb   # scarcity + augmentation ablation
│   └── 04_transfer_learning.ipynb  # FD002 → FD001, plus MC-dropout uncertainty
│                                   # phases 5-6 are runner-only; see scripts/
│
├── api/                            # FastAPI inference service
│   ├── schemas.py                  # request/response validation
│   ├── predictor.py                # model + scaler + calibration, window building
│   └── main.py                     # /health, /predict
│
├── scripts/run_experiments.py      # headless runner for every phase; what CI calls
├── tests/                          # 93 tests, no GPU and no dataset required
├── docs/METHODOLOGY_FIXES.md       # what was wrong, and why it mattered
├── data/{raw,processed}/           # NASA .txt files go in raw/ (gitignored)
├── models/                         # trained artifacts + calibration (gitignored)
├── results/                        # measured experiment outputs (CSV, committed)
│
├── Dockerfile                      # two-stage build of the inference image
├── docker-compose.yml              # local run, artifacts mounted read-only
├── requirements-serve.txt          # serving dependencies only (no notebook stack)
└── requirements.txt                # everything, for running the experiments
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

**Phase 5 — All four subsets.** The two strongest baselines re-run on FD001–FD004, each normalised by a scaler derived from its own regime count rather than chosen by hand, plus a controlled FD001 → FD003 transfer that varies the fault mode while holding operating condition and fleet size fixed.

**Phase 6 — Calibrated intervals.** MC dropout, a five-member deep ensemble and a pinball-loss quantile network, each measured raw and under split conformal, on a four-way split that keeps the calibration engines out of both fitting and model selection.

**Phase 7 — Serving.** `POST /predict` takes a window of readings and returns a point estimate with a 95% interval, says which method produced that interval and whether it carries a coverage guarantee, flags windows that were left-padded because fewer than 30 cycles were supplied, and clamps to the RUL cap. Packaged as a Docker image that mounts its model at runtime rather than baking one in.

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

**Run the experiments** — place the NASA `.txt` files in `data/raw/`, then either read the
notebooks in order:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

or run everything headless, which is what CI and any re-measurement should use — the
notebooks are for reading, this is for running:

```bash
python scripts/run_experiments.py --stage all        # every phase, ~1 hour on CPU
python scripts/run_experiments.py --stage subsets    # or one at a time
python scripts/run_experiments.py --stage all --quick  # smoke-test the wiring
```

Stages: `baselines`, `scarcity`, `transfer`, `subsets`, `cross-transfer`, `calibration`.
Each writes a CSV into `results/`; those CSVs are the source of every number in this
README.

**Serve predictions** — after a stage has persisted a model (`--stage calibration` gives
the service a calibrated interval; `--stage baselines` gives it an uncalibrated one):

```bash
uvicorn api.main:app --reload
```

**Or in Docker** — the image carries `src/` and `api/` and the serving dependencies only;
the trained model is *mounted*, not baked in, so retraining does not mean rebuilding:

```bash
docker compose up --build
# or, without compose:
docker build -t cmapss-rul .
docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" cmapss-rul
```

```bash
curl localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "lstm_fd001_conformal",
  "interval_calibrated": true,
  "sequence_length": 30,
  "expected_sensors": ["sensor_2", "sensor_3", "..."]
}
```

Started with no `models/` mounted the service still comes up and reports
`model_loaded: false`; `/predict` then returns 503 rather than the process refusing to
boot. CI builds the image and asserts exactly that degraded behaviour.

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
  "interval_method": "mc-dropout spread, split-conformal (adaptive)",
  "nominal_coverage": 0.95,
  "cycles_supplied": 30,
  "padded": false,
  "rul_cap": 125
}
```

*(Illustrative response shape — not a measured prediction.)*

`interval_method` is part of the contract, not decoration. The same two fields carry a
conformally calibrated band and a raw MC-dropout one, and those are not the same
product: the first covers 95% of engines, the second 62%. When no calibration is on
disk the service still answers, with `"... (UNCALIBRATED)"` in that field.

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
- [x] Measure and publish the scarcity and augmentation results
- [x] Measure and publish the transfer-learning and uncertainty results
- [x] FD003/FD004 (multi-fault, multi-condition)
- [x] Calibrated intervals (quantile regression, deep ensembles, split conformal)
- [x] Dockerised inference service
- [x] Multi-seed measurement of LSTM run-to-run variance
- [ ] Multi-seed *every* arm, not just FD001, so each table carries its own error bars
- [ ] Per-fault-mode diagnostics on FD003/FD004 — the tree's win there is unexplained, and
      C-MAPSS ships no per-engine fault label, so this needs unsupervised separation first

---

## 👤 Author

**Miloš** — ML/AI Engineer
Building production-grade AI systems with real-world impact.

---

*Dataset: NASA Prognostics Center of Excellence Data Repository*
