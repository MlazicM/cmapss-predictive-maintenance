# Methodology Fixes

A record of defects found in the first version of this project, why each one
mattered, and what replaced it. Kept in the repository rather than quietly
rewritten away: the reasoning is more transferable than the code.

Severity is judged by effect on conclusions, not by how many lines changed.

---

## 1. Scarcity simulation truncated the target distribution

**Severity: critical — invalidated Phases 3 and 4.**

```python
# before
def limited_data(df, fraction=0.3):
    ...
    limited.append(engine_data.head(n_cycles))    # first 30% of each engine's cycles
```

The first cycles of an engine are its healthiest. For an engine of lifetime `L`,
the retained rows carry raw RUL in `[0.7L, L]`, and the pipeline then applies
`clip(upper=125)`. So every retained label collapses onto the cap as soon as

```
0.7 * L >= 125    ⟺    L >= 179
```

which covers most of the FD001 fleet (lifetimes run from ~128 to ~362 cycles).
Even the shortest engine contributes no label below ~90. The model was fitted
on an effectively constant target and then evaluated across the full 0–125
range.

That is not data scarcity — it is a **truncated target distribution**. The
resulting degradation said nothing about how much data the model had, so every
conclusion built on it was unsupported, including the headline claim that
transfer learning recovered a third of the lost performance.

**Fix.** `subsample_engines` drops whole engines and keeps the survivors'
complete run-to-failure histories, so the target distribution still matches
deployment and only data *volume* varies. `truncate_engine_prefix` is retained,
documented as unsound, and guarded by
`test_truncate_engine_prefix_collapses_labels` so it cannot be reintroduced by
accident. Notebook 03 shows the collapse rather than hiding it.

**Habit that catches this class of bug in one line:** print
`evaluate.label_distribution(y)` after every transformation of the training
set. `min == max` is invisible in a loss curve but caps everything downstream.

---

## 2. The reported score came from the model-selection split

**Severity: critical — all published metrics were optimistic and incomparable.**

Every notebook used a single `val` split to drive `EarlyStopping(restore_best_weights=True)`
*and* to produce the reported RMSE/R². Selecting on a split and then reporting
on it is optimistic by construction.

Worse, `test_FD001.txt` and `RUL_FD001.txt` were loaded in three notebooks and
never used. The official test files are the only thing that makes a CMAPSS
number comparable to published work, and they were sitting unopened.

**Fix.** Three disjoint splits: train (80% of training engines), validation
(20%, for early stopping), and the official test files, reported once.
`data.load_test` reconstructs per-cycle labels from `RUL_*.txt`, and
`preprocessing.last_sequence_per_engine` produces the one-prediction-per-engine
form the benchmark asks for.

---

## 3. Augmentation windows spanned the seam between copies

**Severity: high — invalidated the augmentation conclusion.**

```python
# before
augmented = [df]
for _ in range(n_copies):
    noisy = df.copy()
    noisy[sensors] = df[sensors] + np.random.normal(0, 0.01, ...)
    augmented.append(noisy)
return pd.concat(augmented, ignore_index=True)   # then windowed afterwards
```

After the concatenation, selecting one engine returns four lifetimes concatenated
end to end. The sliding window then straddles each seam, producing samples whose
sensors show an engine at end of life while the label says it is brand new.
Verified on a toy frame: per-engine RUL order becomes
`[5,4,3,2,1,0, 5,4,3,2,1,0, ...]` and windows are emitted across every
transition.

Two further problems compounded it:

- **The noise was negligible.** `noise_level=0.01` was applied *after*
  `StandardScaler`, so it perturbed inputs by 1% of one standard deviation —
  indistinguishable from no augmentation, which alone could explain a null
  result.
- **There was no control arm.** Notebook 03 trained only the augmented model.
  The 37.62 RMSE that augmentation was said to fail to improve on appears
  nowhere in the code.

**Fix.** `preprocessing.augment_with_noise` jitters the **sequence tensor**,
after windowing, so every sample stays internally consistent. Noise level is
expressed in standard deviations and swept over realistic values, and notebook 03
runs an explicit un-augmented control.

---

## 4. Transfer learning normalised the target fleet with source statistics

**Severity: high — made the transfer result uninterpretable.**

```python
# before
scaler.fit_transform(fd002[sensors])    # FD002: six operating regimes
scaler.transform(train_data[sensors])   # FD001: one regime
```

FD002 is flown under six discrete operating regimes, and the regime moves the
sensor readings far more than degradation does. A single global scaler
therefore encodes mostly "which regime is this", drowning the degradation
signal. Pushing single-regime FD001 through those statistics puts the
fine-tuning and validation data off-distribution before training starts.

**Fix.** `preprocessing.ConditionScaler` clusters the three setting columns into
regimes and standardises within each; each fleet gets its own. After this,
"+2σ on sensor_4" means the same thing in both fleets — deviation from normal
*for the current operating condition* — which is what makes pre-trained weights
transferable at all.

---

## 5. Two fine-tuning strategies reported as one result

**Severity: high — the comparison did not measure what it claimed.**

```python
# before
model.layers[0].trainable = True          # headline run: no-op, already True; lr=1e-4
...
model_test.layers[0].trainable = False    # sweep: frozen LSTM; optimizer='adam' -> lr=1e-3
```

The headline transfer number came from a fully fine-tuned network at lr 1e-4;
the fraction sweep used a frozen LSTM at lr 1e-3. They were presented as one
coherent finding, and `fraction=0.3` in the sweep could not reproduce the
headline number.

Keras also only picks up a `trainable` change at the next `compile`, so
flipping the flag without recompiling is a silent no-op — which is how a
"frozen" and an "unfrozen" run become the same run.

**Fix.** `models.freeze_recurrent_layers` + `models.compile_for_finetuning`,
always used as a pair. Notebook 04 runs frozen and full fine-tuning as an
explicit four-arm comparison against from-scratch and zero-shot controls, then
sweeps target-data volume with a single strategy so the curve measures one
variable.

---

## 6. Sequence construction dropped every engine's failure window

**Severity: medium — systematic loss of the most informative samples.**

```python
# before
for i in range(len(engine_data) - sequence_length):   # missing + 1
```

An engine of length *n* yields *n − L + 1* windows, not *n − L*. Verified: 6
rows with a window of 3 produced 3 windows instead of 4, and the discarded one
is always the last — the lowest-RUL, closest-to-failure sample the engine has.
One such sample lost per engine, in all three notebooks.

The target was also taken at `i + sequence_length`, one cycle *past* the
window, making the task a one-step-ahead forecast rather than "estimate RUL
given history up to now".

**Fix.** `preprocessing.make_sequences` emits *n − L + 1* windows, labels each
with the RUL at its **last** cycle, takes the feature list as an argument
instead of reading a module global, skips engines shorter than the window, and
is vectorised with `sliding_window_view` instead of re-scanning the frame per
engine.

---

## 7. Deployment was structurally blocked

**Severity: medium.**

`api/schemas.py` was a *directory* containing `predictor.py`, and both it and
`api/main.py` were zero bytes. The deeper cause was that no notebook ever
persisted a model or a scaler — `joblib` was declared in `requirements.txt` and
never imported, and `models/` was gitignored and empty. There was nothing to
serve.

**Fix.** `models.save_artifacts` / `load_artifacts` persist the model and its
scaler as one unit — they are inseparable, since serving a model with mismatched
scaling statistics produces confident nonsense with nothing in the output to
signal it. `api/` is now a real package with validation, MC-dropout intervals,
explicit flagging of left-padded windows, and a 503 path when no model exists.

---

## 8. Reproducibility

**Severity: medium.**

Seeds were set only in notebook 04; notebooks 02 and 03 set none, and the
augmentation noise was unseeded — yet results were published to two decimals.
A single `EarlyStopping` instance was also shared across `fit` calls and across
models in a loop; Keras resets it per training run, but sharing a callback that
carries best-weights state between unrelated models is a bug waiting to happen.

**Fix.** `models.set_global_seeds` covers Python, NumPy and TensorFlow and is
called at the top of every notebook; `models.make_early_stopping` returns a
fresh callback per fit; every stochastic helper takes an explicit `seed`.

---

## 9. Smaller items

| Item | Why it mattered |
|---|---|
| `sep='\s+'` | Invalid escape sequence; `SyntaxWarning` on Python 3.12+. Now `r'\s+'`. |
| `LSTM(64, input_shape=(30, 11))` | Hard-coded dimensions desynchronise silently when the sensor list changes. Now derived from `X.shape[1:]`. |
| Unshuffled engine split | `engines[:80]` bakes in whatever order the file happens to have. Now shuffled under a fixed seed. |
| Constant-sensor screen | `nunique() == 1` misses sensors taking two or three values fleet-wide. Now screened on coefficient of variation. |
| XGBoost fed one cycle at a time | A tree with no history cannot represent a trend, so "the LSTM wins" measured the input representation, not the model. Added `add_rolling_features`. |
| RMSE as the only metric | In maintenance, a late prediction is worse than an early one. Added the asymmetric NASA score. |
| `create_sequences` re-scanning the frame per engine | Quadratic on 260-engine FD002. Now `groupby` + strided views. |
| Notebooks as single cells, no `src/` | `create_sequences` existed in three copies, so a fix landed in one and was missed in two — which is exactly what happened with the off-by-one. |
| 15-panel grid for 14 sensors | Empty axis left framed on the figure. Now hidden. |
| Unpinned dependencies, no `jupyter` | `root_mean_squared_error` silently requires scikit-learn ≥ 1.4. Metrics are now NumPy, and requirements carry lower bounds. |

---

## What this cost, and what it bought

The code ran cleanly throughout. Nothing here was a crash or a stack trace —
every defect produced plausible numbers, which is what made them dangerous. The
pipeline fundamentals were sound from the start: splitting by engine, capping
RUL, and asking about data scarcity are all the right instincts.

The lesson is narrower and more useful than "write better code": **a silent
statistical bug is more expensive than a loud one**, so the defence has to be
assertions on distributions, not on exceptions. The `label_distribution` check
and the regression tests in `tests/` exist for exactly that.
