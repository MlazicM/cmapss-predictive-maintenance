# 🛩️ CMAPSS Predictive Maintenance — Jet Engine RUL Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-lightblue?style=flat-square)

---

End-to-end predictive maintenance system for estimating the **Remaining Useful Life (RUL)** of jet engines using the NASA C-MAPSS benchmark dataset. The system combines classical machine learning (XGBoost) with deep learning (LSTM) and explores transfer learning and data-scarce scenarios — directly addressing real-world industrial constraints where labeled sensor data is limited.

> **Why this matters:** Unplanned engine failures cost the aviation industry billions annually. Accurate RUL prediction enables proactive maintenance scheduling, reducing downtime, improving safety, and optimizing operational costs.

---

## 📊 Key Results

| Experiment | Model | RMSE ↓ | R² ↑ |
|---|---|---|---|
| Full data — FD001 | XGBoost | 20.11 | 0.76 |
| Full data — FD001 | LSTM | **15.26** | **0.87** |
| Limited data (30% per engine) | LSTM | 37.62 | 0.19 |
| Transfer learning (FD002 → FD001) | LSTM | 24.70 | 0.65 |

> Transfer learning recovered **36% of the performance lost** under severe data scarcity (37.62 → 24.70 RMSE), outperforming Gaussian noise augmentation which showed negligible improvement.

---

## 🗂️ Project Structure

```
cmapss-predictive-maintenance/
│
├── data/
│   ├── raw/                        # Original NASA CMAPSS .txt files
│   └── processed/                  # Cleaned, scaled, sequenced data
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_baseline_models.ipynb    # XGBoost + LSTM on full data
│   ├── 03_limited_data_experiments.ipynb  # Data scarcity + augmentation
│   └── 04_transfer_learning.ipynb  # Cross-dataset transfer (FD002 → FD001)
│
├── src/
│   ├── preprocessing.py            # Feature engineering, scaling, sequencing
│   ├── models.py                   # Model definitions
│   └── evaluate.py                 # Metrics and evaluation utilities
│
├── models/                         # Saved model weights (.keras, .json)
├── results/                        # Plots, metrics, experiment logs
├── requirements.txt
└── README.md
```

---

## ⚙️ Methodology

### Data Pipeline
- **RUL Construction:** Piecewise linear capping at **125 cycles** — prevents the model from being misled by artificially high RUL labels in early engine life where degradation hasn't begun
- **Train/Validation Split:** Split by `engine_id`, never by random row — essential for time-series integrity; random splitting would cause data leakage across engine lifecycles
- **Normalization:** `StandardScaler` fitted exclusively on the training set, then applied to validation/test — no information from the future leaks into training statistics

### Feature Selection
11 informative sensors selected based on variance analysis during EDA:
`sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_14`, `sensor_15`, `sensor_20`, `sensor_21`

Sensors with near-zero variance across all engine cycles were discarded.

### Models
- **XGBoost:** Trained on flattened sensor features per cycle; strong baseline
- **LSTM:** 30-cycle sliding window sequences → `LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)`; captures temporal degradation patterns that XGBoost cannot

### Experiments
- **Gaussian Noise Augmentation:** Applied under 30% data per engine; negligible performance recovery — confirms that noise augmentation alone is insufficient for severe data scarcity
- **Transfer Learning:** Pre-trained on FD002 (multi-condition), fine-tuned on FD001 — meaningful generalization across different operational conditions

---

## 🔬 Experiments Overview

### Phase 1 — Exploratory Data Analysis
Sensor variance analysis, RUL distribution inspection, correlation heatmaps. Identified which of the 21 sensors carry meaningful degradation signal and established the piecewise RUL capping strategy.

### Phase 2 — Baseline Modeling
Established XGBoost (RMSE 20.11) and LSTM (RMSE 15.26) baselines on the full FD001 dataset. LSTM's ability to model temporal sequences yielded a 24% improvement over XGBoost.

### Phase 3 — Limited Data & Augmentation
Simulated data-scarce conditions by restricting training to 30% of cycles per engine. RMSE degraded to 37.62. Gaussian noise augmentation showed minimal recovery. Transfer learning (FD002 → FD001) recovered performance to RMSE 24.70, demonstrating the superiority of domain transfer over synthetic augmentation.

### Phase 4 — Deployment *(in progress)*
REST API for real-time RUL inference. Input: raw sensor readings. Output: predicted RUL in cycles with confidence bounds.

---

## 🚀 Installation & Usage

```bash
git clone https://github.com/yourusername/cmapss-predictive-maintenance.git
cd cmapss-predictive-maintenance

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Place the NASA CMAPSS `.txt` files in `data/raw/`, then run notebooks in order:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📦 Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
[Download from NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

| Subset | Train Engines | Operating Conditions | Fault Modes |
|---|---|---|---|
| FD001 | 100 | 1 | 1 |
| FD002 | 260 | 6 | 1 |
| FD003 | 100 | 1 | 2 |

Each engine run begins at healthy state and ends at failure. Sensor readings are recorded at each cycle.

---

## 🛠️ Technologies

`Python 3.11` · `NumPy` · `Pandas` · `Scikit-learn` · `TensorFlow / Keras` · `XGBoost` · `Matplotlib` · `Seaborn`

---

## 📈 Roadmap

- [x] EDA and sensor selection
- [x] XGBoost and LSTM baselines
- [x] Limited data experiments
- [x] Transfer learning (FD002 → FD001)
- [ ] FD003 multi-fault incorporation
- [ ] REST API deployment (FastAPI)
- [ ] Dockerized inference service
- [ ] CI/CD pipeline

---

## 👤 Author

**Miloš** — ML/AI Engineer  
Building production-grade AI systems with real-world impact.

---

*Dataset: NASA Prognostics Center of Excellence Data Repository*
