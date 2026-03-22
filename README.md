<div align="center">

# NextStopAI 🛫

**Two-Stage Deep Learning Recommendation System for Transit Route Prediction**

*MSc Thesis · Data Science & Artificial Intelligence · Lima, Perú 🇵🇪*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-LambdaRank-2980B9?style=flat-square)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Databricks](https://img.shields.io/badge/Databricks-Delta_Lake-FF3621?style=flat-square&logo=databricks&logoColor=white)](https://databricks.com)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-6236FF?style=flat-square)](https://optuna.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.x-E25A1C?style=flat-square&logo=apachespark&logoColor=white)](https://spark.apache.org)

</div>

---

## Overview

**NextStopAI** is a sequential recommendation system that predicts the next flight route a B2C customer is most likely to purchase. The system is built as a full end-to-end MLOps pipeline on Databricks, moving data from raw Bronze tables through a rigorous curation process, training a two-stage deep learning model, and registering it for serving.

| Stage | Notebook | Output |
|---|---|---|
| 1. Data Curation | `01_curation.ipynb` | `recsys_silver.dataset_curado_v2` (103 cols) |
| 2. Model Training | `02_training.ipynb` | Registered model in MLflow + Delta logs |
| 3. Mass Inference | `03_inference.ipynb` | Top-5 predictions per active user |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        BRONZE LAYER                              │
│  recsys_bronze.interactions_raw_v2  ·  recsys_bronze.airports    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   01 · DATA CURATION                             │
│                                                                  │
│  Identity Doc Curation ──► Route (IATA) Curation                │
│         │                         │                             │
│         ▼                         ▼                             │
│   DOC_VALIDO_V2             RUTA_CURADA + RUTA_VALIDA            │
│   ID_PERSONA_V2             Broadcast Join vs airports           │
│         │                         │                             │
│         └────────────┬────────────┘                             │
│                      ▼                                           │
│            Feature Engineering (103 cols)                        │
│   Geo · Haversine Distance · Route Complexity                    │
│   Popularity · Hub Scores · User Profile · Dates                 │
│                      ▼                                           │
│         recsys_silver.dataset_curado_v2                          │
│         Z-ORDER by (ANIO_EMISION, MES_EMISION)                   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   02 · MODEL TRAINING                            │
│                                                                  │
│   STAGE 1 — SASRec (Transformer)                                 │
│   User Sequence → Self-Attention (4 heads) → Top-200 Candidates  │
│                                                                  │
│   STAGE 2 — LambdaRank (LightGBM)                               │
│   200 Candidates + 80 Features → NDCG@5 Optimized → Top-5       │
│   Optuna HPO · 30 trials · Hard Negatives 80%                    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      MLOPS LAYER                                 │
│   MLflow Tracking · Model Registry · Delta Audit Table           │
│   recsys_gold.training_logs_v2                                   │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   03 · MASS INFERENCE                            │
│   Batch predictions · Top-5 per active user                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
NextStopAI/
│
├── README.md                         ← Project overview (this file)
├── requirements.txt                  ← Python dependencies
├── .gitignore
│
├── notebooks/
│   ├── 01_curation.ipynb             ← Data curation pipeline (PySpark)
│   ├── 02_training.ipynb             ← SASRec + LambdaRank training
│   └── 03_inference.ipynb            ← Mass inference pipeline
│
├── docs/
│   ├── 01_curation.md                ← Curation pipeline deep-dive
│   ├── 02_training.md                ← Training pipeline deep-dive
│   ├── 03_inference.md               ← Inference documentation
│   └── data_schema.md                ← Delta Lake schema reference
│
├── data/
│   └── README.md                     ← Data sources (no raw data stored)
│
└── models/
    └── README.md                     ← Model registry & artifacts reference
```

---

## Notebooks

### `01_curation.ipynb` — Data Curation Pipeline

Full PySpark pipeline that transforms raw transactional data into a clean, feature-rich Silver table.

**Curation stages:**

**1 · Identity Document Curation**
Normalizes document types (DNI, RUC, CE, Passport, CI), applies format validation rules per type, detects blacklisted placeholders (`TEST`, `000000`, `S/N`...), and identifies RUCs Masivos (> 1,000 records per entity) via Spark Window Functions. Produces `DOC_VALIDO_V2` and `ID_PERSONA_V2`.

**2 · B2C / B2B Segmentation**
Classifies customers: B2C (DNI, CE, Passport, CI) vs B2B (RUC). All downstream modeling uses the B2C segment exclusively.

**3 · Route (IATA) Curation**
Extracts IATA airport codes from raw route strings using `regexp_extract_all`, removes consecutive duplicates with a custom UDF (`limpiar_y_unir_ruta`), and validates each token against the commercial airport catalog (`SCHEDULED_SERVICE = yes`). Produces `RUTA_CURADA` and `RUTA_VALIDA`.

**4 · Feature Engineering (103 features)**

| Feature Group | Columns | Method |
|---|---|---|
| Route structure | `N_TOKENS`, `N_TRAMOS`, `N_ESCALAS`, `ES_IDA_VUELTA`, `ES_MULTITRAMO` | Native Spark |
| Geographic (OD) | `ORIG_COUNTRY`, `DEST_COUNTRY`, `ORIG_CONT`, `CRUZA_CONTINENTE_OD`, `TIPO_VUELO_OD` | Broadcast join vs airports catalog |
| Distance | `DIST_KM_OD`, `DIST_BIN_OD`, `DELTA_ELEV_FT_OD` | Haversine in Spark SQL (no UDF) |
| Complex route metrics | `DIST_KM_TOTAL_RUTA`, `DIST_KM_MAX/MEAN/STD_TRAMO`, `N_PAISES_VISITADOS`, `N_CONTINENTES_VISITADOS` | Broadcast UDF with airport dict |
| Popularity & Hub | `LOG_POP_RUTA`, `LOG_POP_OD`, `HUB_SCORE_ORIG`, `HUB_SCORE_DEST` | Window aggregations |
| User profile | `U_N_TRIPS`, `U_PCT_INTL`, `U_AVG_DIST_OD`, `U_MAX_DIST_OD`, `U_AVG_TRAMOS` | GroupBy aggregations |
| Temporal | `ANIO_EMISION`, `MES_EMISION`, `DIA_SEMANA_EMISION`, `ANTICIPACION_DIAS`, `DURACION_VIAJE` | Date parsing (coalesce multi-format) |

**5 · Optimized Save**
Writes to Delta Lake with `OPTIMIZE + ZORDER BY (ANIO_EMISION, MES_EMISION)` for fast temporal queries at training time.

> Full details: [`docs/01_curation.md`](docs/01_curation.md)

---

### `02_training.ipynb` — Two-Stage Training Pipeline

#### Stage 1 · SASRec — Candidate Generation

Self-attentive sequential Transformer that learns from user purchase sequences to generate 200 candidate routes per user.

| Parameter | Value |
|---|---|
| Embedding dimension | 128 |
| Attention heads | 4 |
| Max sequence length | 10 |
| Epochs | 40 |
| Batch size | 2,048 |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |

#### Stage 2 · LambdaRank — Re-Ranker

LightGBM ranking model that re-scores the 200 candidates using rich contextual features, optimizing for NDCG@5.

| Aspect | Detail |
|---|---|
| Objective | LambdaRank |
| HPO | Optuna · TPE Sampler · 30 trials |
| Negative sampling | 80% hard (top-2,500 popular routes) + 20% uniform |
| Training users | 22,000 |
| Negatives per positive | 80 |

> Full details: [`docs/02_training.md`](docs/02_training.md)

---

### `03_inference.ipynb` — Mass Inference

Loads the registered model from Databricks Model Registry and generates Top-5 route predictions for all active B2C users in batch mode.

> Full details: [`docs/03_inference.md`](docs/03_inference.md)

---

## Evaluation

Evaluated on **heavy users** (≥ 5 trips in the test holdout) using a temporal split (last 6 months as test).

| Metric | Description |
|---|---|
| **HR@5** | Hit Rate — at least 1 correct route in Top-5 |
| **Precision@5** | Fraction of Top-5 that are relevant |
| **Recall@5** | Fraction of relevant routes recovered |
| **MRR** | Mean Reciprocal Rank |
| **NDCG@5** | Normalized Discounted Cumulative Gain |
| **MAP@5** | Mean Average Precision |

Results per run are tracked in MLflow under `/Shared/RecSys_NextStopAI_Experiment` and persisted to `recsys_gold.training_logs_v2`.

---

## MLOps Stack

| Component | Tool | Purpose |
|---|---|---|
| Compute | Databricks | PySpark + GPU cluster |
| Raw data | Delta Lake Bronze | Transactional interactions + Airport catalog |
| Curated data | Delta Lake Silver | Feature-rich dataset (103 cols) |
| Audit log | Delta Lake Gold | Historical training records |
| Experiment tracking | MLflow | Metrics, params, plots, artifacts |
| Model registry | MLflow + Databricks | Model versioning and serving |
| HPO | Optuna (TPE) | Hyperparameter optimization |

### MLflow artifacts per training run

- Scalar metrics: HR@5, Precision@5, Recall@5, MRR, NDCG@5, MAP@5
- Best Optuna hyperparameters
- SASRec training loss curve (`plots/sasrec_loss.png`)
- Evaluation metrics bar chart (`plots/metrics_bar.png`)
- SASRec model weights (`sasrec_model/sasrec_weights.pth`)
- Inference context (`context_artifacts/inference_context.pkl`)
- LightGBM model with input signature for serving

---

## Reproducibility

```python
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
# Optuna: TPESampler(seed=SEED)
# LightGBM: deterministic=True, force_col_wise=True
```

---

## Setup

**Databricks (recommended)**
```python
%pip install optuna lightgbm
dbutils.library.restartPython()
```

**Local environment**
```bash
pip install torch optuna lightgbm mlflow scikit-learn pandas numpy matplotlib seaborn pyspark
```

**Key configuration**
```python
EXPERIMENT_NAME     = "/Shared/RecSys_NextStopAI_Experiment"
MODEL_REGISTRY_NAME = "RecSys_NextStopAI_Model"
TABLE_LOG_METRICS   = "recsys_gold.training_logs_v2"
TABLE_SILVER        = "recsys_silver.dataset_curado_v2"
TABLE_BRONZE        = "recsys_bronze.interactions_raw_v2"
TABLE_AIRPORTS      = "recsys_bronze.airports"
```

---

## Academic Context

| | |
|---|---|
| **Author** | Miguel Silva Caju |
| **Degree** | MSc Data Science & Artificial Intelligence |
| **Year** | 2025 |
| **GitHub** | [@ergodicdev](https://github.com/ergodicdev) |
| **Contact** | nextstopai.research@gmail.com |

---

<div align="center">
<sub>Built with curiosity, gradient descent, and a lot of Delta Lake queries · ergodicdev · 2025</sub>
</div>
