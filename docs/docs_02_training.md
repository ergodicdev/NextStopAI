# 02 · Training Pipeline

> **Notebook:** `notebooks/02_training.ipynb`  
> **Input:** `recsys_silver.dataset_curado_v2`  
> **Output:** MLflow Model Registry + `recsys_gold.training_logs_v2`

---

## Overview

The training notebook implements a **two-stage recommendation pipeline**:

1. **Stage 1 — SASRec:** A self-attentive sequential Transformer that generates 200 candidate routes per user from their purchase history.
2. **Stage 2 — LambdaRank:** A LightGBM ranking model that re-scores the 200 candidates using rich contextual features, optimized directly for NDCG@5.

The full pipeline is wrapped in a single MLflow run that logs metrics, plots, model artifacts, and writes an audit record to a Delta Gold table.

---

## Configuration

```python
SEED                = 42
EXPERIMENT_NAME     = "/Shared/RecSys_NextStopAI_Experiment"
MODEL_REGISTRY_NAME = "RecSys_NextStopAI_Model"
TABLE_LOG_METRICS   = "recsys_gold.training_logs_v2"
TABLE_NAME          = "recsys_silver.dataset_curado_v2"

# Model hyperparameters
MAX_LEN             = 10      # Max user sequence length
EPOCHS_SASREC       = 40
BATCH_SIZE          = 2048

# Re-ranker configuration
TOPK_CAND           = 200     # SASRec candidates per user
RANK_N_USERS        = 22000   # Users sampled for LambdaRank training
N_NEGS_TOTAL        = 80      # Negatives per positive
NEG_POP_FRAC        = 0.8     # Fraction of hard (popular) negatives
N_TRIALS_OPTUNA     = 30      # Optuna trials for HPO
```

---

## Reproducibility

All randomness is fully controlled before any computation begins:

```python
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

LightGBM uses `deterministic=True, force_col_wise=True` and Optuna uses `TPESampler(seed=SEED)`.

---

## Pipeline Stages

### Stage 0 · Data Loading & Preprocessing

Reads `recsys_silver.dataset_curado_v2` from Delta Lake, applies B2C business filters, and encodes users and items as integer indices.

```python
df_spark.filter(
    (col("TIPO_CLIENTE") == "B2C") &
    (col("RUC_MASIVO") == False) &
    (col("RUTA_VALIDA") == True)
)
```

**Encoding:**
- `user_le = LabelEncoder()` → `u_idx`
- `item_le = LabelEncoder()` → `i_idx`
- All categorical and geographic columns are also label-encoded for LightGBM

---

### Stage 1 · Temporal Split

```python
fecha_corte = fecha_max - pd.DateOffset(months=6)

train_df = df[df[DATE_COL] < fecha_corte]   # Everything before cutoff
test_df  = df[df[DATE_COL] >= fecha_corte]  # Last 6 months as holdout
```

This strict temporal split prevents data leakage — no future information is used in training features.

---

### Stage 2 · Anti-Leakage Feature Engineering

Popularity and hub score features are computed **exclusively from training data** and then applied to the test set. This prevents leakage from test-period item frequencies.

```python
pop_ruta_tr = train_df["i_idx"].value_counts().to_dict()
train_df["LOG_POP_RUTA_TR"] = log1p_safe(train_df["i_idx"].map(pop_ruta_tr))
test_df["LOG_POP_RUTA_TR"]  = log1p_safe(test_df["i_idx"].map(pop_ruta_tr))
```

---

### Stage 3 · SASRec — Candidate Generation

#### Architecture

```python
class SASRec(nn.Module):
    def __init__(self, n_items, embed_dim=128, max_len=10):
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, embed_dim)
        self.attn     = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.ln       = nn.LayerNorm(embed_dim)
        self.fc       = nn.Linear(embed_dim, n_items)
```

| Parameter | Value |
|---|---|
| Embedding dimension | 128 |
| Attention heads | 4 |
| Max sequence length | 10 |
| Padding index | 0 (left-padded sequences) |

#### Training

The model is trained on `(sequence → next item)` pairs extracted from each user's purchase history. Sequences shorter than max_len are left-padded with zeros.

| Parameter | Value |
|---|---|
| Epochs | 40 |
| Batch size | 2,048 |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Device | CUDA if available, else CPU |

#### Candidate Generation

After training, SASRec generates 200 candidate routes per user in batched GPU inference:

```python
with torch.no_grad():
    logits = model_sas(X_eval)           # Forward pass
    probs = torch.softmax(logits, dim=-1)
    _, top_idx = torch.topk(probs, TOPK_CAND)  # Top-200 per user
```

---

### Stage 4 · LambdaRank Re-Ranker

#### Negative Sampling Strategy

For each training user, one positive item (the last purchase) and `N_NEGS_TOTAL = 80` negative items are sampled:

| Type | Fraction | Source |
|---|---|---|
| Hard negatives | 80% | Sampled from top-2,500 popular routes |
| Uniform negatives | 20% | Sampled uniformly across all items |

Hard negatives force the model to distinguish between popular routes the user did NOT choose, which is the core challenge in recommendation.

#### Feature Groups for Ranking

| Group | Features |
|---|---|
| Transaction | `VENTA`, `TARIFA`, `TAX`, `SEGURO`, `DESCUENTO`, `PORC_COMISION`, `N_ESCALAS`, `CLASE`, `TIPO_DE_RUTA_I_R_C` |
| User profile | `U_N_TRIPS`, `U_PCT_INTL`, `U_AVG_DIST_OD`, `U_MAX_DIST_OD`, `U_AVG_TRAMOS` |
| Temporal | `ANIO_EMISION`, `MES_EMISION`, `DIA_SEMANA_EMISION`, `TIENE_ANTICIPACION`, `TIENE_DURACION` |
| Route popularity | `LOG_POP_RUTA_TR`, `LOG_POP_OD_TR`, `LOG_HUB_ORIG_TR`, `LOG_HUB_DEST_TR` |
| Geographic | `ORIG_COUNTRY`, `DEST_COUNTRY`, `ORIG_CONT`, `DEST_CONT`, `CRUZA_CONTINENTE_OD` |
| Route geometry | `DIST_KM_OD`, `N_TRAMOS_CALC`, `N_PAISES_VISITADOS`, `ES_IDA_VUELTA`, `ES_MULTITRAMO` |
| Contextual signals | `global_pop`, `user_loyalty_max`, `days_since_route`, `same_odpair_as_last`, `same_orig_country_as_last`, `same_dest_country_as_last` |

#### Hyperparameter Optimization with Optuna

```python
params tuned:
  learning_rate      → [0.01, 0.08]
  num_leaves         → [31, 320]
  max_depth          → [4, 16]
  min_data_in_leaf   → [20, 320]
  feature_fraction   → [0.4, 0.7]
  bagging_fraction   → [0.6, 1.0]
  bagging_freq       → [1, 10]
  n_estimators       → [600, 2600]
  lambda_l1          → [0.0, 5.0]
  lambda_l2          → [0.0, 10.0]
```

Each trial is evaluated on a validation split (last 20% of training groups) using a custom **NDCG@5** implementation:

```python
def ndcg_at_k(y_true, scores, group, k=5):
    # DCG / IDCG per query group
    # Returns mean NDCG@5 across all groups
```

---

### Stage 5 · Evaluation

Evaluated on **heavy users** (≥ 5 trips in the test period) using the two-stage prediction pipeline:

```python
def predict_top_5(u_idx, sas_candidates, train_df_hist, ranker_model):
    # 1. Build feature rows for all candidates + user history items
    # 2. Score with LambdaRank
    # 3. Return top-5 ranked routes
```

| Metric | Description |
|---|---|
| `hit_rate_at_5` | At least 1 correct route in Top-5 |
| `precision_at_5` | Fraction of Top-5 that are relevant |
| `recall_at_5` | Fraction of relevant routes recovered |
| `mrr` | Mean Reciprocal Rank |
| `ndcg_at_5` | Normalized Discounted Cumulative Gain |
| `map_at_5` | Mean Average Precision |

---

### Stage 6 · MLOps Logging

Everything is logged inside a single `mlflow.start_run()` context:

#### A · Scalar metrics & parameters
```python
mlflow.log_metrics(final_metrics)
mlflow.log_params(best_optuna_params)
```

#### B · Plots
```python
mlflow.log_figure(fig_loss, "plots/sasrec_loss.png")
mlflow.log_figure(fig_metrics, "plots/metrics_bar.png")
```

#### C · Model registration (LightGBM for serving)
```python
mlflow.lightgbm.log_model(
    lgb_model=ranker_model,
    artifact_path="model",
    registered_model_name=MODEL_REGISTRY_NAME,
    signature=signature,
    input_example=input_example
)
```

#### D · Artifacts (SASRec + inference context)
Uses a `tempfile.TemporaryDirectory()` to save weights and context to disk, upload to MLflow, then automatically clean up local files:

```python
torch.save(model_sas.state_dict(), sasrec_path)          # → sasrec_model/sasrec_weights.pth
pickle.dump(inference_context, f)                         # → context_artifacts/inference_context.pkl
```

`inference_context` contains everything needed for inference:
`user_le`, `item_le`, `user_hist`, `item_lookup`, `user_lookup`, `global_pop`, `num_items`, `feat_cols`

#### E · Delta audit log
```python
df_log.write.format("delta").mode("append").saveAsTable("recsys_gold.training_logs_v2")
```

One row per run: `run_id`, `timestamp`, `model_uri`, `ndcg_at_5`, `hit_rate_at_5`, `metrics_json`

---

## Key Design Decisions

**Why two stages instead of a single model?**
SASRec excels at modeling sequential user behavior to retrieve relevant candidates from thousands of routes efficiently. LambdaRank then applies rich contextual features (geography, pricing, recency) that would be too expensive to compute at retrieval scale. The two-stage design balances recall (SASRec) with precision (LambdaRank).

**Why LambdaRank instead of pointwise regression?**
LambdaRank optimizes the ranking objective directly (NDCG) rather than predicting individual relevance scores. This is crucial for Top-5 recommendations where the relative ordering matters more than the absolute score values.

**Why hard negatives at 80%?**
Random negatives are trivially easy — the model quickly learns to distinguish popular routes from noise. Hard negatives (popular routes the user didn't choose) force the model to learn fine-grained preference signals, leading to significantly better NDCG@5.

**Why `tempfile.TemporaryDirectory()` for artifact saving?**
This pattern saves SASRec weights to a temporary directory, uploads them to MLflow, and automatically deletes the local copy when the context block exits. It keeps the cluster storage clean without manual cleanup and avoids accumulating gigabytes of model weights across training runs.
