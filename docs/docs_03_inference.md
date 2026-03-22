# 03 · Mass Inference Pipeline

> **Notebook:** `notebooks/03_inference.ipynb`  
> **Input:** `recsys_silver.dataset_curado_v2` + MLflow Model Registry  
> **Output:** `recsys_gold.recomendaciones_v3` (Top-5 per active B2C user)

---

## Overview

The inference notebook loads the trained two-stage model from MLflow Model Registry and generates Top-5 route recommendations for every eligible B2C customer. It uses a **progressive batch saving strategy** to avoid memory overflow on large user bases, writing results incrementally to a Delta Lake Gold table.

---

## Pipeline Stages

### Stage 1 · Configuration

```python
FULL_MODEL_NAME = "gpu_classic_workspace.default.recsys_nextstopai_model"
TABLE_SOURCE    = "recsys_silver.dataset_curado_v2"
OUTPUT_TABLE    = "recsys_gold.recomendaciones_v3"
TOP_K           = 5
BATCH_SIZE      = 100    # Users per processing batch
SAVE_EVERY_N    = 10     # Save to Delta every 10 batches (~1,000 users)
```

---

### Stage 2 · Model Loading from MLflow Registry

All artifacts are downloaded directly from MLflow using the **latest registered version** — no hardcoded version numbers.

```
MLflow Registry
     │
     ├── inference_context.pkl     ← User/item encoders, histories, lookups
     ├── sasrec_model/
     │   └── sasrec_weights.pth    ← SASRec Transformer weights
     └── model/
         └── model.pkl             ← LightGBM LambdaRank ranker
```

**Artifacts restored from `inference_context.pkl`:**

| Object | Type | Purpose |
|---|---|---|
| `user_le` | LabelEncoder | Maps user IDs to indices |
| `item_le` | LabelEncoder | Maps route codes to indices |
| `train_hist` | dict | User purchase sequences |
| `item_lookup` | dict | Per-item feature vectors |
| `user_lookup` | dict | Per-user profile features |
| `global_pop` | dict | Global route popularity distribution |
| `num_items` | int | Total number of unique routes |
| `feat_cols` | list | Feature column order for the ranker |

**Model version resolution:**
```python
versions = client.search_model_versions(f"name='{FULL_MODEL_NAME}'")
latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
```

---

### Stage 3 · User Preparation & Business Filters

Reads the Silver table and applies the same B2C business filters used during training:

```python
df_spark.filter(
    (col("TIPO_CLIENTE") == "B2C") &
    (col("RUC_MASIVO") == False) &
    (col("RUTA_VALIDA") == True)
)
```

**Eligibility criteria for inference:**
- Must be in the `user_le` encoder (seen during training)
- Must have ≥ 3 purchase records
- Only B2C segment, valid routes, non-bulk RUCs

**Pre-computed behavioral dictionaries** (built once before the main loop for efficiency):

| Dictionary | Content |
|---|---|
| `user_loyalty_dict` | Max normalized frequency of any single route per user |
| `user_last_date` | Most recent purchase date per user |
| `user_item_last_date` | Most recent date per (user, item) pair |

---

### Stage 4 · Progressive Batch Inference

The main loop processes users in batches of 100, combining SASRec candidate generation with LightGBM re-ranking.

#### 4.1 SASRec — Candidate Generation (GPU batched)

```python
# Build sequence for each user in the batch
sequences = [get_sequence(idx, train_hist) for idx in batch_idxs]

# Batch inference on GPU
tensor_seq = torch.tensor(sequences, dtype=torch.long).to(device)
logits = model_sas(tensor_seq)
_, top_cands_torch = torch.topk(logits, 200, dim=-1)
```

Generates 200 candidate routes per user in a single GPU forward pass.

#### 4.2 LightGBM — Per-User Re-Ranking

For each user in the batch, constructs a feature matrix over the 200 candidates and scores them:

**Features built per candidate:**

| Feature | Description |
|---|---|
| `global_pop` | `log1p(pop_val × 10,000)` — log-scaled global popularity |
| `user_loyalty_max` | Max normalized frequency of any single route for this user |
| `days_since_route` | Days since user last traveled this route (730 if never) |
| `same_odpair_as_last` | 1 if candidate OD pair matches user's last trip |
| `same_orig_country_as_last` | 1 if candidate origin country matches last trip |
| `same_dest_country_as_last` | 1 if candidate destination country matches last trip |
| `item_lookup features` | All route-level features from training context |

```python
scores = ranker_model.predict(pd.DataFrame(rows)[feat_cols])
sorted_indices = np.argsort(scores)[::-1][:TOP_K]
```

#### 4.3 Progressive Delta Save

To avoid memory overflow with large user populations, results are flushed to Delta Lake every `SAVE_EVERY_N` batches (every ~1,000 users):

```python
if (batch_num + 1) % SAVE_EVERY_N == 0 or is_last_batch:
    sdf_chunk = spark.createDataFrame(pd.DataFrame(recommendations_buffer))
    sdf_chunk.write.format("delta").mode("append").saveAsTable(OUTPUT_TABLE)
    recommendations_buffer = []   # Free memory
```

This pattern ensures the pipeline can handle tens of thousands of users without running out of driver memory.

---

## Output Schema

**Table:** `recsys_gold.recomendaciones_v3`

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Anonymized user identifier (`ID_PERSONA_V2`) |
| `rank` | int | Position in Top-5 (1 = best) |
| `ruta` | string | Recommended route (e.g. `LIM-MIA-LIM`) |
| `score` | float | LambdaRank confidence score |
| `model_version` | string | MLflow model version used (e.g. `v3`) |
| `execution_date` | timestamp | Inference timestamp |

---

## Key Design Decisions

**Why progressive saving instead of collecting all results first?**  
At 22,000+ users × 5 recommendations each, keeping the full results in memory before writing would require ~550,000 rows in the driver. The incremental save pattern flushes every 1,000 users, keeping peak memory usage low and providing fault tolerance — if the cluster fails mid-run, previously saved batches are preserved in Delta.

**Why download `model.pkl` manually instead of `mlflow.lightgbm.load_model()`?**  
The MLflow LightGBM flavor wrapper adds a deserialization layer that can fail when the registered model was saved with `log_model` but needs to be loaded in a different cluster configuration. Loading the raw pickle directly is more portable and avoids dependency on the MLflow flavor's internal format.

**Why pre-compute `user_loyalty_dict` and `user_item_last_date` before the main loop?**  
Computing these metrics inside the per-user loop would trigger repeated `groupby` operations on a DataFrame for every user. Pre-computing them once and storing in dictionaries reduces the inference time per user from O(n) to O(1) lookups.

**Why `torch.topk(logits, 200)` in batch instead of per-user?**  
Batching the SASRec forward pass across 100 users at once leverages GPU parallelism fully. Calling the model once per user would serialize GPU operations and leave most of the GPU idle.

---

## Performance Notes

| Aspect | Approach | Benefit |
|---|---|---|
| SASRec inference | Batched GPU (100 users/call) | Full GPU utilization |
| Feature dictionaries | Pre-computed before loop | O(1) lookups per user |
| Delta writes | Every 1,000 users (append) | Low peak memory, fault tolerant |
| User filter | Applied in Spark before toPandas | Reduces driver data transfer |
| Progress tracking | `tqdm` with live description | Monitoring at scale |
