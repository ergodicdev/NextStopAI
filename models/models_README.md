# Model Registry

> All models are versioned and served via MLflow Model Registry on Databricks.  
> No model weights are stored in this repository.

---

## Registered Models

### `RecSys_NextStopAI_Model` (LightGBM LambdaRanker)

| Property | Value |
|---|---|
| Registry name | `gpu_classic_workspace.default.recsys_nextstopai_model` |
| Framework | LightGBM |
| Objective | LambdaRank |
| Primary metric | NDCG@5 |
| Input | Feature matrix (candidate routes × contextual features) |
| Output | Ranking scores per candidate |

### SASRec Weights (stored as MLflow artifact)

| Property | Value |
|---|---|
| Artifact path | `sasrec_model/sasrec_weights.pth` |
| Architecture | Self-Attentive Sequential Recommendation |
| Embedding dim | 128 |
| Attention heads | 4 |
| Max sequence length | 10 |

---

## Artifacts per Run

Each MLflow training run stores:

```
run_id/
├── plots/
│   ├── sasrec_loss.png              ← SASRec training loss curve
│   └── metrics_bar.png              ← Evaluation metrics chart
├── sasrec_model/
│   └── sasrec_weights.pth           ← PyTorch state dict
├── context_artifacts/
│   └── inference_context.pkl        ← Encoders, lookups, feature list
└── model/
    └── model.pkl                    ← LightGBM ranker (pickle)
```

## Loading a Model for Inference

```python
from mlflow.tracking import MlflowClient
import mlflow, pickle

client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]

# Download inference context
ctx_path = mlflow.artifacts.download_artifacts(
    run_id=latest.run_id,
    artifact_path="context_artifacts/inference_context.pkl"
)
with open(ctx_path, "rb") as f:
    ctx = pickle.load(f)
```

## Experiment Tracking

MLflow experiment: `/Shared/RecSys_NextStopAI_Experiment`

Metrics tracked per run:

| Metric | Description |
|---|---|
| `hit_rate_at_5` | At least 1 correct route in Top-5 |
| `precision_at_5` | Fraction of Top-5 that are relevant |
| `recall_at_5` | Fraction of relevant routes recovered |
| `mrr` | Mean Reciprocal Rank |
| `ndcg_at_5` | Normalized Discounted Cumulative Gain |
| `map_at_5` | Mean Average Precision |
