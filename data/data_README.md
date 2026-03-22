# Data Sources

> ⚠️ No raw data is stored in this repository. All data lives in Databricks Delta Lake.

---

## Delta Lake Tables

| Table | Layer | Description |
|---|---|---|
| `recsys_bronze.interactions_raw_v2` | Bronze | Raw airline ticketing transactions |
| `recsys_bronze.airports` | Bronze | Commercial airport catalog (IATA codes, coordinates, country, continent) |
| `recsys_silver.dataset_curado_v2` | Silver | Curated dataset — 103 features, B2C validated, Z-Ordered |
| `recsys_gold.recomendaciones_v3` | Gold | Top-5 route recommendations per user |
| `recsys_gold.training_logs_v2` | Gold | MLOps training audit trail |

## Data Access

Data is accessed exclusively through Databricks Spark sessions:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.table("recsys_silver.dataset_curado_v2")
```

## Privacy

All user identifiers are anonymized as `TIPO_DOC_STANDARD_NUM_DOCUMENTO` during the curation pipeline. No personal information (names, addresses, payment data) is stored in any Silver or Gold table.

## Schema Reference

See [`docs/data_schema.md`](../docs/data_schema.md) for the full column-by-column reference of all tables.
