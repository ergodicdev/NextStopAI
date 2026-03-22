# 01 · Data Curation Pipeline

> **Notebook:** `notebooks/01_curation.ipynb`  
> **Input:** `recsys_bronze.interactions_raw_v2` · `recsys_bronze.airports`  
> **Output:** `recsys_silver.dataset_curado_v2` (103 columns, Z-Ordered)

---

## Overview

The curation notebook transforms raw transactional airline ticketing data into a clean, feature-enriched Silver table. The pipeline runs entirely on PySpark, using broadcast joins, window functions, and native Spark SQL expressions to process data at scale without performance bottlenecks.

---

## Pipeline Stages

### Stage 0 · Memory Management

Before running, the notebook clears all cached Spark DataFrames and Python variables from previous runs to avoid memory conflicts in the Databricks cluster.

```python
spark.catalog.clearCache()
gc.collect()
```

---

### Stage 1 · Identity Document Curation

**Goal:** Produce a clean, validated user identifier (`ID_PERSONA_V2`) from raw document fields.

#### 1.1 Base Cleaning
- Trims whitespace and converts to uppercase
- Fills null `TIPO_DOCUMENTO` with `"SIN_TIPO"`

#### 1.2 Document Type Normalization
Maps raw type codes to standardized categories:

| Raw Codes | Standardized |
|---|---|
| `PAS`, `PSP`, `PASS`, `PPT` | `PASAPORTE` |
| `DNI` | `DNI` |
| `RUC` | `RUC` |
| `CE` | `CE` |
| `CI` | `CI` |

#### 1.3 Format Validation Rules

| Document Type | Rule |
|---|---|
| DNI | Exactly 8 digits |
| RUC | Exactly 11 digits |
| CE | 9–12 digits only |
| Passport | Alphanumeric, 6–12 characters |
| CI | Alphanumeric, 6–12 characters |

#### 1.4 Advanced Rules (Blacklist + RUC Masivo)

- **Blacklist patterns:** `000000`, `S/N`, `SN`, `SIN`, `TEST`, `PRUEBA`, `GENERIC`, `GENERAL`, `ESENTER`
- **RUC Masivo:** RUC codes with more than 1,000 records — detected via Spark `Window.partitionBy("ID_PERSONA")` (more efficient than groupBy + join)
- **Passport length check:** Passports shorter than 7 characters are invalidated

#### 1.5 Final Validation Flag: `DOC_VALIDO_V2`

```
DOC_VALIDO_V2 = DOC_VALIDO
             AND NOT DOC_PLACEHOLDER
             AND PASAPORTE_LONGITUD_OK
             AND NOT RUC_MASIVO
```

#### 1.6 B2C / B2B Segmentation

| Segment | Documents |
|---|---|
| **B2C** | DNI, CE, Passport, CI |
| **B2B** | RUC |
| **DESCONOCIDO** | Everything else |

All downstream modeling targets the **B2C** segment exclusively.

---

### Stage 2 · Route (IATA) Curation

**Goal:** Extract clean IATA airport codes from raw route strings and validate them against a commercial airport catalog.

#### 2.1 Route String Cleaning
- Uppercases and trims whitespace
- Normalizes separators (`→`, `>`, `\`, `/`, `;`, `,`) to `-`

#### 2.2 Blacklist Filtering
Routes matching non-flight entries are discarded:
`EQUIPAJE`, `PENALIDAD`, `ASIENTO`, `NO SHOW`, `REEMISION`, `REEMBOLSO`, `FEE`, `CARGO`, `UPGRADE`, etc.

#### 2.3 IATA Token Extraction
Uses `regexp_extract_all(RUTA_SEP, '([A-Z]{3})', 1)` to extract 3-letter codes.

#### 2.4 Consecutive Duplicate Removal (custom UDF)
```python
def limpiar_y_unir_ruta(tokens):
    # Removes A-A-B patterns → A-B
    # Requires at least 2 distinct airports
```

#### 2.5 IATA Validation via Broadcast Join
Validates each token in a route against the commercial airport catalog:
- Filters airports: `TYPE IN ('large_airport', 'medium_airport')` AND `SCHEDULED_SERVICE = 'yes'`
- Uses Left Anti Join to identify routes with at least one invalid IATA code
- Routes with any invalid token are marked `RUTA_VALIDA = False`

**Why `SCHEDULED_SERVICE = yes` matters:**  
Including airports without scheduled commercial service introduces impossible routes, noise in embeddings, and artificial cold-start problems in the model.

---

### Stage 3 · Feature Engineering

#### 3.1 Route Structure Features (Native Spark)

| Feature | Description |
|---|---|
| `N_TOKENS` | Number of airports in route |
| `N_TRAMOS` | Number of flight legs (N_TOKENS - 1) |
| `N_ESCALAS` | Number of stopovers (max(N_TOKENS - 2, 0)) |
| `ORIGEN` | First airport code |
| `DESTINO` | Last airport code |
| `ES_IDA_VUELTA` | True if ORIGEN == DESTINO and N_TOKENS >= 3 |
| `ES_MULTITRAMO` | True if N_TRAMOS >= 2 |
| `N_AEROPUERTOS_UNICOS` | Count of distinct airports (array_distinct) |
| `N_REPETIDOS_CONSEC` | Count of consecutive repeated airports (UDF) |
| `RUTA_LEN_CHARS` | Character length of the route string |

#### 3.2 Geographic Features — OD Pair (Broadcast Joins)

Two broadcast joins against the airport metadata table enrich each record with:

| Feature | Description |
|---|---|
| `ORIG_COUNTRY` / `DEST_COUNTRY` | ISO country code |
| `ORIG_CONT` / `DEST_CONT` | Continent |
| `ORIG_TYPE` / `DEST_TYPE` | Airport type (large/medium) |
| `ES_DOMESTICO_OD` | Origin and destination same country |
| `ES_INTERNACIONAL_OD` | Origin and destination different countries |
| `CRUZA_CONTINENTE_OD` | Crosses continents |
| `TIPO_VUELO_OD` | `NACIONAL` / `INTERNACIONAL` |
| `DELTA_ELEV_FT_OD` | Elevation difference (DEST_ELEV - ORIG_ELEV) |

#### 3.3 Distance — Haversine (Native Spark SQL)

Haversine formula implemented entirely with Spark SQL math functions (no UDF) — 100x faster than a Python UDF:

```python
def spark_haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088  # Earth radius in km
    ...
    return c * R   # Great-circle distance in km
```

| Feature | Description |
|---|---|
| `DIST_KM_OD` | Great-circle distance between origin and destination |
| `DIST_BIN_OD` | Distance bucket: `0-300`, `300-800`, `800-1500`, `1500-3000`, `3000-6000`, `6000+` |

#### 3.4 Complex Route Metrics (Broadcast UDF)

A single UDF with a broadcast dictionary of airport metadata computes per-route metrics by iterating over each flight leg:

| Feature | Description |
|---|---|
| `DIST_KM_TOTAL_RUTA` | Total route distance across all legs |
| `DIST_KM_MAX_TRAMO` | Longest individual leg |
| `DIST_KM_MEAN_TRAMO` | Average leg distance |
| `DIST_KM_STD_TRAMO` | Standard deviation of leg distances |
| `N_TRAMOS_CALC` | Calculated number of legs |
| `N_TRAMOS_NACIONAL` | Legs within the same country |
| `N_TRAMOS_INTERNACIONALES` | International legs |
| `PCT_TRAMOS_INTERNACIONALES` | % of international legs |
| `N_PAISES_VISITADOS` | Unique countries in route |
| `N_CONTINENTES_VISITADOS` | Unique continents in route |
| `CAMBIA_PAIS_EN_RUTA` | True if route crosses at least one country boundary |
| `CAMBIA_CONT_EN_RUTA` | True if route crosses at least one continent boundary |

#### 3.5 Popularity & Hub Scores

| Feature | Description |
|---|---|
| `POP_RUTA_GLOBAL` | Raw count of route occurrences |
| `LOG_POP_RUTA` | Log-transformed global route popularity |
| `POP_OD_PAIR` | Raw count of OD pair occurrences |
| `LOG_POP_OD` | Log-transformed OD pair popularity |
| `HUB_SCORE_ORIG` | Relative importance of origin airport |
| `HUB_SCORE_DEST` | Relative importance of destination airport |
| `LOG_HUB_ORIG` / `LOG_HUB_DEST` | Log-transformed hub scores |

#### 3.6 User Profile Features

Aggregated per user from training history:

| Feature | Description |
|---|---|
| `U_N_TRIPS` | Total number of trips |
| `U_PCT_INTL` | Percentage of international trips |
| `U_AVG_DIST_OD` | Average OD distance |
| `U_MAX_DIST_OD` | Maximum OD distance traveled |
| `U_AVG_TRAMOS` | Average number of legs per trip |

#### 3.7 Temporal Features

Date parsing uses a coalesce strategy to handle multiple formats present in the raw data:

```python
F.coalesce(
    F.to_date(col, "dd/MM/yyyy"),   # Peru/Latam format (priority)
    F.to_date(col, "d/M/yyyy"),     # Without leading zeros
    F.to_date(col, "yyyy-MM-dd"),   # ISO Standard
    F.to_date(col, "yyyy/MM/dd")    # ISO with slashes
)
```

| Feature | Description |
|---|---|
| `FECHA_EMISION_LIMPIA` | Parsed emission date |
| `ANIO_EMISION` | Year |
| `MES_EMISION` | Month (Z-Order key) |
| `DIA_SEMANA_EMISION` | Day of week |
| `ANTICIPACION_DIAS` | Days between booking and departure |
| `ANTICIPACION_BUCKET` | Bucketed anticipation |
| `DURACION_VIAJE` | Trip duration in days |
| `DURACION_BUCKET` | Bucketed duration |

---

### Stage 4 · Quality Audit

Before saving, the notebook runs a null-value audit across all 103 columns and prints:

- Total rows × columns
- Null count and % per column (Top 105 by null rate)
- Global null percentage of the dataset

---

### Stage 5 · Optimized Save

```python
df_save.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("recsys_silver.dataset_curado_v2")

spark.sql("OPTIMIZE recsys_silver.dataset_curado_v2 ZORDER BY (ANIO_EMISION, MES_EMISION)")
```

Column ordering places the Z-Order keys (`ANIO_EMISION`, `MES_EMISION`) first to ensure the Delta OPTIMIZE command can use them correctly.

---

## Output Schema Summary

**Table:** `recsys_silver.dataset_curado_v2`  
**Total columns:** 103  
**Partitioning strategy:** Z-ORDER by `(ANIO_EMISION, MES_EMISION)`

For the full column-by-column schema, see [`docs/data_schema.md`](data_schema.md).

---

## Key Design Decisions

**Why Window Functions instead of GroupBy + Join for RUC Masivo?**  
Window functions operate on the partition in a single pass without shuffling data twice, making them significantly more efficient for counting occurrences per entity at scale.

**Why Broadcast Joins for the airport catalog?**  
The airport catalog is small (~5,000 IATA codes) relative to the transactions table. Broadcasting it to all nodes eliminates the need for a full shuffle join.

**Why native Spark SQL for Haversine instead of a UDF?**  
Python UDFs in PySpark require serialization/deserialization between JVM and Python processes for every row. Using Spark's built-in math functions (`F.sin`, `F.cos`, `F.radians`, `F.sqrt`, `F.asin`) keeps computation within the JVM, achieving ~100x speedup at scale.

**Why `SCHEDULED_SERVICE = yes` for airport filtering?**  
Including non-commercial airports (helipads, military bases, private airstrips) would introduce impossible routes into the training data, degrading embedding quality and creating artificial cold-start problems.
