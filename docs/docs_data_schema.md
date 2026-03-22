# Data Schema Reference

> Delta Lake tables used across the NextStopAI pipeline

---

## Layer Overview

```
recsys_bronze  ──►  recsys_silver  ──►  recsys_gold
   Raw data          Curated data        Outputs
```

| Table | Layer | Role |
|---|---|---|
| `recsys_bronze.interactions_raw_v2` | Bronze | Raw transactional data |
| `recsys_bronze.airports` | Bronze | Airport metadata catalog |
| `recsys_silver.dataset_curado_v2` | Silver | Curated + feature-rich dataset |
| `recsys_gold.recomendaciones_v3` | Gold | Top-5 recommendations per user |
| `recsys_gold.training_logs_v2` | Gold | MLflow training audit trail |

---

## `recsys_silver.dataset_curado_v2`

**103 columns · Z-ORDER by (ANIO_EMISION, MES_EMISION)**

### Identity & Segmentation

| Column | Type | Description |
|---|---|---|
| `ID_PERSONA_V2` | string | Validated user ID (`TIPO_DOC_STD_NUM_DOCUMENTO`) |
| `TIPO_CLIENTE` | string | `B2C` / `B2B` / `DESCONOCIDO` |
| `DOC_VALIDO_V2` | boolean | Final document validity flag |
| `RUC_MASIVO` | boolean | True if RUC has > 1,000 records |

### Transaction Features

| Column | Type | Description |
|---|---|---|
| `NUMERO_DE_BOLETO` | string | Ticket number |
| `CLASE` | string | Booking class |
| `TARIFA` | float | Base fare |
| `VENTA` | float | Total sale amount |
| `TAX` | float | Taxes |
| `SEGURO` | float | Insurance |
| `DESCUENTO` | float | Discount applied |
| `PORC_COMISION` | float | Commission percentage |
| `PORC_OVER` | float | Override percentage |
| `TIPO_DE_RUTA_I_R_C` | string | Route type (International/Regional/Cabotage) |
| `QUIEN_RESERVA` | string | Booking agent type |
| `ES_IDA_VUELTA` | boolean | Round trip flag |
| `ES_MULTITICKET` | boolean | Multi-ticket booking |

### Route Features

| Column | Type | Description |
|---|---|---|
| `RUTA_CURADA` | string | Curated IATA route (e.g. `LIM-MIA-LIM`) |
| `RUTA_VALIDA` | boolean | All IATA codes are commercially valid |
| `N_TOKENS` | int | Number of airports in route |
| `N_TRAMOS` | int | Number of flight legs |
| `N_ESCALAS` | int | Number of stopovers |
| `ORIGEN` | string | Origin airport (first IATA) |
| `DESTINO` | string | Destination airport (last IATA) |
| `ES_MULTITRAMO` | boolean | Route has 2+ legs |
| `RUTA_LEN_CHARS` | int | Character length of route string |
| `N_AEROPUERTOS_UNICOS` | int | Distinct airports in route |
| `N_REPETIDOS_CONSEC` | int | Consecutive repeated airports |

### Geographic Features (OD Pair)

| Column | Type | Description |
|---|---|---|
| `ORIG_COUNTRY` | string | Origin ISO country code |
| `DEST_COUNTRY` | string | Destination ISO country code |
| `ORIG_CONT` | string | Origin continent |
| `DEST_CONT` | string | Destination continent |
| `ORIG_TYPE` | string | Origin airport type |
| `DEST_TYPE` | string | Destination airport type |
| `OD_PAIR` | string | Origin-Destination pair key |
| `ES_DOMESTICO_OD` | boolean | Same country OD |
| `ES_INTERNACIONAL_OD` | boolean | Different country OD |
| `TIPO_VUELO_OD` | string | `NACIONAL` / `INTERNACIONAL` |
| `CRUZA_CONTINENTE_OD` | boolean | OD crosses continents |
| `DELTA_ELEV_FT_OD` | float | Elevation difference DEST - ORIG (ft) |

### Distance Features

| Column | Type | Description |
|---|---|---|
| `DIST_KM_OD` | float | Haversine distance OD (km) |
| `DIST_BIN_OD` | string | Distance bucket: `0-300`, `300-800`, `800-1500`, `1500-3000`, `3000-6000`, `6000+` |
| `DIST_KM_TOTAL_RUTA` | float | Total route distance across all legs |
| `DIST_KM_MAX_TRAMO` | float | Longest individual leg (km) |
| `DIST_KM_MEAN_TRAMO` | float | Average leg distance (km) |
| `DIST_KM_STD_TRAMO` | float | Std dev of leg distances |

### Complex Route Metrics

| Column | Type | Description |
|---|---|---|
| `N_TRAMOS_CALC` | int | Calculated number of legs |
| `N_TRAMOS_NACIONAL` | int | Domestic legs |
| `N_TRAMOS_INTERNACIONALES` | int | International legs |
| `PCT_TRAMOS_INTERNACIONALES` | float | % international legs |
| `N_PAISES_VISITADOS` | int | Unique countries visited |
| `N_CONTINENTES_VISITADOS` | int | Unique continents visited |
| `CAMBIA_PAIS_EN_RUTA` | boolean | Route crosses at least one country border |
| `CAMBIA_CONT_EN_RUTA` | boolean | Route crosses at least one continent |

### Popularity & Hub Scores

| Column | Type | Description |
|---|---|---|
| `POP_RUTA_GLOBAL` | int | Raw route occurrence count |
| `LOG_POP_RUTA` | float | Log-transformed route popularity |
| `POP_OD_PAIR` | int | Raw OD pair occurrence count |
| `LOG_POP_OD` | float | Log-transformed OD popularity |
| `HUB_SCORE_ORIG` | float | Relative importance of origin airport |
| `HUB_SCORE_DEST` | float | Relative importance of destination airport |
| `LOG_HUB_ORIG` | float | Log-transformed origin hub score |
| `LOG_HUB_DEST` | float | Log-transformed destination hub score |

### User Profile Features

| Column | Type | Description |
|---|---|---|
| `U_N_TRIPS` | int | Total trips by user |
| `U_PCT_INTL` | float | % international trips |
| `U_AVG_DIST_OD` | float | Average OD distance (km) |
| `U_MAX_DIST_OD` | float | Maximum OD distance traveled |
| `U_AVG_TRAMOS` | float | Average legs per trip |

### Temporal Features

| Column | Type | Description |
|---|---|---|
| `FECHA_EMISION_LIMPIA` | date | Parsed emission date |
| `ANIO_EMISION` | int | Emission year *(Z-Order key)* |
| `MES_EMISION` | int | Emission month *(Z-Order key)* |
| `DIA_SEMANA_EMISION` | int | Day of week (1=Sun, 7=Sat) |
| `ANTICIPACION_DIAS` | int | Days between booking and departure |
| `ANTICIPACION_BUCKET` | string | Anticipation range bucket |
| `TIENE_ANTICIPACION` | boolean | Anticipation data available |
| `DURACION_VIAJE` | int | Trip duration in days |
| `DURACION_BUCKET` | string | Duration range bucket |
| `TIENE_DURACION` | boolean | Duration data available |

---

## `recsys_gold.recomendaciones_v3`

Top-5 route recommendations per active B2C user.

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Anonymized user identifier |
| `rank` | int | Recommendation position (1–5) |
| `ruta` | string | Recommended IATA route |
| `score` | float | LambdaRank confidence score |
| `model_version` | string | MLflow model version |
| `execution_date` | timestamp | Inference run timestamp |

---

## `recsys_gold.training_logs_v2`

MLOps audit trail — one row per training run.

| Column | Type | Description |
|---|---|---|
| `run_id` | string | MLflow run identifier |
| `timestamp` | timestamp | Training completion time |
| `model_name` | string | Registered model name |
| `model_uri` | string | MLflow artifact URI |
| `ndcg_at_5` | float | Primary evaluation metric |
| `hit_rate_at_5` | float | Hit rate at Top-5 |
| `metrics_json` | string | Full metrics as JSON string |

---

## `recsys_bronze.airports`

Airport metadata catalog used for IATA validation and feature enrichment.

| Column | Type | Description |
|---|---|---|
| `IATA_CODE` | string | 3-letter IATA code |
| `TYPE` | string | `large_airport` / `medium_airport` / ... |
| `SCHEDULED_SERVICE` | string | `yes` / `no` — commercial service flag |
| `ISO_COUNTRY` | string | Country code |
| `CONTINENT` | string | Continent code |
| `LATITUDE_DEG` | float | Latitude |
| `LONGITUDE_DEG` | float | Longitude |
| `ELEVATION_FT` | float | Airport elevation in feet |
