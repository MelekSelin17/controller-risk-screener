# Data

This directory contains an anonymized version of the industrial dataset used in the paper.
All system-specific identifiers have been replaced with generic placeholders (e.g., `CORP`, `PKG_001`, `TBL_001`, `ctrl_001`); all numeric values (latency measurements, row counts, complexity scores, etc.) are preserved exactly as used in the experiments.

## File descriptions

### `data/raw/sp_metrics.csv`
Stored procedure static analysis metrics. One row per stored procedure.

| Column | Type | Description |
|--------|------|-------------|
| OWNER | str | Database schema owner |
| PACKAGE_NAME | str | Package name |
| SUBPROGRAM_NAME | str | Procedure/function name |
| SOURCE_LINE_COUNT | int | Lines of source code |
| SELECT_COUNT | int | Number of SELECT statements |
| JOIN_COUNT | int | Number of JOIN clauses |
| GROUP_BY_COUNT | int | Number of GROUP BY clauses |
| ORDER_BY_COUNT | int | Number of ORDER BY clauses |
| DML_COUNT | int | Total write operations (INSERT+UPDATE+DELETE+MERGE) |
| LOOP_COUNT | int | Number of loop statements |
| FOR_COUNT | int | Number of FOR loop statements |
| READ_HEAVY_FLAG | int | 1 if no write operations, 0 otherwise |

### `data/raw/sp_table_deps.csv`
Stored procedure to table dependency mapping.

| Column | Type | Description |
|--------|------|-------------|
| OWNER | str | Database schema owner |
| PACKAGE_NAME | str | Package name |
| SUBPROGRAM_NAME | str | Procedure/function name |
| TABLE_NAME | str | Table accessed by the stored procedure |

### `data/raw/table_stats.csv`
Database table statistics (from Oracle DBA_TABLES or equivalent).

| Column | Type | Description |
|--------|------|-------------|
| TABLE_NAME | str | Table name |
| NUM_ROWS | int | Estimated row count |
| AVG_ROW_LEN | int | Average row length in bytes |

### `data/raw/controller_sp_mapping.csv`
Mapping from controller source files to stored procedures.

| Column | Type | Description |
|--------|------|-------------|
| file_name | str | Controller file name (e.g. `OrderController.cs`) |
| sp_names | str | Pipe-separated list of SP keys (`OWNER.PKG.PROC\|...`) |
| dispatch_count | int | Number of dispatch calls |
| sp_call_count | int | Number of SP call sites |

### `data/processed/apm_monthly.csv`
Monthly aggregated APM monitoring data (used only for label construction).

| Column | Type | Description |
|--------|------|-------------|
| controller | str | Controller key (normalized) |
| month | str | Month identifier (YYYY-MM) |
| p95_ms | float | 95th-percentile latency in milliseconds |
| call_count | int | Number of requests in this month |

### `data/processed/sonar_features.csv`
Pre-processed SonarQube static analysis features.

| Column | Type | Description |
|--------|------|-------------|
| controller | str | Controller key (normalized) |
| log_ctrl_complexity | float | log(1 + cyclomatic complexity) |
| log_ctrl_functions | float | log(1 + function count) |
| log_dep_complexity_sum | float | log(1 + total downstream complexity) |
| dep_complexity_per_function | float | Dependency complexity per function |
