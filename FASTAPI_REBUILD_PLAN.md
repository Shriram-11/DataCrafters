# DataCrafters FastAPI Rebuild Plan

## 1) Goal

Build DataCrafters as a FastAPI-first data processing API where CSV files are handled in an ephemeral workflow:

- Upload CSV
- Process/profile/validate/transform
- Return result
- Delete temporary file/state

No long-term dataset persistence is required for the current product behavior.

## 2) Product Rules (Current Scope)

- No Django usage in the rebuilt service.
- No permanent dataset storage for uploaded CSV files.
- In-memory processing for small files.
- Temporary disk processing for larger files, then immediate cleanup.
- Optional short TTL cleanup job as a safety net for crash/interrupted requests.

## 3) Target Architecture

Use a clean layered structure with minimal persistence:

- API layer: FastAPI routers and response models
- Service layer: CSV processing workflows
- Engine layer: Polars/Pandas/DuckDB operations
- Temp storage layer: per-request temp files only

```text
datacrafters/
  app/
    main.py
    core/
      config.py
      logging.py
      errors.py
      tempfiles.py
    api/
      v1/
        routers/
          health.py
          process.py
          profile.py
          quality.py
          transform.py
    services/
      csv_ingest_service.py
      profile_service.py
      quality_service.py
      transform_service.py
      export_service.py
      cleanup_service.py
    engines/
      polars_engine.py
      duckdb_engine.py
    schemas/
      process_schemas.py
      profile_schemas.py
      quality_schemas.py
      transform_schemas.py
    tests/
      unit/
      integration/
  pyproject.toml
  docker-compose.yml
```

## 4) Core Request Flow

1. Client uploads CSV to an endpoint.
2. API stores file in memory or temp directory based on file size threshold.
3. Processing service runs selected operations.
4. API returns JSON result and optional downloadable artifact.
5. `finally` block removes temp file and temporary generated artifacts.

## 5) Data Lifetime Policy

- Request memory objects: deleted at request completion.
- Temp files: removed immediately after response.
- Safety cleaner: deletes stale temp files older than TTL (for example 15 minutes).
- Logs: keep only operational metadata (duration, rows processed, status), never raw dataset content.

## 6) API Surface (MVP)

- `POST /api/v1/process/csv`
  - Upload + run selected operations in one request.
- `POST /api/v1/profile/csv`
  - Return profile summary.
- `POST /api/v1/quality/csv`
  - Run data quality checks.
- `POST /api/v1/transform/csv`
  - Return transformed data preview and optional output file.
- `GET /api/v1/health`
  - Service health endpoint.

## 7) Implementation Phases

### Phase 0: Foundation Setup (Day 1)

- Create FastAPI app skeleton and routing.
- Add central config, error handling, structured logging.
- Add temp directory manager utilities.
- Define max upload size and in-memory threshold.

Deliverable:

- Running API with health endpoint and shared app structure.

### Phase 1: CSV Ingest + Cleanup (Day 2)

- Implement CSV upload endpoint.
- Add dual strategy:
  - small file -> memory buffer
  - large file -> temp file
- Add guaranteed cleanup in `try/finally`.
- Add startup cleanup for stale temp files.

Deliverable:

- Reliable upload/process/delete lifecycle with no persistent CSV storage.

### Phase 2: Profiling Module (Day 3)

- Build profile service (columns, nulls, uniques, type inference).
- Return compact profile JSON.
- Add edge-case handling (empty file, malformed CSV, wide tables).

Deliverable:

- `POST /profile/csv` with stable profile output.

### Phase 3: Quality Module (Day 4)

- Add rule engine for null checks, ranges, regex, uniqueness.
- Return violations and summary score.
- Make rule payload configurable per request.

Deliverable:

- `POST /quality/csv` with configurable checks.

### Phase 4: Transform Module (Day 5)

- Add transforms: type cast, fill/drop nulls, dedupe, rename/select columns.
- Return preview rows and optional transformed CSV response.
- Ensure transformed temp outputs are removed after response.

Deliverable:

- `POST /transform/csv` with preview + export path.

### Phase 5: Unified Process Endpoint (Day 6)

- Add one-shot endpoint: upload -> profile -> quality -> transform.
- Return combined response contract with timings.
- Add request options to skip sections.

Deliverable:

- `POST /process/csv` for full workflow in one call.

### Phase 6: Hardening + Tests (Day 7)

- Add unit and integration tests.
- Validate cleanup behavior under failures and interrupted requests.
- Add file-size limits and rate limiting.
- Benchmark on small/medium/large CSV sizes.

Deliverable:

- Production-ready MVP behavior for ephemeral CSV processing.

## 8) Definition of Done

- Endpoints process CSV successfully without persistent dataset storage.
- Temp files are always deleted after request or TTL cleaner pass.
- No request path depends on Django components.
- API returns clear error messages for invalid CSV input.
- Tests cover upload, processing, and cleanup guarantees.

## 9) Immediate Next Actions

1. Scaffold Phase 0 structure in code.
2. Implement Phase 1 upload + cleanup with tests.
3. Build Phase 2 profile endpoint as first vertical slice.
