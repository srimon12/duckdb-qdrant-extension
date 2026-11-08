# DuckDB extension for Qdrant

Query Qdrant from a DuckDB session. DuckDB does the analytics. Qdrant does the retrieval.

Phase 0 registers one table function, `qdrant_query`. It runs a nearest-neighbour query with an optional JSON filter and a limit.

```sql
SELECT * FROM qdrant_query(
  collection := 'articles',
  query := [0.1, 0.2, 0.3]::DOUBLE[],
  limit := 50,
  filter := '{"must":[{"key":"year","range":{"gte":2024}}]}',
  "using" := 'dense',
  score_threshold := 0.75
);
```

| column | type | |
| --- | --- | --- |
| id | VARCHAR | numeric id or UUID |
| score | DOUBLE | |
| payload | VARCHAR | full payload as JSON |

The gRPC URL defaults to `QDRANT_URL`, then `http://localhost:6334`. The API key defaults to `QDRANT_API_KEY`. Pass `url`, `api_key`, or `timeout_secs` on the call to override that. Prefer the environment variable for the key. A key written in SQL stays in the query text.

`SET qdrant_url` is not implemented yet. Hybrid, recommend, and discover are later phases.

The extension loads only in DuckDB `v2.0.0`, and only with `-unsigned`.

```text
make configure
make debug
duckdb -unsigned -cmd "LOAD 'build/debug/extension/qdrant/qdrant.duckdb_extension';"
```

`cargo test --workspace` runs the parser and result-mapping tests. It does not start Qdrant. Wasm builds are not a target: the official client speaks gRPC.
