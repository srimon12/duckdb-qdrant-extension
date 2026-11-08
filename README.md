# DuckDB extension for Qdrant

Query Qdrant from a DuckDB session. Qdrant retrieves. DuckDB does the analytics.

The extension loads only in DuckDB `v2.0.0`, and only with `-unsigned`.

```text
make configure
make debug
duckdb -unsigned -cmd "LOAD 'build/debug/extension/qdrant/qdrant.duckdb_extension';"
```

## Settings

Set these before the query. A function argument overrides the setting for that call. `timeout_secs := 0` means "use the setting".

```sql
SET qdrant_url = 'http://localhost:6334';
SET qdrant_api_key = 'xxx';
SET qdrant_timeout_secs = 30;
SET qdrant_with_payload = true;
```

The defaults for the URL and API key are `QDRANT_URL` and `QDRANT_API_KEY` at extension load. An empty API key sends no key. The key is not written to logs. A key passed in SQL stays in the query text, so prefer the setting or the environment variable.

## Queries

Rows from search functions are `id` VARCHAR, `score` DOUBLE, and `payload` VARCHAR JSON. Point ids are decimal strings or UUIDs.

```sql
SELECT * FROM qdrant_query(
  collection := 'articles',
  query := [0.1, 0.2, 0.3]::DOUBLE[],
  limit := 50,
  filter := '{"must":[{"key":"year","range":{"gte":2024}}]}',
  "using" := 'dense',
  score_threshold := 0.75,
  payload_fields := '["year","title"]'
);
```

`payload_fields` is a JSON array or a comma-separated list. `[]` drops the payload. An empty value follows `qdrant_with_payload`.

Filters accept the Qdrant REST shape: `must`, `should`, `must_not`, `min_should`, field `match`, field `range`, `has_id`, `is_empty`, `is_null`, `has_vector`, and `nested`. Any other field is an error.

```sql
SELECT * FROM qdrant_hybrid(
  collection := 'docs',
  dense_query := [0.1, 0.2]::DOUBLE[],
  sparse_query := '{"indices":[1,42],"values":[0.22,0.8]}',
  fusion := 'rrf',          -- or dbsf
  prefetch_limit := 100,
  dense_using := 'dense',
  sparse_using := 'sparse'
);

SELECT * FROM qdrant_recommend(
  collection := 'articles',
  positive := '[1, 2]',
  negative := '["3"]',
  strategy := 'average'     -- average, best, or sum
);

SELECT * FROM qdrant_discover(
  collection := 'articles',
  target := [0.1, 0.2]::DOUBLE[],
  context := '[{"positive":"1","negative":"2"}]'
);

SELECT * FROM qdrant_context(
  collection := 'articles',
  context := '[{"positive":[0.1,0.2],"negative":[0.3,0.4]}]'
);
```

Large limits are requested in pages of 128 and streamed into DuckDB chunks. Transient gRPC failures are retried three times.

## Scroll, collections, upsert

```sql
SELECT * FROM qdrant_scroll(collection := 'articles', limit := 100, offset := '10');
SELECT * FROM qdrant_collections();
SELECT * FROM qdrant_collection_info(collection := 'articles');

SELECT * FROM qdrant_upsert(
  collection := 'articles',
  ids := ['1', '2'],
  vectors := [[0.1, 0.2], [0.3, 0.4]]::DOUBLE[][],
  payloads := ['{"year":2024}', '{"year":2025}']
);

SELECT qdrant_upsert_one('articles', id, embedding, payload) FROM articles;
```

`qdrant_scroll` returns `id` and `payload`. `offset` is a point id to start from. `qdrant_upsert_one` uses the session settings and returns the update status.

## Tests

`cargo test --workspace` runs the parser tests. Set `QDRANT_TEST_URL` to the gRPC address to run the live test against Qdrant 1.20:

```text
$env:QDRANT_TEST_URL = 'http://127.0.0.1:16334'
cargo test -p qdrant_core --lib live_qdrant_120
```

Wasm is not a target. The client speaks gRPC.
