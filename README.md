# DuckDB extension for Qdrant

Qdrant retrieves. DuckDB filters, joins, and aggregates the rows. One SQL session does both.

The extension speaks gRPC through `qdrant-client` 1.20. It targets Qdrant 1.20.2 and loads only in DuckDB `v2.0.0`, and only with `-unsigned`.

```text
make configure
make debug
duckdb -unsigned -cmd "LOAD 'build/debug/extension/qdrant/qdrant.duckdb_extension';"
```

Wasm is not a target.

## Settings

A named argument on a call overrides the setting for that call. `timeout_secs := 0` means "use the setting".

```sql
SET qdrant_url = 'http://localhost:6334';
SET qdrant_api_key = '';
SET qdrant_timeout_secs = 30;
SET qdrant_with_payload = true;
```

`qdrant_url` and `qdrant_api_key` default to `QDRANT_URL` and `QDRANT_API_KEY` at load. An empty key sends no key. The extension does not write the key into errors. A key typed in SQL stays in the query text, so prefer the setting.

Search rows are always `id` VARCHAR, `score` DOUBLE, and `payload` VARCHAR. `id` is the decimal form of a numeric point id, or the UUID text. `payload` is one JSON object. `qdrant_batch` puts `query_index` BIGINT first, because one call answers many queries.

## Phases

| Phase | What it is | What it is not |
| --- | --- | --- |
| 0-1 | Nearest search, settings, errors, hybrid, named vectors | Recommend, scroll, writes |
| 2 | More retrieval on the same row shape | A write, a collection listing, or a scroll |
| 3 | Catalog, scroll, and a thin upsert | Another search mode |
| 1.20 | `qdrant_text`: BM25 on a payload index | A vector query |
| 1.20.2 | `qdrant_batch`, `qdrant_hybrid_text`: one batch RPC, server-side dense+text fusion | A sparse model |

Full SQL for phase 2 is in [docs/phase-2.md](docs/phase-2.md). Full SQL for phase 3 is in [docs/phase-3.md](docs/phase-3.md). The workload design and a runnable end-to-end example are in [docs/real-world.md](docs/real-world.md) and [demo/](demo/).

### Phase 0–1

`qdrant_query` is nearest neighbours. `qdrant_hybrid` prefetches one dense vector and one sparse vector, then fuses them with `rrf` or `dbsf`. `"using"` is the named vector, and it needs quotes because `USING` is a SQL keyword. `"limit"` needs quotes for the same reason. `"offset"` appears only on `qdrant_scroll`, where it is a point id, and needs quotes too.

```sql
SELECT * FROM qdrant_query(
  collection := 'articles',
  query := [0.1, 0.2, 0.3]::DOUBLE[],
  "limit" := 50,
  filter := '{"must":[{"key":"year","range":{"gte":2024}}]}',
  "using" := 'dense'
);

SELECT * FROM qdrant_hybrid(
  collection := 'docs',
  dense_query := [0.1, 0.2]::DOUBLE[],
  sparse_query := '{"indices":[1,42],"values":[0.22,0.8]}',
  fusion := 'rrf',
  dense_using := 'dense',
  sparse_using := 'sparse'
);
```

Filters use the Qdrant JSON shape: `must`, `should`, `must_not`, `min_should`, field `match`, field `range`, `has_id`, `is_empty`, `is_null`, `has_vector`, and `nested`. An unknown field is an error. It is not dropped.

### Phase 2

Phase 2 is still a read. It adds the other universal-query forms, plus the controls that apply to every search.

- `qdrant_recommend` moves toward positive examples and away from negative ones. There is no target point.
- `qdrant_discover` starts at a target vector and uses context pairs as the direction.
- `qdrant_context` uses those pairs and has no target.
- `score_threshold` drops weaker hits.
- `payload_fields` keeps only the named payload keys. `[]` returns `{}`.
- A limit above 128 is fetched in pages of 128, then streamed into DuckDB chunks. `qdrant_hybrid` pages the server-side fused result with `offset` and raises its prefetch floor to the requested limit. Results were stable in tests; Qdrant does not contractually guarantee the order of fused score ties across pages.

```sql
SELECT id, score, payload->>'year'
FROM qdrant_recommend(
  collection := 'articles',
  positive := '[1, 2]',
  negative := '["3"]',
  strategy := 'average',
  "using" := 'dense',
  score_threshold := 0.2,
  payload_fields := 'year'
);
```

`strategy` is `average` (average vectors), `best` (best score), or `sum` (sum of scores). Positive input is an id list, a vector, or both. The same is true for negative input. At least one positive example is required.

### Phase 3

Phase 3 does not score vectors.

- `qdrant_collections` returns collection names.
- `qdrant_collection_info` returns status, point counts, segment count, and `created_at` (UTC, from Qdrant 1.20).
- `qdrant_scroll` walks points in id order. The row is `id`, `payload`. `offset` is a point id to start from, not a vector name and not a row number.

The upsert pair is frozen. `qdrant_upsert` and `qdrant_upsert_one` stay in the
code for tests and small writes. They are not a loader: a snapshot or the source
pipeline owns bulk load.

```sql
SELECT * FROM qdrant_collection_info(collection := 'articles');
SELECT * FROM qdrant_scroll(collection := 'articles', "limit" := 100, "offset" := '10');
```

#### Lexical (Qdrant 1.20)

`qdrant_text` ranks a payload field by BM25. Qdrant 1.20 computes the score from
the full-text index, so there is no sparse vector and no sparse model. The field
needs a text index with scoring enabled:

```http
PUT /collections/articles/index?wait=true
{"field_name": "text", "field_schema": {"type": "text", "scoring": true}}
```

```sql
SELECT id, score, payload->>'title'
FROM qdrant_text(
  collection := 'articles',
  query := 'cashew granola bars',
  field := 'text',
  "limit" := 10,
  payload_fields := 'product_id,title,brand'
);
```

`field` defaults to `text`. The argument is a payload field name, not a vector
name. `filter`, `score_threshold`, `payload_fields`, and the connection
overrides work the same as on `qdrant_query`. The row is `id`, `score`,
`payload`.

`k` and `b` are the BM25 knobs. They default to `1.2` and `0.75`, which are also
Qdrant's defaults. Pass `'NaN'` or `0` to send no value and let the server's
scorer defaults apply. Named calls that predate the knobs keep working:

```sql
SELECT id, score, payload->>'title'
FROM qdrant_text(
  collection := 'articles',
  query := 'cashew granola bars',
  field := 'text',
  k := 1.5,
  b := 0.5,
  "limit" := 10
);
```

BM25 scores are unbounded and do not share a scale with cosine similarity.
Never compare the two numbers directly. Fuse by rank.

Dense and lexical ranks are not comparable. Fuse them by rank in SQL:

```sql
WITH dense AS (
  SELECT id, row_number() OVER (ORDER BY score DESC) AS r
  FROM qdrant_query(collection := 'articles', query := [...], "limit" := 50)
),
lex AS (
  SELECT id, row_number() OVER (ORDER BY score DESC) AS r
  FROM qdrant_text(collection := 'articles', query := 'cashew granola', "limit" := 50)
)
SELECT coalesce(d.id, l.id) AS id,
       coalesce(1.0 / (60 + d.r), 0) + coalesce(1.0 / (60 + l.r), 0) AS rrf
FROM dense d FULL OUTER JOIN lex l USING (id)
ORDER BY rrf DESC
LIMIT 10;
```

Or let Qdrant fuse the two legs on the server:

```sql
SELECT id, score, payload->>'title'
FROM qdrant_hybrid_text(
  collection := 'articles',
  dense_query := [...]::DOUBLE[],
  text_query := 'cashew granola bars',
  fusion := 'rrf',
  dense_using := 'dense',
  text_field := 'text',
  prefetch_limit := 100,
  "limit" := 10
);
```

`qdrant_hybrid_text` runs two prefetches, one dense vector and one
`TextQueryBuilder` on the payload field, and fuses them with `rrf` or `dbsf`.
There is no sparse model and no sparse vector. `text_field` names the payload
field, and that field needs the BM25 index above. The SQL fusion above stays
useful when you want the two ranked lists in DuckDB. A `limit` above 128 is
paged with `offset`, and the prefetch floor rises to the requested limit. Results
were stable in tests; the order of fused score ties is not a Qdrant guarantee.

#### Batch (Qdrant 1.20.2)

`qdrant_batch` sends many queries in one `query_batch` RPC, with one filter for
all of them. Qdrant 1.20.2 optimized exactly this shape. Rows are
`query_index BIGINT`, `id`, `score`, `payload`.

`queries` is a JSON string. Dense mode is an array of vectors, text mode is an
array of query strings. `mode` is `dense` or `text`. `using` names the vector in
dense mode; `field`, `k`, and `b` name the payload field and BM25 knobs in text
mode. The two modes reject each other's argument: `using` in text mode and a
non-default `field` in dense mode are errors, not ignored.

```sql
SELECT query_index, id, score, payload->>'year'
FROM qdrant_batch(
  collection := 'articles',
  queries := '[[0.1,0.2,0.3],[0.4,0.5,0.6]]',
  mode := 'dense',
  "using" := 'dense',
  "limit" := 10,
  payload_fields := 'year,title'
);

SELECT query_index, id, score, payload->>'title'
FROM qdrant_batch(
  collection := 'articles',
  queries := '["cashew granola bars","protein bars"]',
  mode := 'text',
  field := 'text',
  k := 1.2,
  b := 0.75,
  "limit" := 10
);
```

`limit` is one page per query, so it must be 128 or less. A larger value is an
error, not a paged scan. That is the v1 rule; paging per query is a later change.
The rows keep the server order inside each `query_index`.

A query with no hits contributes no rows. Join against your own query list with a
`LEFT JOIN` when every query id must survive:

```sql
SELECT q.query_id, b.id, b.score
FROM (VALUES (0, 114), (1, 2891)) AS q(query_index, query_id)
LEFT JOIN qdrant_batch(collection := 'articles', queries := '[[0.1,0.2],[0.3,0.4]]',
                       mode := 'dense', "using" := 'dense', "limit" := 10) b
  ON b.query_index = q.query_index;
```

## Out of scope

Collection create and delete, point delete, payload updates, shard and cluster admin, grouping, facets, formula queries, MMR, and returning vectors from a search or scroll. Those stay in Qdrant or need a later phase.

## Tests

`cargo test --workspace` runs the parser and timestamp tests. It skips the live test unless `QDRANT_TEST_URL` is the gRPC address.

```text
$env:QDRANT_TEST_URL = 'http://127.0.0.1:6334'
cargo test -p qdrant_core --lib live_qdrant_120
```

The live test creates a throwaway collection, runs phase 2, phase 3, BM25,
`qdrant_batch` dense and text parity against the single-query calls, and
`qdrant_hybrid_text`, then deletes the collection. It also checks fused paging
past 128 rows and that a batch limit above 128 fails.
