# Phase 2: more retrieval

Phase 2 does not write and does not list collections. Every function here returns the search row: `id` VARCHAR, `score` DOUBLE, `payload` VARCHAR JSON.

The same tail applies to `qdrant_query`, `qdrant_hybrid`, and the three functions below.

| Argument | Meaning |
| --- | --- |
| `limit` | Maximum hits. Values above 128 are fetched in pages of 128. |
| `filter` | Qdrant JSON filter. Unknown fields error. |
| `using` | Named vector. Write it as `"using"` because `USING` is reserved. |
| `score_threshold` | Drop hits worse than this score. `'NaN'` means no threshold. |
| `payload_fields` | JSON array or comma-separated keys. Empty follows `qdrant_with_payload`. `[]` returns `{}`. |
| `url`, `api_key`, `timeout_secs` | Override the session settings. `timeout_secs := 0` keeps the setting. |

## Recommend

Recommend has examples, not a target.

- `positive` is a JSON array of point ids. `positive_vector` is an extra dense example.
- `negative` and `negative_vector` are examples to move away from.
- At least one positive id or the positive vector is required.
- `strategy` is `average`, `best`, or `sum`.

`average` averages the examples into one vector. `best` keeps the strongest example score. `sum` adds positive scores and subtracts negative scores.

```sql
SELECT * FROM qdrant_recommend(
  collection := 'articles',
  positive := '[1, 2]',
  negative := '["9"]',
  positive_vector := [0.1, 0.2, 0.0, 0.0]::DOUBLE[],
  strategy := 'best',
  "using" := 'dense',
  score_threshold := 0.2,
  payload_fields := '["year"]',
  "limit" := 10
);
```

## Discover

Discover has a target and context pairs. Each pair is one positive example and one negative example. The target is the point the search starts from. The pairs say which way to move.

A pair value is a point id or a JSON array of numbers.

```sql
SELECT * FROM qdrant_discover(
  collection := 'articles',
  target := [0.1, 0.2, 0.3]::DOUBLE[],
  context := '[
    {"positive":"1","negative":"2"},
    {"positive":[0.0,1.0,0.0],"negative":[0.0,0.0,1.0]}
  ]',
  "using" := 'dense',
  "limit" := 10
);
```

## Context

Context is discover without a target. Only the pairs remain.

```sql
SELECT * FROM qdrant_context(
  collection := 'articles',
  context := '[{"positive":"1","negative":"2"}]',
  "using" := 'dense',
  "limit" := 10
);
```

## Paging

Qdrant returns one page per RPC. This extension asks for at most 128 hits per RPC and repeats with `offset` until `limit` is filled or a page comes back short. DuckDB then reads those rows in its own chunks. You do not pass the page size.

`qdrant_hybrid` and `qdrant_hybrid_text` follow the same rule. The fused result is paged with `offset`, and the prefetch floor is the requested `limit`, so every page is scored against the same candidate pool. Without that floor, the fused order would shift between pages and ids would repeat.
