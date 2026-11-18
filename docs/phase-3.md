# Phase 3: catalog, scroll, upsert

Phase 3 is the data around search. These functions do not take a query vector and do not return a score.

## Collections

```sql
SELECT name FROM qdrant_collections();

SELECT status, points_count, indexed_vectors_count, segments_count, created_at
FROM qdrant_collection_info(collection := 'articles');
```

`status` is the Qdrant collection status name (`Green`, `Yellow`, `Red`, `Grey`). Counts are the approximate counts Qdrant reports. `created_at` is UTC `YYYY-MM-DDTHH:MM:SSZ` when the server recorded it, and NULL for a collection created before Qdrant 1.20 stored that field.

`url`, `api_key`, and `timeout_secs` override the session settings, same as in phase 2.

## Scroll

Scroll walks points in id order. The row is `id` VARCHAR and `payload` VARCHAR.

```sql
SELECT * FROM qdrant_scroll(
  collection := 'articles',
  "limit" := 100,
  filter := '{"must":[{"key":"year","range":{"gte":2024}}]}',
  "offset" := '10',
  payload_fields := 'year,title'
);
```

`offset` is a point id. It is not a row number and not a named vector. Qdrant starts at that id. The next call can pass the last id from this page when you want the following page yourself. Inside one call, the extension follows Qdrant's next-page id until `limit` is filled, in pages of 128.

`payload_fields` works the same way as in phase 2.

## Upsert

Upsert is frozen. The code stays for tests and small writes, but it is not a
loader and gets no new features: a snapshot or the source pipeline owns bulk
load. It is not update-by-filter, not delete, and not a payload patch.

The batch form sends one RPC:

```sql
SELECT id, status FROM qdrant_upsert(
  collection := 'articles',
  ids := ['1', '2'],
  vectors := [[0.1, 0.2], [0.3, 0.4]]::DOUBLE[][],
  payloads := ['{"year":2024}', '{"year":2025}'],
  "using" := 'dense'
);
```

`ids` and `vectors` must have the same length. An empty `payloads` list writes `{}` for every point. Otherwise `payloads` must match `ids`. `"using"` stores the vector under that name. Omit it for the collection's single unnamed vector. `status` is the Qdrant update status, usually `Completed` or `Acknowledged`.

The scalar form is for a DuckDB table. It uses the session settings and returns the status:

```sql
SELECT qdrant_upsert_one('articles', id, embedding, payload) FROM articles;
```

Each row is its own RPC. Prefer `qdrant_upsert` when the ids and vectors are already lists.

Bulk load is out of scope. Do not grow these functions into a loader: use a
snapshot or the source pipeline, then read the collection through the search
functions.
