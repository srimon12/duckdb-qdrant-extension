# Roadmap: retrieval in SQL, 1.20.2

Status: implemented on `feat/1.20.2-batch-hybrid-demo`. The decisions below are
the ones the code follows; where implementation differs, the note is inline.

Implemented in this PR besides the two P0 items and hybrid text:

- `qdrant_collection_info` returns `created_at` (Qdrant 1.20); a timestamp before
  the Unix epoch is NULL.
- The `CallError` refactor: failures are classified, the API key never enters a
  message, and transient gRPC statuses are retried once per call policy.
- The scroll refactor: payload-only rows, id cursor, 128-point pages.
- The ESCI demo, docs, unit tests, and the extended live test.

Not implemented: deep batch pages (limit above 128), the H&M act, and
everything in the freeze and never lists.

## Position

The extension is a retrieval bridge. Qdrant ranks; DuckDB joins, aggregates,
verifies. It is not a Qdrant API clone.

Build:

- dense, text (BM25), dense+sparse hybrid, dense+text hybrid;
- recommend, discover, context;
- catalog, scroll, settings, thin upsert.

Freeze:

- `qdrant_upsert` and `qdrant_upsert_one` stay in the code and the live test.
  They are not advertised as a loader and get no new features.

Never:

- facet, group_by, formula/decay/boost, MMR;
- collection or payload-index lifecycle, aliases, cluster admin;
- point delete, payload update, bulk loader, snapshots.

Those stay in Qdrant. The demo shows them through the Qdrant API/UI next to SQL.

## P0: text parameters

`qdrant_text` gains BM25 knobs so it can reproduce a tuned text index:

```
qdrant_text(..., "limit" := 10, k := 1.2, b := 0.75)
```

Append `k` and `b` at the end of the signature. `'NaN'` or 0 keeps the server
default. `TextQueryBuilder::new(text).k(k).b(b)`.

## P0: one batch function

One function, two modes. One RPC per call. Same filter for every query, which is
the shape Qdrant 1.20.2 optimized.

```sql
qdrant_batch(
  collection := 'esci-products',
  queries := '[[0.1,0.2],[0.3,0.4]]',  -- JSON: array of arrays when dense
  mode := 'dense',                      -- dense | text
  "using" := 'dense',                   -- dense mode
  field := 'text',                      -- text mode
  k := 1.2,                             -- text mode
  b := 0.75,                            -- text mode
  "limit" := 10,
  filter := '',
  score_threshold := NaN,
  payload_fields := '',
  url := '', api_key := '', timeout_secs := 0
)
```

Rows: `query_index BIGINT, id VARCHAR, score DOUBLE, payload VARCHAR`.

Rules:

- `queries` is a JSON string. Dense: `[[f64,...],...]`. Text: `["a","b",...]`.
- One page per query. `limit` above 128 is an error in v1; document it.
- Result rows keep server order; `query_index` identifies the request.
- Empty list, non-finite number, wrong JSON, or dimension mismatch is a clear
  SQL error.
- One `query` RPC using `QueryPointsBatch` / `query_batch` with one request per
  query. Do not loop client-side; that defeats the purpose.

## P1: dense + text hybrid

```sql
qdrant_hybrid_text(
  collection := 'esci-products',
  dense_query := [...],
  text_query := 'cashew granola bars',
  fusion := 'rrf',              -- rrf | dbsf
  dense_using := '',
  text_field := 'text',
  prefetch_limit := 100,
  "limit" := 10,
  filter := '', score_threshold := NaN, payload_fields := '',
  url := '', api_key := '', timeout_secs := 0
)
```

Two prefetches: dense vector and `TextQueryBuilder` on the payload field. Fusion
is server-side. No sparse model. This is the 1.20 feature that matters for the
demo.

## P1: tests

- Unit: batch JSON parsing (dense/text), limit > 128 rejection, text k/b placed
  on the request, hybrid text prefetch shape and offset, negative `created_at`.
- Live (`QDRANT_TEST_URL`): batch dense and text parity against single-query
  calls on `(id, score)` with the same k/b; hybrid text returns rows; fused
  paging to 130 distinct rows; 130-row page limit error; `qdrant_text` k/b.
- Existing live test must keep passing.

## Demo: One SQL session

Separate from the bench. Lives in `demo/`. Minimal code, readable.

Primary dataset: `Qdrant/esci-products-multilingual-e5-small-embeddings`
(164,900 products, 384d, judgments, published snapshot). Use
`https://snapshots.qdrant.io/esci-products.snapshot` on Qdrant 1.20.2.

Acts:

1. Restore and inspect: snapshot, `qdrant_collection_info`, payload indexes.
2. Retrieval ladder: filtered dense, BM25, server-side dense+text hybrid.
   Show 5 rows each. BM25 needs no model.
3. Decide: join top-K to DuckDB tables: real judgments plus a clearly labeled
   synthetic stock/margin table. Show candidates dropped and the ranking change.
4. Verify: exact DuckDB scan, nDCG/recall, local duck-vss HNSW. Same vectors,
   same ids. Show the index plan.
5. Boundary: Qdrant facet counts and one formula/decay query through REST/UI,
   then DuckDB `GROUP BY` as the cross-check. Say plainly that these stay in
   Qdrant.

Bench integration: none. The demo is self-contained and must not import
`qdrant-duckdb-bench`.

Deliverables:

- `demo/README.md` with one run path and prerequisites. Done.
- SQL files, `demo/embed.py` for query vectors, `demo/report.py` for a static
  HTML/Markdown report from SQL results. Done; the report is `out/report.md`.
- Re-run the measured numbers on 1.20.2; do not carry forward unverified
  figures. Done; see `demo/README.md` and `docs/real-world.md`.
- H&M (`Qdrant/hm_ecommerce_products`, 105,126 products, 384d BGE + SPLADE,
  facets, images) is the optional visual catalog act. Use it only if ingest and
  query model are already reproducible; otherwise record it as a follow-up.
  Not delivered, see below.
- No notebook state, no hidden steps. Done: `demo/report.py` runs the SQL files
  and writes one Markdown file.

Not delivered in this PR:

- H&M visual catalog act. It needs a reproducible ingest and query model, which
  this change does not add. Recorded as a follow-up in `demo/README.md` and
  `docs/real-world.md`.
- Per-query deep pages in `qdrant_batch` (`limit` above 128). v1 rejects them.
- Vectors out of Qdrant (`with_vector`, retrieve by id), which Act 4/5 do not
  need but a local rerank would.

## Bench contract

The bench stays a separate repo. It may add an extension arm, but this repo does
not depend on it. Parity, overhead, and cross-engine fusion are the bench's job.

## Sources

- Qdrant 1.20.2: https://github.com/qdrant/qdrant/releases/tag/v1.20.2
- Qdrant 1.20.0: https://github.com/qdrant/qdrant/releases/tag/v1.20.0
- Qdrant BM25 on a payload index:
  https://qdrant.tech/documentation/search/text-search/full-text-search/
- Batch queries:
  https://qdrant.tech/documentation/concepts/search/#batch-search
- ESCI dataset:
  https://huggingface.co/datasets/Qdrant/esci-products-multilingual-e5-small-embeddings
- H&M dataset: https://huggingface.co/datasets/Qdrant/hm_ecommerce_products
- duckdb-neo: https://crates.io/crates/duckdb-neo
- DuckDB 2.0 release notes: https://duckdb.org/2026/10/21/announcing-duckdb-200.html
- DuckDB VSS:
  https://duckdb.org/docs/2.0/core_extensions/vss.html
- DuckDB HNSW notes used by this repo:
  `duckdb-2-skills/skills/duckdb-search/references/vss.md`
- Prior work: `qdrant-demo-projects/01-jurisboost-legal-ranking`,
  `02-geosmart-spatial-hospitality`, and `qdrant-duckdb-bench`.

## Open details

- H&M act: include only if reproducible inside the PR. Not included; recorded
  as a follow-up in `demo/README.md` and `docs/real-world.md`.
- `qdrant_batch` text mode: `k`/`b` naming must match `qdrant_text`. They do,
  including the defaults `1.2` and `0.75` and the `'NaN'`/`0` passthrough.
- Batch limit above 128: error in v1. Paging per query is a later change.
  Implemented as an `invalid_input` error before the RPC.
- `qdrant_hybrid_text` paging: it matches `qdrant_hybrid`. Both now set `offset`
  on the fused query and use one prefetch pool for every page, so `limit` above
  128 returns distinct rows. Verified in the live test with 130 rows.
