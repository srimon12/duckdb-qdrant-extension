# Real-world workloads, Qdrant 1.20, and DuckDB VSS

This is the design note behind `demo/`. It maps the Qdrant releases from 1.18
to 1.20.2 onto this extension, names the gaps that matter for real workloads,
and describes where Qdrant ends and DuckDB begins.

Baseline for this note and the demo:

| Piece | Version | Why |
| --- | --- | --- |
| Qdrant server | `v1.20.2` | Current patch release. BM25, Turbo8/16, batch search fixes. |
| Rust client | `qdrant-client` 1.20.0 | Latest crate; wire-compatible with 1.20.2. |
| DuckDB | `v2.0.0` | The only version the extension loads in. |
| DuckDB VSS | bundled with DuckDB 2.0 | Local HNSW for offline and rerank work. |

## What changed since 1.18, and what it means here

Only changes that alter how SQL should use Qdrant are listed. Internal storage
and cluster work is omitted unless it changes a query or a load path.

### 1.18

- TurboQuant: vectors can live compressed (`turbo8`), roughly 8x smaller.
- Named vectors can be created and deleted on an existing collection.
- Low-memory mode and strict mode: an update can be rejected when memory is
  high.
- RocksDB support removed.

Extension impact: none. Quantization, named vectors, and memory policy are
server-side. Search results and point ids are unchanged.

### 1.19

- TurboQuant 4-bit as the primary vector storage type (`turbo4`).
- Memory tiers per component: `cold`, `cached`, `pinned`.
- `match.prefix` filters.
- `slice` filter for deterministic sampling and sliced scroll.
- Per-tenant IDF for sparse vectors.
- Global quota API; routing token for read affinity.
- Facet performance work.
- Deprecated search endpoints in the OpenAPI schema and in gRPC. The universal
  query API is the forward path; this extension already uses it.

Extension impact: the filter parser already accepts `match.prefix`. `slice`,
shard keys, and routing tokens are not exposed. They are the next filter-level
gaps for multi-tenant and sampling workloads.

### 1.20

- BM25 scoring directly on a full-text payload index. A plain text query is
  ranked by BM25; no sparse vectors and no external sparse model.
- Turbo8 and Turbo16 datatypes.
- Per-element payload index for arrays of objects (`nested` index parameter).
- `code` tokenizer for full-text indexes.
- `match.substring` filter (full scan only).
- Formula query expressions `acosh`, `max`, `min`.
- `created_at` recorded in collection info.
- HNSW `inline_storage`: vectors stored only in the graph file.
- Snapshot robustness: unique names, never overwrite, 4 MiB transfer buffers,
  60 s download timeout, S3/GCS/Azure retry settings.
- Internal auth enforced by default when an API key is configured.

Extension impact: `created_at` is now returned by `qdrant_collection_info`.
`match.prefix` and `match.substring` are already parsed. **BM25 needed a new
read path, so this change adds `qdrant_text`.** Snapshot recovery still runs
over REST, not gRPC.

### 1.20.1

- PiPNN HNSW construction for large segments.
- Much faster filtered reads on cold on-disk payload indexes.
- Batch searches that apply the same filter to several named vectors are
  faster.
- Turbo8/Turbo16 quantization about 2x faster.
- Fixes: HNSW build regression, empty `group_by` results, best-score threshold
  with Euclid distance.

### 1.20.2

- Batch searches with the same filter are significantly faster.
- Classic HNSW construction is the default again, so `ef_construct` tunes
  quality as documented.
- A collection config without `ef_construct` loads, so a downgrade from 1.21
  keeps working.

Extension impact: none directly, but it defines the next function. At 1.20.2 a
batch of queries sharing one filter is a first-class server operation, and
`qdrant_batch` now sends it as one RPC. `qdrant_hybrid_text` fuses a dense
prefetch and a BM25 text prefetch on the server, so the lexical leg needs no
sparse model.

## Capability map

| Capability | In this extension | Notes |
| --- | --- | --- |
| Dense nearest, named vectors | Yes | `qdrant_query` |
| Dense + sparse hybrid, RRF/DBSF | Yes | `qdrant_hybrid`; sparse vectors must be computed by the caller |
| BM25 over a payload text index | Yes, this change | `qdrant_text`; needs Qdrant 1.20+ and an index with `scoring` enabled, with `k`/`b` knobs |
| Dense + text hybrid, server-side RRF/DBSF | Yes, this change | `qdrant_hybrid_text`; dense prefetch plus BM25 text prefetch, no sparse model |
| Batch queries, one shared filter | Yes, this change | `qdrant_batch`; dense or text, one page per query, limit at most 128 |
| Recommend, discover, context | Yes | `qdrant_recommend`, `qdrant_discover`, `qdrant_context` |
| Filters: match, range, nested, prefix, substring, min_should | Yes | Filter JSON; unknown keys error |
| Score threshold, payload selection, paging past 128 | Yes | Same args on every search; hybrid fusions page with `offset` and a prefetch floor equal to the limit, stable in tests but not tie-ordered by contract |
| Collection list and info | Yes | `created_at` included |
| Scroll | Yes, payload only | No vectors, no bulk export |
| Upsert | Frozen | Code and tests stay; not advertised as a loader, no new features |
| Point delete, payload set/delete | No | |
| Collection create/delete, aliases, payload indexes | No | |
| Count | No | Use `qdrant_collection_info` for an approximate count |
| Facets, grouping, formula queries, MMR | No | DuckDB can group and fuse locally |
| Retrieve by id with vectors | No | Blocks local mirror/rerank without a second data source |
| Snapshots (create/list/restore) | No | Restore is REST-only in Qdrant; the demo uses a small Python script |
| Shard keys, routing token, quotas | No | Server config mostly; shard key is a request field |

## Workloads

### 1. RAG and hybrid search

Shape: embed a query, retrieve with a dense vector, add lexical evidence, fuse,
then filter and join with business data.

Today:

```sql
WITH dense AS (
  SELECT id, row_number() OVER (ORDER BY score DESC) AS r
  FROM qdrant_query(collection := 'docs', query := [...], "limit" := 50)
),
lex AS (
  SELECT id, row_number() OVER (ORDER BY score DESC) AS r
  FROM qdrant_text(collection := 'docs', query := 'refund policy', field := 'text', "limit" := 50)
)
SELECT coalesce(d.id, l.id) AS id,
       coalesce(1.0 / (60 + d.r), 0) + coalesce(1.0 / (60 + l.r), 0) AS rrf
FROM dense d FULL OUTER JOIN lex l USING (id)
ORDER BY rrf DESC
LIMIT 10;
```

This is a real two-engine split: Qdrant scores, DuckDB fuses and joins. The
server-side alternative is `qdrant_hybrid_text`: a dense prefetch plus a BM25
text prefetch, fused by Qdrant with RRF or DBSF. Both paths are available, so
keep the SQL fusion when the two ranked lists are useful on their own, and use
`qdrant_hybrid_text` when one fused top-K is all you need.

### 2. E-commerce filtered search

Shape: ANN inside a category or brand, plus facet counts, plus deduplication.

- Filtered ANN works today. Qdrant 1.20 adds substring matching, but it is a
  full scan, so do not use it as the primary filter on a large collection.
- Facets and `GROUP BY` are better in DuckDB when the payload is available
  locally. Duplicating payloads into a local table is a normal pattern for
  catalog data; recomputing it through `qdrant_scroll` at 128 points per RPC is
  not.
- Duplicate product variants need grouping. `group_by` is server-side only and
  not exposed. DuckDB window functions over the returned payload work for the
  first page; true group-aware pagination needs the server.

### 3. Recommendations

Recommend, discover, and context are all supported, including mixed id and
vector examples. Missing pieces are MMR (diversity), group-by, and formula
queries. When diversity is a hard requirement, either call Qdrant from another
client or rerank the returned candidate set in SQL.

### 4. Multi-tenant

Tenant isolation is a `must` filter on a keyword field. Add a tenant argument
to every call. For large tenants, a shard key or a collection per tenant
performs better; neither shard key nor routing token is exposed. A per-call
`shard_key` argument is a small, high-value addition.

### 5. Evaluation and offline analysis

Shape: many queries, judged products, nDCG, recall, latency.

- Qdrant computes scores; DuckDB computes nDCG, recall, intersections, and
  joins against judgments. The demo does exactly this.
- One `qdrant_batch` call carries many queries and one filter. A 10,000-query
  sweep is a handful of RPCs instead of 10,000. `limit` is capped at 128 per
  query in v1, so deep pages still need a loop.
- Exact comparison (ground truth) is a DuckDB scan. At 165k x 384 it is cheap;
  at 10M x 1024 it is not, so sample.

### 6. Bulk load and snapshots

- `qdrant_upsert_one` is one RPC per row. It is fine for hundreds of rows and
  wrong for millions.
- `qdrant_upsert` takes lists in one statement; building huge list literals in
  SQL is awkward and memory-bound.
- A snapshot is the practical load path for an existing corpus. Restore is
  REST-only (`PUT /collections/{name}/snapshots/recover`), so the demo uses a
  script and then reads through the extension.
- A streaming table-in/table-out upsert is the correct long-term shape: read a
  DuckDB chunk, send a batch, repeat.

### 7. Operations

Collection status, counts, segment count, and `created_at` are visible.
Snapshot creation/deletion and collection lifecycle are not. These belong in a
separate operations phase; they do not affect search.

## Best of both worlds

Three tiers, one dataset, shared ids:

```text
                    HF snapshot / parquet
                     /                \
             Qdrant 1.20.2        DuckDB 2.0
             HNSW + filters       typed columns + SQL
             BM25 payload index   joins, aggregates, nDCG
                     \                /
                  same point ids, same embedding model
                          |
                    duck-vss HNSW
                    offline / edge / CI
```

| Need | Use |
| --- | --- |
| Low-latency filtered ANN at scale | Qdrant |
| Quantization, memory tiers, sharding | Qdrant |
| BM25 ranking with server-managed stats | Qdrant (`qdrant_text`) |
| Joins with orders, stock, users | DuckDB |
| nDCG, recall, exact ground truth | DuckDB |
| Offline, CI, edge, no server | duck-vss on a local table |
| Rerank a candidate set exactly | DuckDB (needs vectors locally) |

Rules that make the split work:

1. Keep ids stable across both sides. Numeric ids are decimal strings; UUIDs
   stay text.
2. One embedding model, one prefix convention. E5 needs `query: ` for queries
   and `passage: ` for documents. A mismatch silently degrades recall.
3. Do not copy vectors through `qdrant_scroll`; it returns payload only. Use
   the source parquet or a future vector-returning function.
4. Use Qdrant for candidate generation and DuckDB for truth. Exact distance on
   the full corpus is the ground truth, not another ANN.
5. When fusing, use RRF on ranks, not raw score sums. BM25 and cosine scores
   are not comparable.

### duck-vss constraints that change the SQL

From the DuckDB search notes (branch `audit/retrieval-follow-up` of
`duckdb-2-skills`):

- Index key must be exactly `FLOAT[N]`. `DOUBLE[]` and `FLOAT[]` fail.
- The query vector must be a literal `FLOAT[N]` for the index to accelerate.
  A `DOUBLE[N]` cast or a scalar subquery falls back to a scan.
- Match the metric: `cosine` index with `array_cosine_distance`, `l2sq` with
  `array_distance`, `ip` with `array_negative_inner_product`. A mismatch is a
  silent scan.
- `hnsw_prefilter` defaults to true. Do not disable it on selective filters.
- On DuckDB 2.0.0 do not paginate with `OFFSET`; the HNSW scan repeats the
  first page. Fetch `k + offset` and slice.
- Persistence is experimental: set `hnsw_enable_experimental_persistence`
  after `LOAD vss`, and keep a backup. The index is not buffer-managed, so it
  does not count toward `memory_limit`.
- Deletes are marked, not removed. Rebuild after a large delete.
- Reject NULL, zero, and non-finite vectors before insert. NULL never enters
  the index and shortens a `LIMIT k` page without an error.

## Demo

`demo/` restores Qdrant's published ESCI snapshot, queries it through this
extension with the same model that produced the dataset, and compares the
result with an exact DuckDB scan and a local HNSW index.

- Dataset: `Qdrant/esci-products-multilingual-e5-small-embeddings`.
- Snapshot: `https://snapshots.qdrant.io/esci-products.snapshot`, Qdrant 1.20+.
- Corpus: 164,900 Amazon ESCI products, 384-dimensional multilingual-e5-small.
- Judgments: 8,956 queries, 181,701 labels (`E`, `S`, `C`, `I`).
- Model: `intfloat/multilingual-e5-small`, queried with the `query: ` prefix.

The demo measures:

- nDCG@10 of dense retrieval inside Qdrant.
- nDCG@10 of BM25 over the snapshot's `text` index.
- nDCG@10 after DuckDB-side RRF fusion of dense and BM25.
- nDCG@10 of server-side dense + text fusion through `qdrant_hybrid_text`.
- Recall@10 of Qdrant dense retrieval against an exact DuckDB scan.
- HNSW plan and latency for one literal query in duck-vss, against the same
  rows used by the exact scan.

The dense and BM25 runs go through `qdrant_batch`: fifty queries per RPC,
`query_index` maps a row back to its query.

The acts match the brief:

1. Restore and inspect: `demo/qdrant_setup.py` prints points, vectors, payload
   indexes, and the BM25 marker; `demo/inspect.sql` reads `qdrant_collection_info`
   and a payload sample back through SQL.
2. Retrieval ladder: `out/topk.sql`, `out/text_topk.sql`, and `out/hybrid.sql`.
3. Decide: `demo/decide.sql` joins server-side hybrid top-10 to real judgments
   and to a synthetic, hash-derived stock/margin table, then reports candidates
   kept or dropped and nDCG@10 before and after. The synthetic table is labeled
   as such in every output.
4. Verify: `out/exact.sql`, `demo/fuse.sql`, `demo/evaluate.sql`, and
   `out/hnsw.sql`.
5. Boundary: `demo/boundary.py` calls Qdrant facets and a `$score + boost`
   formula over REST; `demo/groupby.sql` is the DuckDB `GROUP BY` cross-check.
   The measured facet counts match exactly. These features stay in Qdrant.

`demo/report.py` runs the result SQL and writes a static `out/report.md`.
`demo/smoke.sql` binds all three new functions through DuckDB so argument-order
mistakes fail fast.

## Roadmap

Ordered by impact per unit of work, not by wishlist.

1. ~~`qdrant_batch` — many query vectors, one filter, one RPC.~~ Done in this
   change. Deep pages per query (limit above 128) are still a follow-up.
2. **Vectors out** — `with_vector` on `qdrant_scroll`, and a retrieve-by-id
   function. Unlocks mirroring, rerank, and DuckDB HNSW without a second copy
   of the source data.
3. ~~Dense + BM25 hybrid — server-side RRF with no sparse model.~~ Done in this
   change as `qdrant_hybrid_text`.
4. **`shard_key` and routing** — per-call args for multi-tenant read/write.
5. **`qdrant_count`** — exact count with a filter; cheap and useful for
   ingestion checks.
6. **Facets and group_by** — only if local mirrors are not acceptable.
7. **Snapshot list/create** — gRPC supports them; recovery stays REST.
8. **Collection and payload-index lifecycle** — separate ops phase.
9. **H&M catalog act** — the optional visual act in `demo/`; needs a
   reproducible ingest and query model before it can join the demo.

Not planned: cluster administration, formula queries, MMR, and streaming
upsert. The first three are better handled by a dedicated client; streaming
upsert is a larger API change and should be designed with the batch function.

## Verification status

`cargo test -p qdrant_core --lib` covers the request builders, filter parser,
timestamp formatting, and result mapping, including batch JSON parsing and the
hybrid text prefetch with its offset. The live test exercises Qdrant 1.20 only
when `QDRANT_TEST_URL` points at a gRPC endpoint; it includes a BM25 index,
`qdrant_text` with `k`/`b`, `qdrant_batch` dense and text parity against
single-query calls on `(id, score)`, `qdrant_hybrid_text`, fused paging to 130
distinct rows, and the 128-row batch limit error.

The demo ran end to end on Windows: Qdrant `v1.20.2` in Docker (WSL2), DuckDB
`v2.0.0` CLI, the extension built from this tree, and the published ESCI
snapshot (164,900 points). With the first 50 judged queries, re-measured on
1.20.2:

| Run | nDCG@10 |
| --- | --- |
| hybrid, `qdrant_hybrid_text` RRF, prefetch 100 | 0.2962 |
| BM25, `qdrant_batch` text mode | 0.2858 |
| RRF in DuckDB over two top-50 lists | 0.2846 |
| Dense, `qdrant_batch` dense mode | 0.2396 |

Dense recall@10 against the exact DuckDB scan was 0.868. `EXPLAIN ANALYZE` on the
literal query plans the local `products_hnsw` index. The exact scan is a
minute-scale job at 165k x 384.

The synthetic business rule in Act 3 keeps 365 of 500 candidates and moves
nDCG@10 to 0.232. The server-side facet counts for `brand` and `color` match
the DuckDB `GROUP BY` exactly.

Batch parity: the same 50 dense queries, limit, and payload fields returned
identical `(query_id, id, score)` rows as 50 separate `qdrant_query` calls. That
is one RPC against 50; the demo does not time it.

Fused paging: `qdrant_hybrid_text` with `"limit" := 150` on esci-products
returned 150 rows and 150 distinct ids. The fix is a per-page `offset` on the
fused query plus one prefetch pool for every page. Results were stable in these
tests; the order of fused score ties is not a Qdrant guarantee, so a tie on a
page boundary can still repeat or skip a point.

One finding from the run: DuckDB parses named arguments whose names are SQL
keywords only when quoted. `"using"`, `"limit"`, and `"offset"` need quotes;
`limit :=` is a syntax error. All examples here and in the demo use the quoted
form.
