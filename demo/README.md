# Real-world demo: ESCI products, Qdrant 1.20.2, DuckDB 2.0

One corpus, one embedding model, two engines:

- **Qdrant 1.20.2** serves filtered dense ANN and BM25 over the payload index.
- **DuckDB 2.0** joins, aggregates, computes nDCG and recall, and holds a local
  duck-vss HNSW index for offline work.
- **Same ids and same model** on both sides, so results are comparable row by
  row.

Corpus: [`Qdrant/esci-products-multilingual-e5-small-embeddings`](https://huggingface.co/datasets/Qdrant/esci-products-multilingual-e5-small-embeddings),
164,900 Amazon ESCI products, 384-dimensional `intfloat/multilingual-e5-small`
embeddings, plus 8,956 judged queries.

- Qdrant publishes a ready-made snapshot: `https://snapshots.qdrant.io/esci-products.snapshot`.
  It includes the payload indexes, including BM25 on `text`, and needs Qdrant
  1.20 or later.
- The Hugging Face dataset publishes the same rows as parquet. That copy feeds
  the DuckDB side. Joining on `id`/`product_id` is valid because both come from
  the same generation run.

Snapshot recovery is REST-only in Qdrant, so it uses a small script. Every
query after it goes through the extension.

## Prerequisites

- Docker with Compose
- DuckDB **v2.0.0** CLI (the extension loads only in that version)
- Python 3.10+ and `uv` (`pip install uv`)
- About 4 GB free: 460 MB parquet, the snapshot, the Qdrant volume, the local
  DuckDB file, PyTorch, and the embedding model
- Build tools for the extension (`make configure && make debug`)

## 1. Start Qdrant 1.20.2

```bash
docker compose -f demo/compose.yaml up -d
```

## 2. Restore the published snapshot

```bash
python demo/qdrant_setup.py
```

Defaults: REST `http://localhost:6333`, collection `esci-products`, snapshot
`https://snapshots.qdrant.io/esci-products.snapshot`. The script waits for
readiness, calls `PUT /collections/esci-products/snapshots/recover`, then polls
until the collection is `green` and prints counts, vectors, and payload
indexes.

Useful flags:

```bash
python demo/qdrant_setup.py --force             # recover over existing points
python demo/qdrant_setup.py --skip-recover      # report only
python demo/qdrant_setup.py --snapshot file:///data/other.snapshot --collection other
```

Qdrant downloads an `https` snapshot itself. To download it to disk first, use
the Compose mount (`demo/snapshots` on the host is `/snapshots` in the
container):

```bash
mkdir -p demo/snapshots
curl -L -o demo/snapshots/esci.snapshot https://snapshots.qdrant.io/esci-products.snapshot
python demo/qdrant_setup.py --snapshot file:///snapshots/esci.snapshot
```

With a native Qdrant process, pass the host path instead, for example
`file:///C:/data/esci.snapshot`.

Expect `dense=384 Cosine` and a `text` payload index. The `text` index has
BM25 scoring enabled; that is what `qdrant_text` uses.

## 3. Embed the evaluation queries

```bash
uv run --with sentence-transformers --with pyarrow demo/embed.py --limit 50
```

This downloads `queries.jsonl` from the dataset, embeds the first 50 judged
queries with the dataset's own model using the `query: ` prefix, and writes the
files below. FastEmbed does not ship `intfloat/multilingual-e5-small`, so this
uses sentence-transformers and downloads PyTorch on the first run.

| File | Contents |
| --- | --- |
| `out/queries.parquet` | `query_id`, `query`, `embedding` |
| `out/judged.parquet` | `query_id`, `product_id`, gain (`E`=1, `S`=0.1, `C`=0.01, `I`=0) |
| `out/topk.sql` | dense top-10 per query, one `qdrant_batch` RPC |
| `out/text_topk.sql` | BM25 top-10 per query, one `qdrant_batch` RPC |
| `out/hybrid.sql` | dense + BM25 fused on the server with `qdrant_hybrid_text` |
| `out/exact.sql` | exact top-10 per query from the local `products` table |
| `out/hnsw.sql` | one literal query for the duck-vss index |
| `out/demo.sql` | readable dense, BM25, and hybrid results for three queries |

The 50 dense queries go to Qdrant in one `query_batch` RPC, and the 50 BM25
queries in another. `query_index` joins a row back to its `query_id` inside
DuckDB. The hybrid file is one call per query, because a hybrid call pairs one
vector with one text query.

The extension scores vectors; it does not embed text. Embedding stays in a
client process so the model, prefix, and normalization are explicit and the
same vector can be used by Qdrant and DuckDB.

## 4. Build the extension

```bash
make configure
make debug
```

On Windows, `make` needs a Python on `PATH`. If `make configure` fails with
`Python was not found`, create the venv first and point `make` at it:

```powershell
uv venv configure/venv --python 3.13
uv pip install --python configure/venv/Scripts/python.exe duckdb==2.0.0
make configure
make debug PYTHON_BIN=configure/venv/Scripts/python.exe
```

The DuckDB CLI v2.0.0 asset is
`duckdb-cli-windows-amd64.tar.gz` (or the Linux equivalent) from the
[DuckDB v2.0.0 release](https://github.com/duckdb/duckdb/releases/tag/v2.0.0).

The loadable file is
`build/debug/extension/qdrant/qdrant.duckdb_extension`.

## 5. Load the local mirror and build HNSW

```bash
duckdb -unsigned demo/demo.duckdb \
  -cmd "LOAD 'build/debug/extension/qdrant/qdrant.duckdb_extension';" \
  -f demo/local.sql
```

This reads the 460 MB parquet file over HTTPS, casts `vector` to `FLOAT[384]`,
and builds a cosine HNSW index. The cast matters: duck-vss accepts only
`FLOAT[N]`, and the metric must match `array_cosine_distance`.

The HNSW index is not buffer-managed. Budget a few hundred MB and a minute for
the build.

## 6. Run the Qdrant side

Load the extension in every CLI invocation. `-unsigned` is required. The
snippets below are POSIX shell; in PowerShell, put the `LOAD` inline in
`-cmd "LOAD '...'"`.

```bash
LOAD="LOAD 'build/debug/extension/qdrant/qdrant.duckdb_extension';"

duckdb -unsigned demo/demo.duckdb -cmd "$LOAD" -f out/topk.sql
duckdb -unsigned demo/demo.duckdb -cmd "$LOAD" -f out/text_topk.sql
duckdb -unsigned demo/demo.duckdb -cmd "$LOAD" -f out/hybrid.sql
```

`out/topk.sql` calls `qdrant_batch` once for all 50 query vectors:

```sql
SELECT q.query_id, b.id, b.score, b.payload
FROM qdrant_batch(
  collection := 'esci-products',
  queries := '[[0.1,...],[0.3,...]]',
  mode := 'dense',
  "using" := 'dense',
  "limit" := 10,
  payload_fields := 'product_id,title,brand'
) b
JOIN (VALUES (0, 114), (1, 2891)) AS q(query_index, query_id)
  ON q.query_index = b.query_index;
```

`out/text_topk.sql` does the same in text mode, with the BM25 knobs spelled out:

```sql
qdrant_batch(
  collection := 'esci-products',
  queries := '["cashew granola bars","protein bars"]',
  mode := 'text',
  field := 'text',
  k := 1.2,
  b := 0.75,
  "limit" := 10,
  payload_fields := 'product_id,title,brand'
)
```

`out/hybrid.sql` pairs each query vector with its text and asks Qdrant to fuse:

```sql
qdrant_hybrid_text(
  collection := 'esci-products',
  dense_query := [...],
  text_query := 'cashew granola bars',
  fusion := 'rrf',
  dense_using := 'dense',
  text_field := 'text',
  prefetch_limit := 100,
  "limit" := 10,
  payload_fields := 'product_id,title,brand'
)
```

Filters compose with all three. For example, BM25 for one color, where `color`
has a keyword payload index:

```sql
SELECT id, round(score, 4) AS score, payload->>'title' AS title
FROM qdrant_text(
  collection := 'esci-products',
  query := 'granola bars',
  field := 'text',
  k := 1.2,
  b := 0.75,
  filter := '{"must":[{"key":"color","match":{"value":"Green"}}]}',
  "limit" := 10,
  payload_fields := 'product_id,title,brand,color'
);
```

## 7. Exact scan in DuckDB

```bash
duckdb -unsigned demo/demo.duckdb -cmd "$LOAD" -f out/exact.sql
```

Fifty full scans of 164,900 x 384 vectors. This is the ground truth for the
recall number in the next step.

## 8. Fuse ranks and evaluate

```bash
duckdb -unsigned demo/demo.duckdb -cmd "$LOAD" -f demo/fuse.sql
duckdb -unsigned demo/demo.duckdb -cmd "$LOAD" -f demo/evaluate.sql
```

`demo/fuse.sql` builds RRF over the dense and BM25 runs entirely in SQL, as the
local cross-check. `demo/evaluate.sql` reports nDCG@10 for each run, including
the server-side `hybrid` run, and dense recall@10 against the exact scan.

Read the output like this:

- Compare `dense`, `bm25`, `hybrid`, and `rrf` nDCG. The snapshot's BM25 index
  and the dataset's embeddings are independent signals; fusion often wins, but
  the actual numbers depend on the query set, so report the run you measured.
- `hybrid` fuses on Qdrant with `prefetch_limit := 100`; `rrf` fuses the two
  top-50 lists in DuckDB. They are different candidate pools, so expect close
  but not identical numbers.
- Compare dense recall against exact. Anything below 1.0 is HNSW
  approximation. Raise `hnsw_ef_search` or the collection's HNSW `ef` to trade
  latency for recall.
- nDCG is computed only over judged products. A product returned without a
  judgment counts as irrelevant, which is the ESCI convention.

## 9. Local HNSW plan and latency

```bash
duckdb -unsigned demo/demo.duckdb \
  -cmd "LOAD vss; LOAD 'build/debug/extension/qdrant/qdrant.duckdb_extension';" \
  -f out/hnsw.sql
```

`EXPLAIN ANALYZE` should show `HNSW_INDEX_SCAN` for the literal query. If it
shows a sequential scan, check the literal cast (`::FLOAT[384]`, not
`::DOUBLE[]`) and the metric (`cosine` index with `array_cosine_distance`).

## 10. Readable Qdrant results

```bash
duckdb -unsigned demo/demo.duckdb -cmd "$LOAD" -f out/demo.sql
```

Dense, BM25, and server-side hybrid results side by side for three queries, with
titles and brands read from the payload in SQL.

## Measured on Qdrant 1.20.2

Fifty judged ESCI queries, snapshot `esci-products.snapshot` (164,900 points),
DuckDB 2.0.0, extension built from this tree:

| Run | nDCG@10 |
| --- | --- |
| hybrid (`qdrant_hybrid_text`, RRF, prefetch 100) | 0.2962 |
| BM25 (`qdrant_batch` text mode) | 0.2858 |
| RRF in DuckDB over two top-50 lists | 0.2846 |
| dense (`qdrant_batch` dense mode) | 0.2396 |

Dense recall@10 against the exact DuckDB scan: 0.846.

The same 50 dense queries, same limit and payload fields, as one batch call and
as 50 separate `qdrant_query` calls, returned identical `(query_id, id, score)`
rows (0 differences). Wall clock per CLI process: 226 ms for the batch file and
3,080 ms for the 50-call file, with about 80 ms of startup in each. That is one
50-query RPC against 50 RPCs, not a benchmark.

## What this demonstrates

1. A published Qdrant snapshot restores into 1.20.2 and is immediately
   queryable through the extension with the model that built it.
2. Dense ANN, 1.20 BM25, and server-side dense+text fusion are all reachable
   from SQL, and ranks can also be fused where the data already is.
3. One `qdrant_batch` RPC answers many queries; DuckDB joins `query_index` back
   to its own query table.
4. DuckDB is the honest evaluator: exact distance, nDCG, recall, and joins
   with judgments, without a second pipeline.
5. The same rows and model work in duck-vss, so the local/offline path is not a
   different index with different behavior.
6. The remaining gaps are concrete: vectors out of Qdrant, deep batch pages
   above 128 per query, and the H&M catalog act. See
   [`docs/real-world.md`](../docs/real-world.md#roadmap).

## Limits

- Fifty queries is a demo, not a benchmark. Raise `--limit` for a real sweep.
  `qdrant_batch` removes the per-query RPC, but `limit` is still one page of at
  most 128 rows per query.
- The exact scan is only practical at this corpus size. At 10M x 1024, sample.
- Snapshot recovery requires a server that accepts the snapshot's format. The
  published ESCI snapshot requires 1.20+, which matches this demo.
- The H&M act (`Qdrant/hm_ecommerce_products`, BGE + SPLADE, facets, images) is
  not in this demo. It needs a reproducible ingest and query model first; it is
  a follow-up, not a blocker.

## Cleanup

```bash
docker compose -f demo/compose.yaml down -v
rm -rf demo/qdrant_storage demo/demo.duckdb demo/demo.duckdb.wal out
```
