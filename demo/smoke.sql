-- Bind-level smoke test for the 1.20.2 functions.
-- Catches argument-order and signature mistakes at bind time, before any
-- result quality matters.
--
-- Prerequisites: Qdrant at the session qdrant_url with the demo collection
-- (384-dimensional unnamed dense vector `dense`, BM25 `text` payload index).
--
--   duckdb -unsigned -cmd "LOAD 'build/debug/extension/qdrant/qdrant.duckdb_extension'; SET qdrant_url='http://127.0.0.1:6334';" -f demo/smoke.sql
--
-- The 384-dimension vector is built in SQL so the file stays readable.

-- qdrant_text with k/b.
SELECT 'qdrant_text k/b' AS check, count(*) AS rows
FROM qdrant_text(
    collection := 'esci-products',
    query := 'cashew granola bars',
    field := 'text',
    k := 1.2,
    b := 0.75,
    "limit" := 3
);

-- qdrant_batch, dense mode: two queries in one RPC.
SELECT 'qdrant_batch dense' AS check, count(*) AS rows,
       count(DISTINCT query_index) AS queries
FROM qdrant_batch(
    collection := 'esci-products',
    queries := '[' || to_json(list_resize([0.02], 384, 0.02)) || ','
        || to_json(list_resize([0.02], 384, 0.02)) || ']',
    mode := 'dense',
    "using" := 'dense',
    "limit" := 2
);

-- qdrant_batch, text mode: the k/b names match qdrant_text.
SELECT 'qdrant_batch text' AS check, count(*) AS rows,
       count(DISTINCT query_index) AS queries
FROM qdrant_batch(
    collection := 'esci-products',
    queries := '["cashew granola bars","protein bars"]',
    mode := 'text',
    field := 'text',
    k := 1.2,
    b := 0.75,
    "limit" := 2
);

-- qdrant_hybrid_text: dense prefetch plus text prefetch, fused on the server.
SELECT 'qdrant_hybrid_text' AS check, count(*) AS rows
FROM qdrant_hybrid_text(
    collection := 'esci-products',
    dense_query := list_resize([0.02], 384, 0.02)::DOUBLE[],
    text_query := 'cashew granola bars',
    fusion := 'rrf',
    dense_using := 'dense',
    text_field := 'text',
    prefetch_limit := 20,
    "limit" := 3
);

-- Paging past 128 on the fused query: 150 rows, 150 distinct ids.
SELECT 'hybrid paging' AS check, count(*) AS rows, count(DISTINCT id) AS distinct_ids
FROM qdrant_hybrid_text(
    collection := 'esci-products',
    dense_query := list_resize([0.02], 384, 0.02)::DOUBLE[],
    text_query := 'cashew granola bars',
    fusion := 'rrf',
    dense_using := 'dense',
    text_field := 'text',
    prefetch_limit := 100,
    "limit" := 150
);
