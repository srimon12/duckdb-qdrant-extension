-- Reciprocal rank fusion of the dense and BM25 runs inside DuckDB.
-- RRF compares ranks, so raw cosine and BM25 scores never mix.
-- Run after out/topk.sql and out/text_topk.sql.

CREATE OR REPLACE TABLE fused_topk AS
WITH dense AS (
    SELECT
        query_id,
        id,
        row_number() OVER (PARTITION BY query_id ORDER BY score DESC, id) AS r
    FROM qdrant_topk
),
lex AS (
    SELECT
        query_id,
        id,
        row_number() OVER (PARTITION BY query_id ORDER BY score DESC, id) AS r
    FROM text_topk
)
SELECT
    coalesce(d.query_id, l.query_id) AS query_id,
    coalesce(d.id, l.id) AS id,
    coalesce(1.0 / (60 + d.r), 0) + coalesce(1.0 / (60 + l.r), 0) AS score,
    coalesce(dp.payload, lp.payload) AS payload
FROM dense d
FULL OUTER JOIN lex l
    ON d.query_id = l.query_id AND d.id = l.id
LEFT JOIN qdrant_topk dp
    ON dp.query_id = coalesce(d.query_id, l.query_id)
    AND dp.id = coalesce(d.id, l.id)
LEFT JOIN text_topk lp
    ON lp.query_id = coalesce(d.query_id, l.query_id)
    AND lp.id = coalesce(d.id, l.id);

SELECT
    query_id,
    payload->>'title' AS title,
    round(score, 6) AS rrf
FROM fused_topk
ORDER BY query_id, rrf DESC
LIMIT 20;
