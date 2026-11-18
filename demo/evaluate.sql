-- nDCG@10 and recall@10 over the ESCI judgments.
-- Order: out/topk.sql, out/text_topk.sql, out/hybrid.sql, demo/fuse.sql, then this file.
--
-- Gains follow the ESCI paper: E=1.0, S=0.1, C=0.01, I=0.0.
-- A returned product with no judgment counts as irrelevant.

CREATE OR REPLACE TABLE judged AS
SELECT query_id, product_id, gain
FROM read_parquet('out/judged.parquet');

CREATE OR REPLACE TABLE runs AS
SELECT 'dense' AS run, query_id, id, payload->>'product_id' AS product_id, score
FROM qdrant_topk
UNION ALL
SELECT 'bm25', query_id, id, payload->>'product_id', score
FROM text_topk
UNION ALL
SELECT 'hybrid', query_id, id, payload->>'product_id', score
FROM hybrid_topk
UNION ALL
SELECT 'rrf', query_id, id, payload->>'product_id', score
FROM fused_topk;

CREATE OR REPLACE TABLE ranked AS
SELECT
    run,
    query_id,
    product_id,
    score,
    row_number() OVER (PARTITION BY run, query_id ORDER BY score DESC, id) AS rnk
FROM runs;

CREATE OR REPLACE TABLE dcg AS
SELECT
    r.run,
    r.query_id,
    sum(coalesce(j.gain, 0.0) / log2(r.rnk + 1)) AS dcg
FROM ranked r
LEFT JOIN judged j
    ON j.query_id = r.query_id AND j.product_id = r.product_id
WHERE r.rnk <= 10
GROUP BY r.run, r.query_id;

CREATE OR REPLACE TABLE ideal AS
SELECT query_id, sum(gain / log2(rnk + 1)) AS idcg
FROM (
    SELECT
        query_id,
        gain,
        row_number() OVER (PARTITION BY query_id ORDER BY gain DESC) AS rnk
    FROM judged
)
WHERE rnk <= 10
GROUP BY query_id;

CREATE OR REPLACE TABLE ndcg_at_10 AS
SELECT
    d.run,
    count(*) AS queries,
    round(avg(d.dcg / nullif(i.idcg, 0)), 4) AS ndcg_at_10
FROM dcg d
JOIN ideal i ON i.query_id = d.query_id
GROUP BY d.run;

SELECT * FROM ndcg_at_10 ORDER BY ndcg_at_10 DESC;

-- How often Qdrant's dense top-10 agrees with the exact DuckDB top-10.
CREATE OR REPLACE TABLE recall_at_10 AS
SELECT
    round(avg(hits / 10.0), 4) AS recall_at_10_vs_exact
FROM (
    SELECT
        r.query_id,
        count(e.product_id) AS hits
    FROM ranked r
    LEFT JOIN exact_topk e
        ON e.query_id = r.query_id AND e.product_id = r.product_id
    WHERE r.run = 'dense' AND r.rnk <= 10
    GROUP BY r.query_id
);

SELECT * FROM recall_at_10;
