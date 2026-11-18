-- Act 3: join retrieval to real judgments and a synthetic business table.
-- Run after out/topk.sql and out/hybrid.sql.
--
-- The gains are real ESCI judgments. `units` and `margin` are synthetic: they
-- come from a hash of product_id so the act is reproducible. They are not
-- business data and must be labeled that way wherever they are shown.

CREATE OR REPLACE TABLE judged AS
SELECT query_id, product_id, gain
FROM read_parquet('out/judged.parquet');

CREATE OR REPLACE TABLE synthetic_stock AS
SELECT
    product_id,
    CAST(hash(product_id) % 7 AS INTEGER) AS units,            -- 0 means out of stock
    round(4 + (hash(product_id) % 300) / 10.0, 2) AS margin    -- 4.00 to 33.90
FROM (SELECT DISTINCT payload->>'product_id' AS product_id FROM hybrid_topk);

CREATE OR REPLACE TABLE business_candidates AS
SELECT *
FROM (
    SELECT
        h.query_id,
        h.id,
        h.payload->>'product_id' AS product_id,
        h.payload->>'title' AS title,
        h.score,
        row_number() OVER (PARTITION BY h.query_id ORDER BY h.score DESC, h.id) AS rank_before,
        s.units,
        s.margin
    FROM hybrid_topk h
    JOIN synthetic_stock s ON s.product_id = h.payload->>'product_id'
) ranked
WHERE rank_before <= 10;

-- The rule: in stock and at least 8 points of margin.
CREATE OR REPLACE TABLE business_kept AS
SELECT * FROM business_candidates WHERE units > 0 AND margin >= 8;

CREATE OR REPLACE TABLE business_reranked AS
SELECT
    query_id,
    id,
    product_id,
    score,
    rank_before,
    row_number() OVER (PARTITION BY query_id ORDER BY rank_before) AS rank_after
FROM business_kept;

CREATE OR REPLACE TABLE business_summary AS
SELECT
    query_id,
    count(*) AS candidates,
    count(*) FILTER (WHERE units > 0 AND margin >= 8) AS kept,
    count(*) FILTER (WHERE NOT (units > 0 AND margin >= 8)) AS dropped
FROM business_candidates
GROUP BY query_id;

SELECT * FROM business_summary ORDER BY query_id LIMIT 5;

-- One query, before and after the rule. A dropped row has no rank_after.
SELECT
    c.rank_before,
    c.title,
    c.units,
    c.margin,
    coalesce(CAST(r.rank_after AS VARCHAR), 'dropped') AS rank_after,
    j.gain
FROM business_candidates c
LEFT JOIN business_reranked r ON r.query_id = c.query_id AND r.id = c.id
LEFT JOIN judged j ON j.query_id = c.query_id AND j.product_id = c.product_id
WHERE c.query_id = (SELECT min(query_id) FROM business_candidates)
ORDER BY c.rank_before;

-- nDCG@10 of the retrieved page before and after the rule.
CREATE OR REPLACE TABLE business_ndcg AS
WITH ranked AS (
    SELECT 'before' AS stage, query_id, product_id, rank_before AS rnk
    FROM business_candidates
    UNION ALL
    SELECT 'after', query_id, product_id, rank_after
    FROM business_reranked
),
dcg AS (
    SELECT r.stage, r.query_id, sum(coalesce(j.gain, 0) / log2(r.rnk + 1)) AS dcg
    FROM ranked r
    LEFT JOIN judged j ON j.query_id = r.query_id AND j.product_id = r.product_id
    GROUP BY r.stage, r.query_id
),
ideal AS (
    SELECT query_id, sum(gain / log2(rnk + 1)) AS idcg
    FROM (
        SELECT query_id, gain,
               row_number() OVER (PARTITION BY query_id ORDER BY gain DESC) AS rnk
        FROM judged
    )
    WHERE rnk <= 10
    GROUP BY query_id
)
SELECT
    d.stage,
    count(*) AS queries,
    round(avg(d.dcg / nullif(i.idcg, 0)), 4) AS ndcg_at_10
FROM dcg d
JOIN ideal i ON i.query_id = d.query_id
GROUP BY d.stage;

SELECT * FROM business_ndcg ORDER BY stage;
