-- Local mirror of the ESCI products, for exact scans and duck-vss.
-- Run once in demo/demo.duckdb:
--   duckdb -unsigned demo/demo.duckdb -f demo/local.sql
--
-- 164,900 rows x 384 dimensions. The parquet file is about 460 MB.

INSTALL httpfs;
LOAD httpfs;
LOAD vss;

-- Persistence is experimental: the flag gates index creation in a file database.
SET hnsw_enable_experimental_persistence = true;

CREATE OR REPLACE TABLE products AS
SELECT
    id,
    product_id,
    title,
    brand,
    color,
    text,
    "vector"::FLOAT[384] AS emb
FROM read_parquet(
    'https://huggingface.co/datasets/Qdrant/esci-products-multilingual-e5-small-embeddings/resolve/main/data/train-00000-of-00001.parquet'
);

-- Cosine, because the dataset used cosine. Metric and distance function must match.
CREATE INDEX products_hnsw ON products USING HNSW (emb) WITH (metric = 'cosine');

SELECT
    (SELECT count(*) FROM products) AS points,
    (SELECT count(DISTINCT product_id) FROM products) AS products,
    (SELECT count(*) FROM products WHERE emb IS NULL) AS null_vectors;
