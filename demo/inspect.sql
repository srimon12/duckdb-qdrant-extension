-- Act 1: what the snapshot restored.
-- Prerequisites: demo/qdrant_setup.py restored esci-products, and the CLI
-- loaded the extension and set qdrant_url.
--
-- The payload indexes, including the BM25 `text` index, are printed by
-- qdrant_setup.py from the REST API. This file reads them back through SQL.

SELECT
    status,
    points_count,
    indexed_vectors_count,
    segments_count,
    created_at
FROM qdrant_collection_info(collection := 'esci-products');

SELECT id, payload->>'title' AS title, payload->>'brand' AS brand
FROM qdrant_scroll(
    collection := 'esci-products',
    "limit" := 3,
    payload_fields := 'product_id,title,brand'
);
