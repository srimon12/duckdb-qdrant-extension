-- The v1 batch limit. This file is expected to fail.
-- Run it to see the error message a caller gets before any RPC:
--   Invalid Input Error: limit 130 is above 128; qdrant_batch returns one page per query in v1
--
-- Prerequisites are the same as demo/smoke.sql.

SELECT count(*) AS rows
FROM qdrant_batch(
    collection := 'esci-products',
    queries := '[' || to_json(list_resize([0.02], 384, 0.02)) || ']',
    mode := 'dense',
    "using" := 'dense',
    "limit" := 130
);
