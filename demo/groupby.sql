-- Act 5 cross-check: DuckDB GROUP BY over the local mirror.
-- Compare with the facet counts from demo/boundary.py.
-- Both sides count the same 164,900 rows.

SELECT brand, count(*) AS products
FROM products
WHERE brand IS NOT NULL
GROUP BY brand
ORDER BY products DESC
LIMIT 5;

SELECT color, count(*) AS products
FROM products
WHERE color IS NOT NULL
GROUP BY color
ORDER BY products DESC
LIMIT 5;
