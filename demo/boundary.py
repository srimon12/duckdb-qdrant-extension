#!/usr/bin/env python3
"""Act 5: the boundary. Facets and a formula query stay in Qdrant.

The extension does not expose facet counts, grouping, or formula/boost
expressions. This script calls both over the Qdrant REST API and prints the
result. `demo/groupby.sql` is the DuckDB cross-check for the facet counts.

Standard library only. The collection needs a dense vector named `dense`.

Example:
    python demo/boundary.py
"""

import argparse
import json
import urllib.request


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode() or "{}")


def seed_vector(base, collection):
    scroll = call(
        base,
        f"/collections/{collection}/points/scroll",
        {"limit": 1, "with_payload": False, "with_vector": ["dense"]},
    )
    return scroll["result"]["points"][0]["vector"]["dense"]


def facets(base, collection, field, limit):
    result = call(
        base,
        f"/collections/{collection}/facet",
        {"key": field, "limit": limit, "exact": True},
    )["result"]
    counts = ", ".join(f"{hit['value']}={hit['count']}" for hit in result["hits"])
    print(f"facet {field}: {counts}")


def formula(base, collection, vector, brand):
    result = call(
        base,
        f"/collections/{collection}/points/query",
        {
            "prefetch": {"query": vector, "using": "dense", "limit": 100},
            "query": {
                "formula": {
                    "sum": [
                        "$score",
                        {
                            "mult": [
                                0.5,
                                {"key": "brand", "match": {"value": brand}},
                            ]
                        },
                    ]
                }
            },
            "limit": 5,
            "with_payload": ["title", "brand"],
        },
    )["result"]
    print(f"formula: dense score + 0.5 when brand = {brand!r}")
    for point in result["points"]:
        payload = point.get("payload") or {}
        title = (payload.get("title") or "")[:70]
        print(f"  {point['score']:.4f}  {payload.get('brand', '')}  {title}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant REST URL")
    parser.add_argument("--collection", default="esci-products")
    parser.add_argument("--brand", default="Nature Valley")
    parser.add_argument("--facet-limit", type=int, default=5)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    for field in ("brand", "color"):
        facets(base, args.collection, field, args.facet_limit)
    formula(base, args.collection, seed_vector(base, args.collection), args.brand)
    print("These stay in Qdrant. demo/groupby.sql is the DuckDB cross-check.")


if __name__ == "__main__":
    main()
