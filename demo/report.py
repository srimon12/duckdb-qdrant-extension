#!/usr/bin/env python3
"""Write a static Markdown report from the demo's SQL results.

Runs the demo SQL that builds the result tables, then reads those tables and
writes one Markdown file. No notebook state and no hidden steps: everything in
the report comes from a query in this file or in demo/*.sql.

Prerequisites, in order:
    demo/qdrant_setup.py           restore the snapshot
    demo/embed.py --limit 50       query vectors and generated SQL
    out/topk.sql                   dense batch run
    out/text_topk.sql              BM25 batch run
    out/hybrid.sql                 server-side hybrid run
    out/exact.sql                  exact scan
    demo/local.sql                 local mirror with HNSW

Run:
    configure/venv/Scripts/python.exe demo/report.py
    # or: uv run --with duckdb==2.0.0 demo/report.py --out out/report.md

Writes out/report.md by default.
"""

import argparse
import datetime
import pathlib

import duckdb

EXTENSION = "build/debug/extension/qdrant/qdrant.duckdb_extension"


def load(con, extension, url):
    con.execute(f"LOAD '{extension}'")
    con.execute(f"SET qdrant_url = '{url}'")
    for name in ("decide.sql", "fuse.sql", "evaluate.sql"):
        con.execute((pathlib.Path("demo") / name).read_text(encoding="utf-8"))


def table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="demo/demo.duckdb")
    parser.add_argument("--extension", default=EXTENSION)
    parser.add_argument("--url", default="http://127.0.0.1:6334", help="Qdrant gRPC URL")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out/report.md"))
    args = parser.parse_args()

    con = duckdb.connect(
        args.db, config={"allow_unsigned_extensions": "true"}
    )
    load(con, args.extension, args.url)

    info = con.execute(
        "SELECT status, points_count, created_at FROM qdrant_collection_info"
        "(collection := 'esci-products')"
    ).fetchone()
    runs = con.execute(
        "SELECT run, queries, ndcg_at_10 FROM ndcg_at_10 ORDER BY ndcg_at_10 DESC"
    ).fetchall()
    recall = con.execute("SELECT recall_at_10_vs_exact FROM recall_at_10").fetchone()[0]
    batch = con.execute(
        "SELECT count(*) AS rows, count(DISTINCT query_id) AS queries FROM qdrant_topk"
    ).fetchone()
    fate = con.execute(
        "SELECT sum(candidates), sum(kept), sum(dropped) FROM business_summary"
    ).fetchone()
    business = con.execute(
        "SELECT stage, ndcg_at_10 FROM business_ndcg ORDER BY stage DESC"
    ).fetchall()

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# ESCI retrieval report

Generated {now} from DuckDB demo/demo.duckdb and Qdrant at {args.url}.

## Collection

Status `{info[0]}`, {info[1]} points, created at {info[2]}.

## Runs, 50 judged queries

{table(runs, ["run", "queries", "nDCG@10"])}

Dense recall@10 against the exact DuckDB scan: {recall}.

The dense run is {batch[0]} rows for {batch[1]} queries in one `qdrant_batch` RPC.

## Business filter (judgments real, stock and margin synthetic)

{table([fate], ["candidates", "kept", "dropped"])}

{table(business, ["stage", "nDCG@10"])}

The stock and margin columns are derived from a hash of `product_id`. They are a
reproducible stand-in for business data, not real inventory.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
