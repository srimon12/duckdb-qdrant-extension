#!/usr/bin/env python3
"""Restore a Qdrant snapshot over the REST API, then wait for the collection.

Qdrant has no gRPC snapshot recovery, so this step stays outside the DuckDB
extension. Everything after it uses the extension.

Example:
    python demo/qdrant_setup.py \
        --snapshot https://snapshots.qdrant.io/esci-products.snapshot

Standard library only.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(method, url, body=None, api_key=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode() or "{}"
    return json.loads(payload)


def wait_ready(base, api_key, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/readyz", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise SystemExit(f"qdrant at {base} did not become ready in {seconds}s")


def existing(base, collection, api_key):
    try:
        return call("GET", f"{base}/collections/{collection}", api_key=api_key)["result"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def summary(info):
    vectors = info["config"]["params"].get("vectors") or {}
    if "size" in vectors:
        vectors = {"": vectors}
    print(f"status:        {info.get('status')}")
    print(f"points:        {info.get('points_count')}")
    print(f"segments:      {info.get('segments_count')}")
    print(f"created_at:    {info.get('created_at')}")
    print("vectors:       " + ", ".join(
        f"{name or '(default)'}={params.get('size')} {params.get('distance')}"
        for name, params in vectors.items()
    ))
    indexes = info.get("payload_schema") or {}
    if indexes:
        print("payload index: " + ", ".join(sorted(indexes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant REST URL")
    parser.add_argument("--collection", default="esci-products")
    parser.add_argument(
        "--snapshot",
        default="https://snapshots.qdrant.io/esci-products.snapshot",
        help="snapshot location: https URL or file://",
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--wait", type=int, default=900, help="seconds to wait")
    parser.add_argument("--force", action="store_true", help="recover over a non-empty collection")
    parser.add_argument("--skip-recover", action="store_true")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    wait_ready(base, args.api_key, 120)
    info = existing(base, args.collection, args.api_key)

    if info and info.get("points_count") and not args.force:
        print(f"collection {args.collection!r} already has {info['points_count']} points")
        summary(info)
        print("pass --force to recover the snapshot again")
        return

    if not args.skip_recover:
        if info:
            print(f"collection {args.collection!r} exists; recovering into it")
        else:
            print(f"creating {args.collection!r} from {args.snapshot}")
        body = {"location": args.snapshot}
        try:
            call(
                "PUT",
                f"{base}/collections/{args.collection}/snapshots/recover",
                body,
                args.api_key,
                timeout=args.wait,
            )
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise SystemExit(f"snapshot recover failed: HTTP {error.code}: {detail}")

    deadline = time.time() + args.wait
    info = existing(base, args.collection, args.api_key)
    while time.time() < deadline:
        if info and info.get("status") == "green" and info.get("points_count"):
            summary(info)
            return
        time.sleep(2)
        info = existing(base, args.collection, args.api_key)

    if info:
        summary(info)
    raise SystemExit(f"collection {args.collection!r} did not become green in {args.wait}s")


if __name__ == "__main__":
    sys.exit(main())
