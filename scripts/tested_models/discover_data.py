#!/usr/bin/env python3
"""Discover available ground-truth model/hardware pairs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import requests
import urllib3
from dataset_common import (
    fetch_rows,
    load_required_models,
    pair_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("public/config.json"))
    parser.add_argument("--api-url", default=os.getenv("PERF_DATA_API_URL"))
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for the internal ground-truth API.",
    )
    parser.add_argument(
        "--accelerator",
        action="append",
        dest="accelerators",
        help="Optional accelerator filter; may be repeated. Defaults to every accelerator returned by the source.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/tested-models/discovery.json"))
    return parser.parse_args()


def discover(config: Path, api_url: str, accelerators: list[str], output: Path, *, insecure: bool = False) -> dict:
    models = load_required_models(config)
    pairs: list[dict] = []
    session = requests.Session()

    for model_id in models:
        try:
            rows = fetch_rows(session, api_url, model_id, verify=not insecure)
        except requests.HTTPError as exc:
            print(f"request_error {model_id}: HTTP {exc.response.status_code if exc.response is not None else 'error'}")
            continue
        except requests.RequestException:
            print(f"request_error {model_id}: request failed")
            continue
        except ValueError:
            print(f"invalid_source_data {model_id}")
            continue

        grouped: dict[str, list[dict]] = {}
        for row in rows:
            accelerator = row.get("accelerator")
            if not isinstance(accelerator, str) or not accelerator.strip():
                continue
            if accelerators and accelerator not in accelerators:
                continue
            grouped.setdefault(accelerator, []).append(row)

        for accelerator, accelerator_rows in sorted(grouped.items()):
            item = {
                "id": pair_id(model_id, accelerator),
                "model_id": model_id,
                "accelerator": accelerator,
                "status": "available" if accelerator_rows else "empty",
                "records": len(accelerator_rows),
            }
            pairs.append(item)
            print(f"{item['status']:9} {model_id} on {accelerator}: {item['records']} records")

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "ground_truth_api",
        "config": str(config),
        "required_models": models,
        "accelerators": sorted({pair["accelerator"] for pair in pairs}),
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    args = parse_args()
    if not args.api_url:
        raise SystemExit("Provide --api-url or PERF_DATA_API_URL; endpoint values are never stored in manifests.")
    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    accelerators = [value.strip() for value in args.accelerators or [] if value.strip()]
    result = discover(args.config, args.api_url, accelerators, args.output, insecure=args.insecure)
    available = sum(pair["status"] == "available" for pair in result["pairs"])
    print(f"Discovered {available} available pairs; missing/empty pairs are recorded in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
