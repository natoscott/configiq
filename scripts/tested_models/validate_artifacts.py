#!/usr/bin/env python3
"""Validate a tested-model registry and its Triton artifact repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("data/tested-models/stage6/registry/registry.json"))
    parser.add_argument("--triton-dir", type=Path, default=Path("data/tested-models/stage6/registry/triton"))
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    if registry.get("schemaVersion") != 1:
        raise ValueError("unsupported registry schema")
    pairs = registry.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("registry contains no classifier pairs")

    ids = set()
    for pair in pairs:
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or pair_id in ids:
            raise ValueError(f"invalid or duplicate pair id: {pair_id}")
        ids.add(pair_id)
        if pair.get("classifier") != "xgboost":
            raise ValueError(f"unsupported classifier: {pair_id}")
        model_dir = args.triton_dir / pair_id
        config_path = model_dir / "config.pbtxt"
        model_path = model_dir / "1" / "xgboost.json"
        for path in (config_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = config_path.read_text()
        feature_count = len(pair.get("featureSchema", {}).get("feature_names", []))
        if 'backend: "fil"' not in config or 'is_classifier' not in config:
            raise ValueError(f"invalid FIL config: {config_path}")
        if f"dims: [ {feature_count} ]" not in config:
            raise ValueError(f"FIL input dimension does not match feature schema: {config_path}")

    print(f"validated {len(ids)} tested-model Triton artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
