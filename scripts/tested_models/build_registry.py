#!/usr/bin/env python3
"""Build the public tested-model registry from validated classifier artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/tested-models/stage5/models"))
    parser.add_argument("--triton-dir", type=Path, default=Path("data/tested-models/stage5/triton"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/stage6/registry"))
    parser.add_argument("--dashboard-dir", type=Path, default=Path("data/tested-models/stage6/registry/dashboard"))
    return parser.parse_args()


def system_id(accelerator: str) -> str:
    return {
        "h200": "h200_sxm",
        "b200": "b200_sxm",
        "b300": "b300_sxm",
        "mi300x": "mi300x",
        "mi355x": "mi355x",
    }.get(accelerator.lower(), accelerator.lower())


def build_registry(manifest_path: Path, model_dir: Path, triton_dir: Path, output_dir: Path, dashboard_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    pairs = {pair["id"]: pair for pair in manifest.get("pairs", [])}
    report = json.loads((model_dir / "training-report.json").read_text())
    registry_pairs = []
    triton_output = output_dir / "triton"
    if triton_output.exists():
        shutil.rmtree(triton_output)

    for pair_id, report_entry in sorted(report.get("pairs", {}).items()):
        if report_entry.get("status") != "trained":
            continue
        source = pairs.get(pair_id)
        if source is None:
            raise ValueError(f"trained pair is absent from dataset manifest: {pair_id}")
        model_path = model_dir / f"{pair_id}.json"
        metadata_path = model_dir / f"{pair_id}.metadata.json"
        triton_path = triton_dir / pair_id / "1" / "model.json"
        config_path = triton_dir / pair_id / "config.pbtxt"
        for path in (model_path, metadata_path, triton_path, config_path):
            if not path.exists():
                raise FileNotFoundError(path)
        dashboard_path = dashboard_dir / f"{pair_id}.json"
        if not dashboard_path.exists():
            raise FileNotFoundError(dashboard_path)
        shutil.copytree(triton_dir / pair_id, triton_output / pair_id)

        registry_pairs.append(
            {
                "id": pair_id,
                "modelId": source["model_id"],
                "systemId": system_id(source["accelerator"]),
                "classifier": "xgboost",
                "tritonModel": pair_id,
                "dashboardPath": f"dashboard/{pair_id}.json",
                "metrics": {
                    "validation": report_entry.get("validation"),
                    "test": report_entry.get("test"),
                },
                "thresholds": report_entry.get("thresholds", {}),
                "featureSchema": report_entry.get("feature_schema", {}),
                "artifactVersion": report_entry.get("model_format"),
            }
        )

    registry = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "pairs": registry_pairs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return registry


def main() -> int:
    args = parse_args()
    registry = build_registry(args.manifest, args.model_dir, args.triton_dir, args.output_dir, args.dashboard_dir)
    print(f"wrote {len(registry['pairs'])} tested-model pairs to {args.output_dir / 'registry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
