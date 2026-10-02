#!/usr/bin/env python3
"""Create deterministic, configuration-grouped train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stage5_common import configuration_group, read_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/tested-models/dataset-manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tested-models/stage5/splits"))
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "pairs": {}}
    for frame in read_manifest(args.manifest):
        pair = str(frame["pair_id"].iloc[0])
        frame = frame.copy()
        frame["configuration_group"] = configuration_group(frame)
        groups = sorted(frame["configuration_group"].unique())
        # Hash ordering makes assignment independent of parquet row order.
        groups = sorted(groups, key=lambda group: (hashlib.sha256(f"{args.seed}|{pair}|{group}".encode()).hexdigest(), group))
        assignments = {group: ("test" if index % 10 < 2 else "validation" if index % 10 < 4 else "train") for index, group in enumerate(groups)}
        frame["split"] = frame["configuration_group"].map(assignments)
        for split in ("train", "validation", "test"):
            frame[frame["split"] == split].to_parquet(output / f"{pair}.{split}.parquet", index=False)
        summary["pairs"][pair] = {"rows": len(frame), "groups": len(groups), "counts": frame["split"].value_counts().to_dict()}
    (output / "split-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
