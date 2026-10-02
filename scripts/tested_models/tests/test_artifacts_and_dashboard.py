from __future__ import annotations

import json
import sys

import build_dashboard
import build_registry
import pandas as pd
import pytest
import validate_artifacts


def test_registry_build_and_artifact_validation_require_complete_bundle(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": "model__h200", "model_id": "model", "accelerator": "H200"}]}))
    model_dir = tmp_path / "models"
    triton_dir = tmp_path / "triton"
    dashboard_dir = tmp_path / "dashboard"
    model_dir.mkdir()
    (model_dir / "training-report.json").write_text(
        json.dumps(
            {
                "pairs": {
                    "model__h200": {
                        "status": "trained",
                        "model_format": "xgboost-json",
                        "feature_schema": {"feature_names": ["tp"]},
                    }
                }
            }
        )
    )
    (model_dir / "model__h200.json").write_text("{}")
    (model_dir / "model__h200.metadata.json").write_text("{}")
    (triton_dir / "model__h200" / "1").mkdir(parents=True)
    (triton_dir / "model__h200" / "1" / "model.json").write_text("{}")
    (triton_dir / "model__h200" / "config.pbtxt").write_text("name: model__h200")
    dashboard_dir.mkdir()
    (dashboard_dir / "model__h200.json").write_text("{}")

    registry_dir = tmp_path / "registry"
    registry = build_registry.build_registry(manifest, model_dir, triton_dir, registry_dir, dashboard_dir)
    assert registry["pairs"][0]["systemId"] == "h200_sxm"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_artifacts.py",
            "--registry",
            str(registry_dir / "registry.json"),
            "--triton-dir",
            str(registry_dir / "triton"),
        ],
    )
    assert validate_artifacts.main() == 0

    (registry_dir / "triton" / "model__h200" / "config.pbtxt").unlink()
    with pytest.raises(FileNotFoundError):
        validate_artifacts.main()


def test_dashboard_payload_contains_only_sanitized_derived_data(tmp_path, monkeypatch) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair, "model_id": "model", "accelerator": "H200"}]}))
    labeled_dir = tmp_path / "labeled"
    model_dir = tmp_path / "models"
    labeled_dir.mkdir()
    model_dir.mkdir()
    frame = pd.DataFrame(
        [
            {"concurrency": 1, "output_tok/sec": 10, "request_latency_median": 2, "label_knee": 0, "within_tested_region": 1, "runtime_args": "https://secret.invalid"},
            {"concurrency": 2, "output_tok/sec": 20, "request_latency_median": 3, "label_knee": 1, "within_tested_region": 0, "runtime_args": "secret"},
        ]
    )
    for split in ("train", "validation", "test"):
        frame.to_parquet(labeled_dir / f"{pair}.{split}.parquet", index=False)
    (model_dir / f"{pair}.metadata.json").write_text(
        json.dumps({"status": "trained", "feature_schema": {"feature_names": ["tp"]}, "validation": {}, "test": {}})
    )
    (model_dir / f"{pair}.json").write_text("{}")

    class FakeModel:
        feature_importances_: list[float]

        def __init__(self):
            self.feature_importances_ = [1.0]

        def load_model(self, path):
            assert path == model_dir / f"{pair}.json"

    monkeypatch.setattr(build_dashboard, "XGBClassifier", FakeModel)
    output_dir = tmp_path / "dashboard"
    assert build_dashboard.build_dashboard(manifest, labeled_dir, model_dir, output_dir) == 1

    payload = json.loads((output_dir / f"{pair}.json").read_text())
    serialized = json.dumps(payload)
    assert "runtime_args" not in serialized
    assert "secret.invalid" not in serialized
    assert payload["records"] == 6
    assert payload["labelDistribution"] == {"within": 3, "outside": 3}
