from __future__ import annotations

import json
import sys

import label_data
import pandas as pd
import train_classifiers


def test_labels_do_not_flag_latency_when_training_baseline_is_missing() -> None:
    train = pd.DataFrame([{"model_id": "model", "accelerator": "H200", "tp": 1, "output_tok/sec": 10}])
    validation = train.copy()
    test = train.copy()

    labeled, rules = label_data.label_pair(train, validation, test)

    assert rules["train_latency_baseline"] is None
    assert labeled["train"]["label_latency"].tolist() == [0]
    assert labeled["train"]["within_tested_region"].tolist() == [1]


def test_training_reports_single_class_train_data_without_fitting(tmp_path, monkeypatch) -> None:
    pair = "model__h200"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"id": pair}]}))
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    frame = pd.DataFrame([{"within_tested_region": 1, "model_id": "model", "accelerator": "H200"}])
    for split in ("train", "validation", "test"):
        frame.to_parquet(labeled_dir / f"{pair}.{split}.parquet", index=False)
    (labeled_dir / "label-rules.json").write_text(json.dumps({pair: {}}))

    monkeypatch.setattr(train_classifiers, "read_manifest", lambda _: [pd.DataFrame({"pair_id": [pair]})])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_classifiers.py",
            "--manifest",
            str(manifest),
            "--labeled-dir",
            str(labeled_dir),
            "--output-dir",
            str(tmp_path / "models"),
        ],
    )

    assert train_classifiers.main() == 0
    report = json.loads((tmp_path / "models" / "training-report.json").read_text())
    assert report["pairs"][pair]["status"] == "skipped_single_class_train"
