from __future__ import annotations

import json

import discover_data


def test_discover_groups_all_accelerators_from_one_model_request(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"testedModels": ["model-a", "model-b"]}))
    calls: list[str] = []

    def fake_fetch_rows(session, api_url, model_id, accelerator=None, *, verify=True):
        calls.append(model_id)
        return (
            [{"accelerator": "H200"}, {"accelerator": "H200"}, {"accelerator": "B200"}]
            if model_id == "model-a"
            else [{"accelerator": "MI300X"}]
        )

    monkeypatch.setattr(discover_data, "fetch_rows", fake_fetch_rows)
    result = discover_data.discover(config, "https://source.invalid", [], tmp_path / "discovery.json")

    assert calls == ["model-a", "model-b"]
    assert [(pair["model_id"], pair["accelerator"], pair["records"]) for pair in result["pairs"]] == [
        ("model-a", "B200", 1),
        ("model-a", "H200", 2),
        ("model-b", "MI300X", 1),
    ]
    assert json.loads((tmp_path / "discovery.json").read_text())["pairs"] == result["pairs"]
