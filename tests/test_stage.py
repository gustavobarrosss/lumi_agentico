import json

import pytest
import yaml

from harness import stage

# Registro de exemplo copiado de sample_data/capability_1_bi_apis/output.json da VAKRA
# (aí `sequence` é uma LISTA com um dict; no dataset do HF é um dict).
SAMPLE_LIST_SHAPE = {
    "uuid": "8a751d8b-bf2b-4cdd-bdfe-54efdb1c4be8",
    "domain": "beer_factory",
    "output": [{
        "turn_id": 0,
        "query": "How many breweries are located in North America?",
        "answer": ["23"],
        "sequence": [{"tool_call": [
            {"name": "initialize_active_data", "arguments": {
                "condition_sequence": [],
                "alias_to_table_dict": {"": {"original_table_name": "rootbeerbrand", "modified_table_name": "rootbeerbrand"}},
                "database_path": "/x/beer_factory.sqlite"}, "label": "starting_table_var"},
            {"name": "select_data_equal_to", "arguments": {"data_source": "$starting_table_var.data_path$",
                                                           "key_name": "Country", "value": "United States"}, "label": "F0"},
            {"name": "compute_data_count", "arguments": {"data_source": "$F0.data_path$", "key_name": "BrandID", "distinct": False}, "label": "COUNT"},
        ]}],
    }],
}


def _dict_shape(uuid: str, domain: str, tool: str) -> dict:
    return {"uuid": uuid, "domain": domain, "output": [{
        "turn_id": 0, "query": f"pergunta {uuid}", "answer": ["1"],
        "sequence": {"tool_call": [
            {"name": "initialize_active_data", "arguments": {"database_path": "/x.sqlite", "condition_sequence": [], "alias_to_table_dict": {}}},
            {"name": tool, "arguments": {}},
        ]}}]}


def test_mapping_handles_both_sequence_shapes():
    dict_rec = _dict_shape("u-dict", "genes", "filter_data")
    entries, warnings = stage.mapping_entries([SAMPLE_LIST_SHAPE, dict_rec])
    assert not warnings
    assert entries["8a751d8b-bf2b-4cdd-bdfe-54efdb1c4be8"]["server_type"] == "selection"
    assert entries["8a751d8b-bf2b-4cdd-bdfe-54efdb1c4be8"]["domain"] == "beer_factory"
    assert entries["u-dict"]["server_type"] == "slot_filling"
    assert entries["u-dict"]["init_args"]["database_path"] == "/x.sqlite"


def test_mapping_skips_records_without_init_call():
    rec = {"uuid": "u", "domain": "d", "output": [{"query": "q", "sequence": {"tool_call": [{"name": "select_data_like", "arguments": {}}]}}]}
    entries, warnings = stage.mapping_entries([rec])
    assert entries == {} and "initialize_active_data" in warnings[0]


def _fake_vendor(tmp_path):
    cfg = tmp_path / "vendor" / "environment" / "configs"
    cfg.mkdir(parents=True)
    base = "base-uuid-1:\n  domain: language_corpus\n  server_type: selection\n"
    (cfg / stage.MAPPING_NAME).write_text(base, encoding="utf-8")
    (cfg / "generate_task_1_tool_universe_mapping.py").write_text("# gerador\n", encoding="utf-8")
    return cfg, base


def test_build_dev_configs_extends_without_touching_base(tmp_path):
    cfg, base = _fake_vendor(tmp_path)
    out_dir = tmp_path / "dev_output"
    out_dir.mkdir()
    (out_dir / "beer_factory.json").write_text(json.dumps([SAMPLE_LIST_SHAPE]), encoding="utf-8")

    info = stage.build_dev_configs(cfg, out_dir, tmp_path / "staged")
    merged_text = (tmp_path / "staged" / stage.MAPPING_NAME).read_text(encoding="utf-8")
    merged = yaml.safe_load(merged_text)

    assert merged_text.startswith(base)  # base preservado byte a byte
    assert set(merged) == {"base-uuid-1", "8a751d8b-bf2b-4cdd-bdfe-54efdb1c4be8"}
    assert info == {"base_uuids": 1, "dev_uuids": 1, "warnings": []}
    assert (tmp_path / "staged" / "generate_task_1_tool_universe_mapping.py").exists()  # resto copiado


def test_build_dev_configs_rejects_uuid_clash(tmp_path):
    cfg, _ = _fake_vendor(tmp_path)
    out_dir = tmp_path / "dev_output"
    out_dir.mkdir()
    (out_dir / "x.json").write_text(json.dumps([_dict_shape("base-uuid-1", "genes", "filter_data")]), encoding="utf-8")
    with pytest.raises(ValueError, match="já existem"):
        stage.build_dev_configs(cfg, out_dir, tmp_path / "staged")


def test_stage_dev_copies_only_dev_domains(tmp_path):
    train = tmp_path / "train"
    for cap in ("capability_1_bi_apis", "capability_2_dashboard_apis"):
        for kind in ("input", "output"):
            for dom in ("genes", "books"):
                p = train / cap / kind / f"{dom}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps([{"uuid": f"{cap}-{dom}"}]), encoding="utf-8")
    splits = {"per_capability": {"1": {"dev": ["genes"]}, "2": {"dev": ["genes", "ausente"]}}}

    report = stage.stage_dev(train, splits, tmp_path / "dev")

    assert report[1] == {"domains": ["genes"], "missing": [], "samples": 1}
    assert report[2]["missing"] == ["ausente"]
    assert (tmp_path / "dev" / "capability_1_bi_apis" / "input" / "genes.json").exists()
    assert (tmp_path / "dev" / "capability_1_bi_apis" / "output" / "genes.json").exists()
    assert not (tmp_path / "dev" / "capability_1_bi_apis" / "input" / "books.json").exists()


def test_compose_override_replaces_only_configs_mount(tmp_path):
    path = stage.write_compose_override(tmp_path, tmp_path / "work" / "dev" / "configs")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(data["services"]) == ["capability_1_bi_apis"]
    (vol,) = data["services"]["capability_1_bi_apis"]["volumes"]
    assert vol["target"] == "/app/environment/configs" and vol["read_only"] is True
    assert vol["source"].endswith("work/dev/configs")
