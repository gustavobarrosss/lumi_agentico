"""Monta o conjunto de dev para o runner/avaliador OFICIAIS, sem alterar o código deles.

Por que isto existe (achados da leitura do código da VAKRA):

1. O benchmark_runner.py só lê `data/test/capability_*/input/` e não cai para o treino.
   Ele aceita as variáveis CAPABILITY_1_DIR / CAPABILITY_2_DIR, então apontamos para
   `work/dev/<capacidade>/` (cópia dos domínios de dev tirados de `data/train`).

2. A capacidade 1 chama `get_data(tool_universe_id=<uuid>)` e o servidor resolve o UUID
   em `environment/configs/mcp_tool_universe_id_mapping.yaml`, arquivo gerado a partir
   da saída do conjunto de TESTE. Os UUIDs de treino não estão lá (o UUID do exemplo
   de treino da própria VAKRA não consta), então o runner falharia com
   "Universe switch failed". Geramos um mapa estendido (base + UUIDs de dev) com a
   MESMA lógica de `generate_task_1_tool_universe_mapping.py` e o montamos no container
   por cima de `/app/environment/configs` via docker-compose.override.yml.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from .paths import CAPABILITIES, DEV_DIR, VENDOR_VAKRA

MAPPING_NAME = "mcp_tool_universe_id_mapping.yaml"

# Cópias das listas do gerador oficial (generate_task_1_tool_universe_mapping.py).
SELECTION_ONLY_TOOLS = {
    "select_data_equal_to", "select_data_not_equal_to",
    "select_data_greater_than", "select_data_less_than",
    "select_data_greater_than_equal_to", "select_data_less_than_equal_to",
    "select_data_contains", "select_data_like",
    "sort_data_ascending", "sort_data_descending",
    "compute_data_min", "compute_data_max", "compute_data_sum",
    "compute_data_mean", "compute_data_count", "compute_data_std",
    "compute_data_argmin", "compute_data_argmax",
    "truncate", "transform_data_to_substring",
    "transform_data_to_absolute_value", "transform_data_to_datetime_part",
}
SLOT_FILLING_ONLY_TOOLS = {
    "filter_data", "retrieve_data", "sort_data", "aggregate_data", "transform_data",
}


def stage_dev(train_root: Path, splits: dict, out_dir: Path = DEV_DIR) -> dict[int, dict]:
    """Copia input/ e output/ dos domínios de dev. Retorna contagens por capacidade."""
    report: dict[int, dict] = {}
    for cap_id, cap in CAPABILITIES.items():
        wanted = splits["per_capability"][str(cap_id)]["dev"]
        copied, missing, n_samples = [], [], 0
        for domain in wanted:
            src_in = train_root / cap / "input" / f"{domain}.json"
            src_out = train_root / cap / "output" / f"{domain}.json"
            if not (src_in.exists() and src_out.exists()):
                missing.append(domain)
                continue
            for kind, src in (("input", src_in), ("output", src_out)):
                dst = out_dir / cap / kind / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            n_samples += len(json.loads(src_in.read_text(encoding="utf-8")))
            copied.append(domain)
        report[cap_id] = {"domains": copied, "missing": missing, "samples": n_samples}
    return report


def _first_turn_tool_calls(record: dict) -> tuple[dict | None, list[dict]]:
    """Primeiro turno do GT e suas tool calls. `sequence` aparece como dict (dataset) ou
    como lista de um dict (sample_data da VAKRA); aceitamos os dois."""
    turns = record.get("output") or []
    if not turns:
        return None, []
    first = turns[0]
    seq = first.get("sequence", {})
    if isinstance(seq, list):
        calls: list[dict] = []
        for item in seq:
            calls.extend(item.get("tool_call", []))
        return first, calls
    return first, list(seq.get("tool_call", []))


def mapping_entries(records: list[dict]) -> tuple[dict, list[str]]:
    """Mesma lógica do gerador oficial. Retorna (entradas, avisos)."""
    entries: dict[str, dict] = {}
    warnings: list[str] = []
    for rec in records:
        uuid = rec["uuid"]
        if uuid in entries:
            continue
        first, calls = _first_turn_tool_calls(rec)
        if first is None or not calls:
            warnings.append(f"{uuid}: sem tool_call no GT, ignorado")
            continue
        init = next((c for c in calls if c["name"] == "initialize_active_data"), None)
        if init is None:
            warnings.append(f"{uuid}: sem initialize_active_data, ignorado")
            continue
        server_type = "slot_filling"
        for call in calls:
            if call["name"] in SELECTION_ONLY_TOOLS:
                server_type = "selection"
                break
            if call["name"] in SLOT_FILLING_ONLY_TOOLS:
                break
        entries[uuid] = {
            "domain": rec["domain"],
            "server_type": server_type,
            "init_args": init["arguments"],
            "query": first["query"],
        }
    return entries, warnings


def build_dev_configs(vendor_configs: Path, dev_output_dir: Path, out_configs: Path) -> dict:
    """Cópia de environment/configs com o mapa estendido pelos UUIDs de dev (capacidade 1)."""
    out_configs.mkdir(parents=True, exist_ok=True)
    for f in vendor_configs.iterdir():
        if f.is_file() and f.name != MAPPING_NAME:
            shutil.copy2(f, out_configs / f.name)

    base_text = (vendor_configs / MAPPING_NAME).read_text(encoding="utf-8")
    base = yaml.safe_load(base_text) or {}

    records: list[dict] = []
    for f in sorted(dev_output_dir.glob("*.json")):
        records.extend(json.loads(f.read_text(encoding="utf-8")))
    entries, warnings = mapping_entries(records)

    clash = set(entries) & set(base)
    if clash:
        raise ValueError(f"{len(clash)} UUIDs de dev já existem no mapa base, ex.: {sorted(clash)[:3]}")

    extra = yaml.dump(entries, default_flow_style=False, allow_unicode=True)
    # Acrescenta ao texto original: preserva o mapa base byte a byte.
    merged = base_text if base_text.endswith("\n") else base_text + "\n"
    (out_configs / MAPPING_NAME).write_text(merged + extra, encoding="utf-8")
    return {"base_uuids": len(base), "dev_uuids": len(entries), "warnings": warnings}


def write_compose_override(vendor_dir: Path, configs_dir: Path) -> Path:
    """docker-compose.override.yml (não versionado) trocando só o volume de configs da capacidade 1.
    O Compose mescla volumes pelo ponto de montagem, então os demais continuam como estão."""
    override = {
        "services": {
            "capability_1_bi_apis": {
                "volumes": [
                    {
                        "type": "bind",
                        "source": configs_dir.resolve().as_posix(),
                        "target": "/app/environment/configs",
                        "read_only": True,
                    }
                ]
            }
        }
    }
    path = vendor_dir / "docker-compose.override.yml"
    header = "# Gerado por `harness stage-dev`. Não editar; não versionar.\n"
    path.write_text(header + yaml.dump(override, sort_keys=False), encoding="utf-8")
    return path


def stage_all(train_root: Path, splits: dict, vendor_dir: Path = VENDOR_VAKRA, out_dir: Path = DEV_DIR) -> dict:
    report = {"copied": stage_dev(train_root, splits, out_dir)}
    cap1 = CAPABILITIES[1]
    report["mapping"] = build_dev_configs(
        vendor_dir / "environment" / "configs", out_dir / cap1 / "output", out_dir / "configs"
    )
    report["override"] = str(write_compose_override(vendor_dir, out_dir / "configs"))
    return report
