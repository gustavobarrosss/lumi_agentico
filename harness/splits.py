"""Separação dev/treino POR DOMÍNIO.

O conjunto escondido do desafio usa domínios fora do treino. Para que a nossa
validação meça a mesma coisa (generalização para domínios novos), o dev tem de
ser feito por domínio, não por amostra. E como vários domínios existem nas duas
capacidades (ex.: hockey, airline), a escolha é feita sobre a UNIÃO dos domínios:
um domínio de dev nunca aparece no treino de nenhuma capacidade.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path

from .paths import CAPABILITIES, CONFIGS

HF_TREE = "https://huggingface.co/api/datasets/ibm-research/VAKRA/tree/main/train/{cap}/input"
DEFAULT_SEED = 13
DEFAULT_DEV_FRACTION = 0.2
MIN_DEV_PER_CAP = 4


def fetch_hf_sizes() -> dict[int, dict[str, int]]:
    """Domínio -> bytes do arquivo de entrada, direto da API do HF (sem baixar dados)."""
    sizes: dict[int, dict[str, int]] = {}
    for cap_id, cap in CAPABILITIES.items():
        with urllib.request.urlopen(HF_TREE.format(cap=cap), timeout=30) as resp:
            rows = json.load(resp)
        sizes[cap_id] = {
            Path(r["path"]).stem: int(r.get("size", 0))
            for r in rows
            if r["type"] == "file" and r["path"].endswith(".json")
        }
    return sizes


def sizes_from_dir(train_root: Path) -> dict[int, dict[str, int]]:
    """Mesmo mapa, a partir de data/train já baixado."""
    sizes: dict[int, dict[str, int]] = {}
    for cap_id, cap in CAPABILITIES.items():
        files = sorted((train_root / cap / "input").glob("*.json"))
        sizes[cap_id] = {f.stem: f.stat().st_size for f in files}
    return sizes


def _rank(domain: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{domain}".encode()).hexdigest()


def make_splits(
    sizes: dict[int, dict[str, int]],
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SEED,
    min_dev_per_cap: int = MIN_DEV_PER_CAP,
) -> dict:
    """Escolhe os domínios de dev de forma determinística (depende só de seed e nomes)."""
    union = sorted({d for per_cap in sizes.values() for d in per_cap})
    ordered = sorted(union, key=lambda d: _rank(d, seed))
    n_dev = max(1, math.ceil(dev_fraction * len(union)))

    dev: list[str] = ordered[:n_dev]
    # Garante um mínimo de domínios de dev em cada capacidade.
    for cap_id, per_cap in sizes.items():
        while sum(1 for d in dev if d in per_cap) < min_dev_per_cap:
            nxt = next((d for d in ordered if d in per_cap and d not in dev), None)
            if nxt is None:
                raise ValueError(
                    f"capacidade {cap_id}: só há {len(per_cap)} domínios, "
                    f"não dá para ter {min_dev_per_cap} de dev"
                )
            dev.append(nxt)

    dev_set = set(dev)
    per_capability = {}
    for cap_id, per_cap in sizes.items():
        cap_dev = sorted(d for d in per_cap if d in dev_set)
        cap_train = sorted(d for d in per_cap if d not in dev_set)
        total = sum(per_cap.values()) or 1
        per_capability[str(cap_id)] = {
            "dev": cap_dev,
            "train": cap_train,
            "dev_share_of_input_bytes": round(sum(per_cap[d] for d in cap_dev) / total, 3),
        }

    return {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "criterio": "domínios inteiros; escolha pela união das capacidades 1 e 2 (sem vazamento entre elas)",
        "dev_domains": sorted(dev_set),
        "per_capability": per_capability,
    }


def check_no_leak(splits: dict) -> None:
    """Falha se algum domínio de dev aparecer no treino de qualquer capacidade."""
    dev = set(splits["dev_domains"])
    for cap_id, info in splits["per_capability"].items():
        leaked = dev & set(info["train"])
        if leaked:
            raise AssertionError(f"capacidade {cap_id}: domínios de dev no treino: {sorted(leaked)}")


def load_splits(path: Path | None = None) -> dict:
    path = path or CONFIGS / "splits.json"
    splits = json.loads(Path(path).read_text(encoding="utf-8"))
    check_no_leak(splits)
    return splits


def save_splits(splits: dict, path: Path | None = None) -> Path:
    path = path or CONFIGS / "splits.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
