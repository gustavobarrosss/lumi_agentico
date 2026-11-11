"""Layout de pastas e nomes de capacidades. Único lugar onde esses nomes aparecem."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
VENDOR_VAKRA = ROOT / "vendor" / "vakra"
WORK = ROOT / "work"
DEV_DIR = WORK / "dev"
RUNS = ROOT / "runs"
CACHE = ROOT / ".cache"

# Só as capacidades 1 e 2 contam para o vencedor (plano da sprint).
CAPABILITIES: dict[int, str] = {
    1: "capability_1_bi_apis",
    2: "capability_2_dashboard_apis",
}
# Nome que o avaliador oficial espera em --capability_name.
EVALUATOR_NAMES: dict[int, str] = {
    1: "capability_bi_apis",
    2: "capability_dashboard_apis",
}


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Carrega KEY=VALUE do .env na raiz para os.environ SEM sobrescrever o que já está definido.
    Aceita comentários (# no começo da linha ou depois de espaço) e valores entre aspas."""
    import os

    path = path or ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ("'", '"') and value.count(value[0]) >= 2:
            value = value[1 : value.index(value[0], 1)]
        elif value.startswith("#"):
            value = ""  # valor vazio seguido de comentário
        else:
            value = value.split(" #", 1)[0].strip()
        key = key.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def official_config() -> dict:
    return load_json(CONFIGS / "official.json")
